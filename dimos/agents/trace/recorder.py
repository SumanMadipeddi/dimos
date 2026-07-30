# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""memory2-backed persistence for mission-trace events.

:class:`MissionTraceRecorder` subscribes to the shared ``mission_trace``
channel (fed by :class:`~dimos.agents.mcp.mcp_client.McpClient` and
:class:`~dimos.agents.mcp.mcp_server.McpServer`) and appends every
:class:`~dimos.agents.trace.types.MissionEvent` into a memory2
``mission_events`` stream.

It subclasses :class:`~dimos.memory2.module.MemoryModule` (not the sensor
``Recorder``) on purpose: the sensor recorder disables itself under global
replay so replayed sensor data isn't re-recorded, but mission execution during
a replay run is genuinely *new* and must still be captured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reactivex.disposable import Disposable

from dimos.agents.trace.attribution import FailureAttribution, attribute_failures
from dimos.agents.trace.types import EventType, MissionEvent
from dimos.core.core import rpc
from dimos.core.stream import In
from dimos.memory2.module import MemoryModule, MemoryModuleConfig
from dimos.utils.logging_config import get_run_log_dir, setup_logger

if TYPE_CHECKING:
    from dimos.memory2.stream import Stream

logger = setup_logger()

MISSION_EVENTS_STREAM = "mission_events"
# Sentinel default; when unchanged, the recorder resolves the db into the
# current per-run log directory so a run looks like:
#   <run-dir>/{main.jsonl, mission_trace.db}
_DEFAULT_DB_PATH = "mission_trace.db"


class MissionTraceRecorderConfig(MemoryModuleConfig):
    db_path: str | Path = _DEFAULT_DB_PATH
    stream_name: str = MISSION_EVENTS_STREAM


class MissionTraceRecorder(MemoryModule):
    """Persist mission-trace events into a memory2 ``mission_events`` stream."""

    config: MissionTraceRecorderConfig
    mission_trace: In[MissionEvent]

    _unsubscribe: Any = None

    @rpc
    def start(self) -> None:
        super().start()
        self._resolve_db_path()
        # Touch the store/stream so the db + stream exist immediately (even
        # before the first event), which makes replay/e2e inspection reliable.
        self._stream()
        logger.info(
            "MissionTraceRecorder recording",
            db_path=str(self.config.db_path),
            replay=self.config.g.replay,
        )
        self._unsubscribe = self.mission_trace.subscribe(self._on_event)
        self.register_disposable(Disposable(self._unsubscribe))

    def _resolve_db_path(self) -> None:
        """Place the trace db in the per-run log dir when using the default.

        An explicitly configured ``db_path`` (e.g. a test's ``tmp_path``) is
        respected. The default is only rebased into the run log directory when
        one can be resolved.

        The recorder runs in a forkserver worker where the in-process
        ``get_run_log_dir()`` global is unset; the CLI/runtime exports
        ``DIMOS_RUN_LOG_DIR`` (inherited by child processes), so fall back to it
        to keep the db next to ``main.jsonl`` instead of the project root.
        """
        if str(Path(self.config.db_path).name) != _DEFAULT_DB_PATH:
            return
        run_log_dir = get_run_log_dir()
        if run_log_dir is None:
            env_dir = os.environ.get("DIMOS_RUN_LOG_DIR")
            if env_dir:
                run_log_dir = Path(env_dir)
        if run_log_dir is not None:
            self.config.db_path = Path(run_log_dir) / _DEFAULT_DB_PATH

    def _stream(self) -> Stream[MissionEvent]:
        return self.store.stream(self.config.stream_name, MissionEvent)

    def _on_event(self, event: MissionEvent) -> None:
        """Append an event to the stream, keyed by its own timestamp + tags.

        Runs on the transport delivery thread; SQLite is opened with
        ``check_same_thread=False`` and events are low-rate, so a direct append
        is safe. Never lets a persistence error propagate back onto the bus.
        """
        try:
            self._stream().append(event, ts=event.timestamp, tags=event.tags())
        except Exception:
            logger.exception("failed to record mission event", event_type=event.event_type)

    # --- Query surface -------------------------------------------------------

    def events(
        self,
        *,
        mission_id: str | None = None,
        event_type: EventType | str | None = None,
    ) -> list[MissionEvent]:
        """Return recorded events, optionally filtered, in chronological order."""
        stream: Stream[MissionEvent] = self._stream()
        if mission_id is not None:
            stream = stream.tags(mission_id=mission_id)
        if event_type is not None:
            value = event_type.value if isinstance(event_type, EventType) else event_type
            stream = stream.tags(event_type=value)
        return [obs.data for obs in stream.order_by("ts")]

    def attributions(self, *, mission_id: str | None = None) -> list[FailureAttribution]:
        """Compute deterministic failure attributions from recorded events."""
        return attribute_failures(self.events(mission_id=mission_id))


mission_trace_recorder = MissionTraceRecorder.blueprint
