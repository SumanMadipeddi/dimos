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

"""Recorder persistence tests (Test Level: recorder, no robot/LLM/replay data).

Events are fed directly to :meth:`MissionTraceRecorder._on_event`, which is what
the transport subscription calls — this exercises the full memory2 persistence
and query path without needing a wired transport.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dimos.agents.trace.attribution import FailureCategory
from dimos.agents.trace.recorder import MISSION_EVENTS_STREAM, MissionTraceRecorder
from dimos.agents.trace.types import (
    EventSource,
    EventType,
    MissionContext,
    MissionEvent,
    Status,
)
from dimos.core.global_config import global_config


def _event(
    event_type: EventType,
    *,
    mission_id: str,
    ts: float,
    error_code: str | None = None,
    tool_name: str | None = None,
) -> MissionEvent:
    ctx = MissionContext(run_id="run-1", mission_id=mission_id, turn_id=1)
    return MissionEvent(
        event_type=event_type,
        source=EventSource.CLIENT,
        run_id=ctx.run_id,
        mission_id=ctx.mission_id,
        turn_id=ctx.turn_id,
        timestamp=ts,
        error_code=error_code,
        tool_name=tool_name,
        status=Status.FAILURE if event_type is EventType.TOOL_FAILED else None,
    )


@pytest.fixture
def recorder(tmp_path: Path) -> Iterator[MissionTraceRecorder]:
    rec = MissionTraceRecorder(db_path=str(tmp_path / "mission.db"))
    try:
        yield rec
    finally:
        rec.stop()


def test_append_and_query_by_mission(recorder: MissionTraceRecorder) -> None:
    recorder._on_event(_event(EventType.MISSION_STARTED, mission_id="m-1", ts=1.0))
    recorder._on_event(_event(EventType.INPUT_RECEIVED, mission_id="m-1", ts=2.0))

    events = recorder.events(mission_id="m-1")
    assert [e.event_type for e in events] == [
        EventType.MISSION_STARTED,
        EventType.INPUT_RECEIVED,
    ]


def test_query_by_event_type(recorder: MissionTraceRecorder) -> None:
    recorder._on_event(
        _event(EventType.TOOL_STARTED, mission_id="m-1", ts=1.0, tool_name="observe")
    )
    recorder._on_event(
        _event(EventType.TOOL_COMPLETED, mission_id="m-1", ts=2.0, tool_name="observe")
    )
    recorder._on_event(_event(EventType.TOOL_STARTED, mission_id="m-1", ts=3.0, tool_name="move"))

    started = recorder.events(event_type=EventType.TOOL_STARTED)
    assert len(started) == 2
    assert all(e.event_type is EventType.TOOL_STARTED for e in started)


def test_events_are_chronological(recorder: MissionTraceRecorder) -> None:
    recorder._on_event(_event(EventType.AGENT_TURN_STARTED, mission_id="m-1", ts=3.0))
    recorder._on_event(_event(EventType.MISSION_STARTED, mission_id="m-1", ts=1.0))
    recorder._on_event(_event(EventType.INPUT_RECEIVED, mission_id="m-1", ts=2.0))

    timestamps = [e.timestamp for e in recorder.events(mission_id="m-1")]
    assert timestamps == sorted(timestamps)


def test_multiple_missions_isolated(recorder: MissionTraceRecorder) -> None:
    recorder._on_event(_event(EventType.MISSION_STARTED, mission_id="m-1", ts=1.0))
    recorder._on_event(_event(EventType.MISSION_STARTED, mission_id="m-2", ts=2.0))

    assert {e.mission_id for e in recorder.events(mission_id="m-1")} == {"m-1"}
    assert {e.mission_id for e in recorder.events(mission_id="m-2")} == {"m-2"}


def test_round_trips_full_event(recorder: MissionTraceRecorder) -> None:
    original = MissionEvent.create(
        EventType.TOOL_COMPLETED,
        EventSource.SERVER,
        MissionContext(run_id="run-1", mission_id="m-9", turn_id=2, invocation_id="pt-1"),
        tool_name="observe",
        status=Status.SUCCESS,
        duration_ms=42.0,
        summary="I see a chair.",
        attributes={"args": {"x": 1}},
    )
    recorder._on_event(original)
    (restored,) = recorder.events(mission_id="m-9")
    assert restored == original


def test_attributions_from_recorded_events(recorder: MissionTraceRecorder) -> None:
    recorder._on_event(_event(EventType.TOOL_STARTED, mission_id="m-1", ts=1.0, tool_name="grab"))
    recorder._on_event(
        _event(
            EventType.TOOL_FAILED,
            mission_id="m-1",
            ts=2.0,
            tool_name="grab",
            error_code="EXECUTION_FAILED",
        )
    )
    (attribution,) = recorder.attributions(mission_id="m-1")
    assert attribution.category is FailureCategory.SKILL_EXECUTION
    assert attribution.tool_name == "grab"


def test_records_even_under_replay_mode(tmp_path: Path) -> None:
    # The mission recorder must keep recording new execution events during a
    # replay run, unlike the sensor Recorder.
    original = global_config.replay
    global_config.replay = True
    rec = MissionTraceRecorder(db_path=str(tmp_path / "replay.db"))
    try:
        assert rec.config.g.replay is True
        rec._on_event(_event(EventType.MISSION_STARTED, mission_id="m-1", ts=1.0))
        assert len(rec.events(mission_id="m-1")) == 1
    finally:
        rec.stop()
        global_config.replay = original


def test_db_file_created_and_closes_cleanly(tmp_path: Path) -> None:
    db_path = tmp_path / "closes.db"
    rec = MissionTraceRecorder(db_path=str(db_path))
    rec._on_event(_event(EventType.MISSION_STARTED, mission_id="m-1", ts=1.0))
    assert db_path.exists()
    # Stop should dispose the store without raising.
    rec.stop()


def test_stream_name_default(recorder: MissionTraceRecorder) -> None:
    assert MISSION_EVENTS_STREAM == "mission_events"
    assert recorder.config.stream_name == "mission_events"


def test_resolve_db_path_falls_back_to_run_log_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In a forkserver worker `get_run_log_dir()` is None, but the CLI exports
    # DIMOS_RUN_LOG_DIR (inherited by children) — the db must still land under
    # the run log dir next to main.jsonl, not at the project root.
    import dimos.agents.trace.recorder as recorder_mod

    monkeypatch.setattr(recorder_mod, "get_run_log_dir", lambda: None)
    monkeypatch.setenv("DIMOS_RUN_LOG_DIR", str(tmp_path))
    rec = MissionTraceRecorder()  # default db_path
    try:
        rec._resolve_db_path()
        assert Path(rec.config.db_path) == tmp_path / "mission_trace.db"
    finally:
        rec.stop()


def test_resolve_db_path_respects_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicitly configured db_path (e.g. a test's tmp_path) is never rebased.
    explicit = tmp_path / "custom.db"
    monkeypatch.setenv("DIMOS_RUN_LOG_DIR", str(tmp_path / "run"))
    rec = MissionTraceRecorder(db_path=str(explicit))
    try:
        rec._resolve_db_path()
        assert Path(rec.config.db_path) == explicit
    finally:
        rec.stop()
