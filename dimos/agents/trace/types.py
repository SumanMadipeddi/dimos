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

"""Normalized mission-trace event schema.

A :class:`MissionEvent` is a single, immutable, deterministically serializable
fact about a mission's execution. Events are correlated by the
:class:`MissionContext` hierarchy (``run_id`` -> ``mission_id`` -> ``turn_id``
-> ``invocation_id``) and persisted into a memory2 ``mission_events`` stream by
:class:`dimos.agents.trace.recorder.MissionTraceRecorder`.

Design constraints (see the feature spec):

* versioned schema (:data:`SCHEMA_VERSION`)
* deterministic ``to_dict``/``from_dict`` round-trip (memory2 / SQLite safe)
* immutable (frozen dataclass)
* bounded, sanitized ``attributes`` — never persists secrets, auth headers, or
  private chain-of-thought, and never duplicates raw sensor payloads
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import time
from typing import Any
import uuid

SCHEMA_VERSION = 1

# Bounds applied to sanitized ``attributes`` so a trace event can never grow
# without limit or smuggle a raw payload into the mission stream.
_MAX_STRING_LEN = 2000
_MAX_ITEMS = 64
_MAX_DEPTH = 4

# Substrings (case-insensitive) that mark a key whose value must never be
# persisted. Kept deliberately narrow so ordinary tool args survive.
_SECRET_KEY_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "cookie",
    "access_token",
    "refresh_token",
    "bearer",
    "credential",
    "private_key",
)
_REDACTED = "[redacted]"


class EventType(str, Enum):
    """The kind of execution fact a :class:`MissionEvent` records."""

    # Mission / input lifecycle (client authoritative)
    MISSION_STARTED = "mission_started"
    INPUT_RECEIVED = "input_received"

    # Agent turn lifecycle (client authoritative)
    AGENT_TURN_STARTED = "agent_turn_started"
    AGENT_MESSAGE = "agent_message"
    TOOL_SELECTED = "tool_selected"
    AGENT_TURN_COMPLETED = "agent_turn_completed"

    # Tool execution lifecycle (server authoritative)
    TOOL_STARTED = "tool_started"
    TOOL_REFUSED = "tool_refused"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    # Background / asynchronous tool lifecycle (tool-stream)
    TOOL_PROGRESS = "tool_progress"
    TOOL_STOPPED = "tool_stopped"

    # Physical verification / terminal semantics (reserved; not auto-emitted)
    PHYSICAL_TASK_STARTED = "physical_task_started"
    PHYSICAL_TASK_COMPLETED = "physical_task_completed"
    TASK_VERIFICATION = "task_verification"
    MISSION_COMPLETED = "mission_completed"
    MISSION_FAILED = "mission_failed"
    MISSION_CLOSED = "mission_closed"


class EventSource(str, Enum):
    """Which layer produced the event."""

    CLIENT = "client"
    SERVER = "server"
    TOOL_STREAM = "tool_stream"
    RECORDER = "recorder"


class Status(str, Enum):
    """Coarse outcome for events that carry one.

    ``UNVERIFIED`` is used where an RPC returned but the physical result has not
    been independently confirmed (see the feature spec's Layer E): we never
    claim success we cannot prove.
    """

    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    REFUSED = "refused"
    UNVERIFIED = "unverified"


def new_event_id() -> str:
    """Generate a fresh, unique event id."""
    return uuid.uuid4().hex


def _current_run_id() -> str:
    """Best-effort DimOS run id.

    Reuses the CLI/runtime ``DIMOS_RUN_ID`` env var so the mission trace shares
    the same run identifier as logs and the run registry. Falls back to the
    per-run log directory name, then to ``"local"`` for in-process/test runs.
    """
    run_id = os.environ.get("DIMOS_RUN_ID")
    if run_id:
        return run_id
    from dimos.utils.logging_config import get_run_log_dir

    log_dir = get_run_log_dir()
    if log_dir is not None:
        return log_dir.name
    return "local"


@dataclass(frozen=True)
class MissionContext:
    """The correlation hierarchy carried by every mission event.

    * ``run_id`` — one DimOS process/run (reuses ``DIMOS_RUN_ID``).
    * ``mission_id`` — one externally initiated top-level user mission.
    * ``turn_id`` — one model/agent processing cycle within the mission.
    * ``invocation_id`` — one tool invocation (reuses the MCP ``progressToken``).
    """

    run_id: str
    mission_id: str
    turn_id: str | int | None = None
    invocation_id: str | None = None

    @classmethod
    def for_mission(cls, mission_id: str, run_id: str | None = None) -> MissionContext:
        return cls(
            run_id=run_id if run_id is not None else _current_run_id(), mission_id=mission_id
        )

    def with_turn(self, turn_id: str | int) -> MissionContext:
        return MissionContext(
            run_id=self.run_id,
            mission_id=self.mission_id,
            turn_id=turn_id,
            invocation_id=self.invocation_id,
        )

    def with_invocation(self, invocation_id: str | None) -> MissionContext:
        return MissionContext(
            run_id=self.run_id,
            mission_id=self.mission_id,
            turn_id=self.turn_id,
            invocation_id=invocation_id,
        )


def _sanitize_value(value: Any, depth: int) -> Any:
    """Recursively bound and JSON-normalize an attribute value.

    Strings are truncated, containers are capped in length and depth, and
    anything not natively JSON-friendly is coerced to a truncated ``repr`` so
    the event stays picklable and small.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LEN:
            return value[:_MAX_STRING_LEN] + "...[truncated]"
        return value
    if depth >= _MAX_DEPTH:
        return _truncate_repr(value)
    if isinstance(value, dict):
        return sanitize_attributes(value, _depth=depth + 1)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v, depth + 1) for v in list(value)[:_MAX_ITEMS]]
    return _truncate_repr(value)


def _truncate_repr(value: Any) -> str:
    text = repr(value)
    if len(text) > _MAX_STRING_LEN:
        return text[:_MAX_STRING_LEN] + "...[truncated]"
    return text


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def sanitize_attributes(attributes: dict[str, Any] | None, *, _depth: int = 0) -> dict[str, Any]:
    """Return a bounded, secret-free copy of ``attributes``.

    Drops secret-like keys, truncates long strings, caps container size/depth,
    and coerces non-serializable values to a truncated ``repr``. Safe to call on
    tool arguments and results.
    """
    if not attributes:
        return {}
    out: dict[str, Any] = {}
    for i, (key, value) in enumerate(attributes.items()):
        if i >= _MAX_ITEMS:
            break
        skey = str(key)
        if _is_secret_key(skey):
            out[skey] = _REDACTED
            continue
        out[skey] = _sanitize_value(value, _depth)
    return out


@dataclass(frozen=True)
class MissionEvent:
    """A single, immutable, timestamped mission-execution fact."""

    event_type: EventType
    source: EventSource
    run_id: str
    mission_id: str

    schema_version: int = SCHEMA_VERSION
    event_id: str = field(default_factory=new_event_id)
    timestamp: float = field(default_factory=time.time)

    turn_id: str | int | None = None
    invocation_id: str | None = None
    tool_name: str | None = None

    status: Status | None = None
    duration_ms: float | None = None
    error_code: str | None = None

    summary: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: EventType,
        source: EventSource,
        context: MissionContext,
        *,
        tool_name: str | None = None,
        status: Status | None = None,
        duration_ms: float | None = None,
        error_code: str | None = None,
        summary: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> MissionEvent:
        """Build an event from a :class:`MissionContext`, sanitizing attributes."""
        return cls(
            event_type=event_type,
            source=source,
            run_id=context.run_id,
            mission_id=context.mission_id,
            turn_id=context.turn_id,
            invocation_id=context.invocation_id,
            tool_name=tool_name,
            status=status,
            duration_ms=duration_ms,
            error_code=error_code,
            summary=summary,
            attributes=sanitize_attributes(attributes),
        )

    @property
    def context(self) -> MissionContext:
        return MissionContext(
            run_id=self.run_id,
            mission_id=self.mission_id,
            turn_id=self.turn_id,
            invocation_id=self.invocation_id,
        )

    def tags(self) -> dict[str, Any]:
        """memory2 query tags. Kept small and scalar for cheap filtering."""
        tags: dict[str, Any] = {
            "run_id": self.run_id,
            "mission_id": self.mission_id,
            "event_type": self.event_type.value,
            "source": self.source.value,
        }
        if self.turn_id is not None:
            tags["turn_id"] = self.turn_id
        if self.invocation_id is not None:
            tags["invocation_id"] = self.invocation_id
        if self.tool_name is not None:
            tags["tool_name"] = self.tool_name
        if self.status is not None:
            tags["status"] = self.status.value
        return tags

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-friendly representation."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "mission_id": self.mission_id,
            "event_type": self.event_type.value,
            "source": self.source.value,
            "turn_id": self.turn_id,
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "status": self.status.value if self.status is not None else None,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
            "summary": self.summary,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionEvent:
        status = data.get("status")
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            run_id=data["run_id"],
            mission_id=data["mission_id"],
            event_type=EventType(data["event_type"]),
            source=EventSource(data["source"]),
            turn_id=data.get("turn_id"),
            invocation_id=data.get("invocation_id"),
            tool_name=data.get("tool_name"),
            status=Status(status) if status is not None else None,
            duration_ms=data.get("duration_ms"),
            error_code=data.get("error_code"),
            summary=data.get("summary"),
            attributes=dict(data.get("attributes") or {}),
        )
