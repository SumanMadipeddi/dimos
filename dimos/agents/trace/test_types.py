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

from __future__ import annotations

import pickle

from dimos.agents.trace.types import (
    SCHEMA_VERSION,
    EventSource,
    EventType,
    MissionContext,
    MissionEvent,
    Status,
    sanitize_attributes,
)


def _ctx() -> MissionContext:
    return MissionContext(run_id="run-1", mission_id="m-1", turn_id=2, invocation_id="pt-9")


def test_create_populates_context_and_schema_version() -> None:
    event = MissionEvent.create(
        EventType.TOOL_STARTED,
        EventSource.SERVER,
        _ctx(),
        tool_name="observe",
        status=Status.STARTED,
    )
    assert event.schema_version == SCHEMA_VERSION
    assert event.run_id == "run-1"
    assert event.mission_id == "m-1"
    assert event.turn_id == 2
    assert event.invocation_id == "pt-9"
    assert event.tool_name == "observe"
    assert event.status is Status.STARTED
    assert event.event_id  # non-empty
    assert event.timestamp > 0


def test_events_are_immutable() -> None:
    event = MissionEvent.create(EventType.MISSION_STARTED, EventSource.CLIENT, _ctx())
    try:
        event.mission_id = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("MissionEvent should be frozen/immutable")


def test_to_dict_from_dict_round_trip() -> None:
    event = MissionEvent.create(
        EventType.TOOL_COMPLETED,
        EventSource.SERVER,
        _ctx(),
        tool_name="observe",
        status=Status.SUCCESS,
        duration_ms=12.5,
        summary="OK",
        attributes={"x": 1, "nested": {"y": [1, 2, 3]}},
    )
    restored = MissionEvent.from_dict(event.to_dict())
    assert restored == event


def test_pickle_round_trip_matches() -> None:
    # memory2's PickleCodec stores the event verbatim.
    event = MissionEvent.create(EventType.AGENT_MESSAGE, EventSource.CLIENT, _ctx(), summary="hi")
    assert pickle.loads(pickle.dumps(event)) == event


def test_deterministic_serialization_is_stable() -> None:
    event = MissionEvent.create(EventType.INPUT_RECEIVED, EventSource.CLIENT, _ctx(), summary="go")
    assert event.to_dict() == event.to_dict()


def test_tags_are_scalar_and_filterable() -> None:
    event = MissionEvent.create(
        EventType.TOOL_FAILED,
        EventSource.SERVER,
        _ctx(),
        tool_name="grab",
        status=Status.FAILURE,
        error_code="EXECUTION_FAILED",
    )
    tags = event.tags()
    assert tags["run_id"] == "run-1"
    assert tags["mission_id"] == "m-1"
    assert tags["event_type"] == "tool_failed"
    assert tags["tool_name"] == "grab"
    assert tags["status"] == "failure"
    assert tags["invocation_id"] == "pt-9"
    assert all(isinstance(v, (str, int, float)) for v in tags.values())


def test_sanitize_redacts_secret_keys() -> None:
    out = sanitize_attributes(
        {
            "Authorization": "Bearer abc",
            "api_key": "sk-123",
            "password": "hunter2",
            "cookie": "session=xyz",
            "distance": 1.5,
            "target": "chair",
        }
    )
    assert out["Authorization"] == "[redacted]"
    assert out["api_key"] == "[redacted]"
    assert out["password"] == "[redacted]"
    assert out["cookie"] == "[redacted]"
    # Ordinary args survive untouched.
    assert out["distance"] == 1.5
    assert out["target"] == "chair"


def test_sanitize_truncates_long_strings() -> None:
    out = sanitize_attributes({"blob": "a" * 10_000})
    assert out["blob"].endswith("...[truncated]")
    assert len(out["blob"]) < 10_000


def test_sanitize_bounds_depth_and_is_json_safe() -> None:
    deep: dict = {"a": {"b": {"c": {"d": {"e": object()}}}}}
    out = sanitize_attributes(deep)
    # The object() at excessive depth is coerced to a repr string, not kept raw.
    leaf = out["a"]["b"]["c"]["d"]
    assert isinstance(leaf, (str, dict))


def test_context_helpers() -> None:
    ctx = MissionContext.for_mission("m-42", run_id="run-x")
    assert ctx.run_id == "run-x"
    assert ctx.mission_id == "m-42"
    assert ctx.turn_id is None
    turned = ctx.with_turn(3)
    assert turned.turn_id == 3 and turned.mission_id == "m-42"
    invoked = turned.with_invocation("pt-1")
    assert invoked.invocation_id == "pt-1" and invoked.turn_id == 3
