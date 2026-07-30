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

from dimos.agents.trace.attribution import (
    FailureCategory,
    attribute_failures,
    attribute_mission,
)
from dimos.agents.trace.types import (
    EventSource,
    EventType,
    MissionContext,
    MissionEvent,
    Status,
)


def _event(
    event_type: EventType,
    *,
    mission_id: str = "m-1",
    ts: float,
    error_code: str | None = None,
    tool_name: str | None = None,
    summary: str | None = None,
) -> MissionEvent:
    ctx = MissionContext(run_id="run-1", mission_id=mission_id, turn_id=1)
    return MissionEvent(
        event_type=event_type,
        source=EventSource.SERVER,
        run_id=ctx.run_id,
        mission_id=ctx.mission_id,
        turn_id=ctx.turn_id,
        timestamp=ts,
        error_code=error_code,
        tool_name=tool_name,
        summary=summary,
        status=Status.FAILURE if event_type is EventType.TOOL_FAILED else None,
    )


def test_no_failure_returns_none() -> None:
    events = [
        _event(EventType.MISSION_STARTED, ts=1),
        _event(EventType.TOOL_COMPLETED, ts=2),
        _event(EventType.AGENT_TURN_COMPLETED, ts=3),
    ]
    assert attribute_mission(events) is None
    assert attribute_failures(events) == []


def test_skill_result_failure_is_skill_execution() -> None:
    events = [
        _event(EventType.TOOL_STARTED, ts=1, tool_name="grab"),
        _event(EventType.TOOL_FAILED, ts=2, tool_name="grab", error_code="EXECUTION_FAILED"),
    ]
    attribution = attribute_mission(events)
    assert attribution is not None
    assert attribution.category is FailureCategory.SKILL_EXECUTION
    assert attribution.tool_name == "grab"
    assert attribution.error_code == "EXECUTION_FAILED"


def test_tool_not_found_is_tool_selection() -> None:
    events = [_event(EventType.TOOL_FAILED, ts=1, tool_name="fly", error_code="TOOL_NOT_FOUND")]
    attribution = attribute_mission(events)
    assert attribution is not None
    assert attribution.category is FailureCategory.TOOL_SELECTION


def test_infrastructure_error_code_is_infrastructure() -> None:
    events = [_event(EventType.TOOL_FAILED, ts=1, tool_name="move", error_code="RPC_ERROR")]
    attribution = attribute_mission(events)
    assert attribution is not None
    assert attribution.category is FailureCategory.INFRASTRUCTURE


def test_refused_is_control() -> None:
    events = [
        _event(EventType.TOOL_REFUSED, ts=1, tool_name="patrol", error_code="CAPABILITY_BUSY")
    ]
    attribution = attribute_mission(events)
    assert attribution is not None
    assert attribution.category is FailureCategory.CONTROL


def test_first_failure_wins() -> None:
    events = [
        _event(EventType.TOOL_FAILED, ts=5, error_code="EXECUTION_FAILED"),
        _event(EventType.TOOL_FAILED, ts=1, error_code="RPC_ERROR"),
    ]
    attribution = attribute_mission(events)
    assert attribution is not None
    # Earliest failure (ts=1) determines the attribution.
    assert attribution.category is FailureCategory.INFRASTRUCTURE


def test_multiple_missions_isolated() -> None:
    events = [
        _event(EventType.TOOL_FAILED, mission_id="a", ts=1, error_code="EXECUTION_FAILED"),
        _event(EventType.TOOL_COMPLETED, mission_id="b", ts=2),
        _event(EventType.TOOL_FAILED, mission_id="c", ts=3, error_code="TOOL_NOT_FOUND"),
    ]
    attributions = attribute_failures(events)
    by_mission = {a.mission_id: a.category for a in attributions}
    assert by_mission == {
        "a": FailureCategory.SKILL_EXECUTION,
        "c": FailureCategory.TOOL_SELECTION,
    }


def test_reasoning_never_inferred() -> None:
    # A plain failure is never guessed as a reasoning failure.
    events = [_event(EventType.TOOL_FAILED, ts=1, error_code="EXECUTION_FAILED")]
    attribution = attribute_mission(events)
    assert attribution is not None
    assert attribution.category is not FailureCategory.REASONING
