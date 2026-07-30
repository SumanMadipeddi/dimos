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

"""Deterministic failure attribution derived from raw mission events.

Attribution is an *interpretation* of the factual event history; it is kept
strictly separate from event recording. Rules are conservative and
deterministic — where evidence is insufficient the category is
:attr:`FailureCategory.UNKNOWN` rather than a guess. In particular this module
never inspects hidden model reasoning, so ``reasoning`` is only assigned when an
external oracle has already labelled it (not inferred here).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dimos.agents.trace.types import EventType, MissionEvent

# error_code markers that indicate an infrastructure/transport-layer failure
# rather than a skill-execution failure.
_INFRASTRUCTURE_CODES = frozenset(
    {
        "TRANSPORT_ERROR",
        "RPC_ERROR",
        "SERIALIZATION_ERROR",
        "WORKER_CRASH",
        "MCP_PROTOCOL_ERROR",
        "CONNECTION_ERROR",
    }
)


class FailureCategory(str, Enum):
    """Which execution layer a mission failure is attributed to."""

    INPUT = "input"
    PERCEPTION = "perception"
    REASONING = "reasoning"
    TOOL_SELECTION = "tool_selection"
    SKILL_EXECUTION = "skill_execution"
    CONTROL = "control"
    VERIFICATION = "verification"
    INFRASTRUCTURE = "infrastructure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureAttribution:
    """A deterministic interpretation of why a mission failed."""

    run_id: str
    mission_id: str
    category: FailureCategory
    reason: str
    event_id: str | None = None
    tool_name: str | None = None
    error_code: str | None = None


def _classify_event(event: MissionEvent) -> FailureCategory:
    """Map a single terminal-failure event to a category (deterministic)."""
    if event.event_type is EventType.TOOL_REFUSED:
        # A capability lock prevented the controller from starting the action.
        return FailureCategory.CONTROL
    if event.event_type is EventType.MISSION_FAILED:
        return FailureCategory.UNKNOWN
    # TOOL_FAILED
    if event.error_code == "TOOL_NOT_FOUND":
        return FailureCategory.TOOL_SELECTION
    if event.error_code in _INFRASTRUCTURE_CODES:
        return FailureCategory.INFRASTRUCTURE
    # A skill returning success=False or raising is a skill-execution failure.
    return FailureCategory.SKILL_EXECUTION


def attribute_mission(events: list[MissionEvent]) -> FailureAttribution | None:
    """Attribute a single mission's failure, or ``None`` if it did not fail.

    ``events`` must all belong to one mission. The first failure-bearing event
    (``tool_failed`` / ``tool_refused`` / ``mission_failed``), in chronological
    order, determines the attribution.
    """
    if not events:
        return None
    ordered = sorted(events, key=lambda e: e.timestamp)
    failure_types = {
        EventType.TOOL_FAILED,
        EventType.TOOL_REFUSED,
        EventType.MISSION_FAILED,
    }
    failing = next((e for e in ordered if e.event_type in failure_types), None)
    if failing is None:
        return None

    category = _classify_event(failing)
    reason = failing.summary or f"{failing.event_type.value} ({failing.error_code})"
    return FailureAttribution(
        run_id=failing.run_id,
        mission_id=failing.mission_id,
        category=category,
        reason=reason,
        event_id=failing.event_id,
        tool_name=failing.tool_name,
        error_code=failing.error_code,
    )


def attribute_failures(events: list[MissionEvent]) -> list[FailureAttribution]:
    """Group events by mission and attribute each failed mission.

    Returns one :class:`FailureAttribution` per failed mission, ordered by the
    earliest event timestamp of each mission.
    """
    by_mission: dict[str, list[MissionEvent]] = {}
    for event in events:
        by_mission.setdefault(event.mission_id, []).append(event)

    attributions: list[FailureAttribution] = []
    for mission_events in by_mission.values():
        attribution = attribute_mission(mission_events)
        if attribution is not None:
            attributions.append(attribution)

    attributions.sort(key=lambda a: min(e.timestamp for e in by_mission[a.mission_id]))
    return attributions
