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

"""Replayable agent mission trace and failure attribution.

This package adds a structured, persistent, memory2-backed record of what
happened during an agent mission — from user request through model decision,
tool selection, MCP invocation, background progress and (eventually)
verification — without changing the existing DimOS execution path.

Public surface:

* :mod:`dimos.agents.trace.types` — the normalized :class:`MissionEvent`
  schema, its :class:`EventType`/:class:`EventSource` enums, the
  :class:`MissionContext` correlation tuple, and payload sanitization.
* :mod:`dimos.agents.trace.recorder` — :class:`MissionTraceRecorder`, a
  memory2 ``MemoryModule`` that persists events into a ``mission_events``
  stream (records even under global replay mode).
* :mod:`dimos.agents.trace.attribution` — deterministic failure attribution
  derived from the raw event history.
"""

from dimos.agents.trace.types import (
    SCHEMA_VERSION,
    EventSource,
    EventType,
    MissionContext,
    MissionEvent,
    Status,
    new_event_id,
    sanitize_attributes,
)

__all__ = [
    "SCHEMA_VERSION",
    "EventSource",
    "EventType",
    "MissionContext",
    "MissionEvent",
    "Status",
    "new_event_id",
    "sanitize_attributes",
]
