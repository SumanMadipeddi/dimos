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

"""Blueprint-wiring checks for the mission-trace channel (no network/robot).

Only blueprint introspection is exercised here — modules are never
instantiated, so no transport/LCM is created.
"""

from __future__ import annotations

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.trace.recorder import MissionTraceRecorder
from dimos.agents.trace.types import MissionEvent
from dimos.core.coordination.blueprints import autoconnect


def _mission_trace_refs(blueprint):
    return [
        (atom.name, ref.direction, ref.type)
        for atom in blueprint.blueprints
        for ref in atom.streams
        if ref.name == "mission_trace"
    ]


def test_client_server_recorder_share_mission_trace_channel() -> None:
    blueprint = autoconnect(
        McpClient.blueprint(),
        McpServer.blueprint(),
        MissionTraceRecorder.blueprint(),
    )
    refs = _mission_trace_refs(blueprint)

    directions = sorted(direction for _, direction, _ in refs)
    # Two publishers (client + server) fan into one recorder subscriber.
    assert directions == ["in", "out", "out"]
    # All three ports share the exact same payload type, so the coordinator
    # assigns them a single (name, type) transport.
    assert {ref_type for _, _, ref_type in refs} == {MissionEvent}


def test_recorder_input_present() -> None:
    refs = _mission_trace_refs(autoconnect(MissionTraceRecorder.blueprint()))
    assert refs == [("missiontracerecorder", "in", MissionEvent)]
