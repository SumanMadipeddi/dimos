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

"""Deterministic mission-trace tests for the MCP client/server layers.

No real robot, no hosted LLM, no network transport: the model is a
:class:`MockModel` with scripted responses and the MCP HTTP layer is a mocked
``requests.Session`` (Test Levels 2 and 3 in the feature spec).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import MagicMock, create_autospec

from langchain_core.messages import AIMessage, HumanMessage
import pytest
import requests

from dimos.agents.mcp.mcp_client import McpClient, MessageSource, _QueuedAgentMessage
from dimos.agents.mcp.mcp_server import app, handle_request
from dimos.agents.testing import MockModel
from dimos.agents.trace.types import EventSource, EventType, Status
from dimos.core.module import SkillInfo

# --- Client-side (Test Level 3: deterministic MockModel mission) --------------


def _payload_with_capture(captured: list[dict[str, object]]) -> Callable:
    def payload_fn(body: dict[str, object]) -> dict[str, object]:
        method = body["method"]
        req_id = body["id"]
        if method == "initialize":
            result: object = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "observe",
                        "description": "Observe surroundings",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        elif method == "tools/call":
            captured.append(body)
            result = {"content": [{"type": "text", "text": "I see a chair and a table."}]}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "?"}}
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    return payload_fn


def _mock_session(payload_fn: Callable) -> MagicMock:
    def _post(url: str, *, json: dict, timeout: float | None = None) -> MagicMock:
        resp = create_autospec(requests.Response, instance=True, spec_set=True)
        resp.json.return_value = payload_fn(json)
        return resp

    session = create_autospec(requests.Session, instance=True, spec_set=True)
    session.post.side_effect = _post
    return session


@pytest.fixture
def observe_client():
    """An McpClient whose model deterministically calls ``observe`` then replies."""
    from langchain.agents import create_agent

    captured: list[dict[str, object]] = []
    client = McpClient(mcp_server_url="http://localhost:9990/mcp")
    client._http_client = _mock_session(_payload_with_capture(captured))
    tools = client._fetch_tools()

    model = MockModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "observe", "args": {}, "id": "call-1"}],
            ),
            AIMessage(content="I see a chair and a table."),
        ]
    )
    client._state_graph = create_agent(model=model, tools=tools, system_prompt="You are a robot.")
    try:
        yield client, captured
    finally:
        client.stop()


def test_client_emits_full_mission_lifecycle(observe_client) -> None:
    client, _ = observe_client
    events = []
    client.mission_trace.subscribe(events.append)

    client._process_message(
        client._state_graph,
        _QueuedAgentMessage(
            HumanMessage(content="Observe your surroundings and describe what you see."),
            MessageSource.EXTERNAL_INPUT,
            "mission-abc",
        ),
    )

    types = [e.event_type for e in events]
    # The deterministic observe mission produces this structural sequence.
    assert types[0] is EventType.MISSION_STARTED
    assert types[1] is EventType.INPUT_RECEIVED
    assert EventType.AGENT_TURN_STARTED in types
    assert EventType.TOOL_SELECTED in types
    assert EventType.AGENT_MESSAGE in types
    assert types[-1] is EventType.AGENT_TURN_COMPLETED

    # One mission id, one run id, consistent turn id across the whole turn.
    assert {e.mission_id for e in events} == {"mission-abc"}
    assert len({e.run_id for e in events}) == 1
    assert {e.turn_id for e in events} == {1}

    selected = next(e for e in events if e.event_type is EventType.TOOL_SELECTED)
    assert selected.tool_name == "observe"
    assert selected.source is EventSource.CLIENT

    # No hidden chain-of-thought is persisted — only visible assistant text.
    assistant = next(e for e in events if e.event_type is EventType.AGENT_MESSAGE)
    assert assistant.summary == "I see a chair and a table."
    assert assistant.status is Status.UNVERIFIED


def test_mcp_call_propagates_correlation_metadata(observe_client) -> None:
    client, captured = observe_client

    client._process_message(
        client._state_graph,
        _QueuedAgentMessage(
            HumanMessage(content="observe"), MessageSource.EXTERNAL_INPUT, "mission-xyz"
        ),
    )

    assert captured, "expected an MCP tools/call to be issued"
    meta = captured[0]["params"]["_meta"]  # type: ignore[index]
    assert isinstance(meta, dict)
    # progressToken doubles as the invocation id; mission/turn ride along.
    assert isinstance(meta["progressToken"], str) and meta["progressToken"]
    assert meta["missionId"] == "mission-xyz"
    assert meta["turnId"] == 1
    assert meta["runId"]


def test_tool_progress_does_not_open_new_mission(observe_client) -> None:
    client, _ = observe_client
    # An external input sets the active mission.
    client._enqueue(HumanMessage(content="observe"), MessageSource.EXTERNAL_INPUT)
    external = client._message_queue.get_nowait()

    # A tool-progress update re-enters the queue but keeps the same mission id.
    client._on_tool_stream_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progressToken": "pt", "progress": 1, "message": "still going"},
        }
    )
    progress = client._message_queue.get_nowait()

    assert external.source is MessageSource.EXTERNAL_INPUT
    assert progress.source is MessageSource.TOOL_PROGRESS
    assert progress.mission_id == external.mission_id


def test_distinct_external_requests_get_distinct_missions(observe_client) -> None:
    client, _ = observe_client
    client._enqueue(HumanMessage(content="a"), MessageSource.EXTERNAL_INPUT)
    client._enqueue(HumanMessage(content="b"), MessageSource.EXTERNAL_INPUT)
    first = client._message_queue.get_nowait()
    second = client._message_queue.get_nowait()
    assert first.mission_id != second.mission_id


# --- Server-side (Test Level 2: mocked MCP) -----------------------------------


def _server_trace_sink() -> list:
    events: list = []
    app.state.trace_sink = events.append
    return events


def _clear_server_trace_sink() -> None:
    app.state.trace_sink = None
    app.state.trace_invocations = {}


def test_server_emits_tool_started_and_completed_on_success() -> None:
    schema = '{"type": "object", "properties": {}}'
    skills = [SkillInfo(class_name="S", func_name="observe", args_schema=schema)]
    rpc_calls = {"observe": MagicMock(return_value="seen")}
    events = _server_trace_sink()
    try:
        asyncio.run(
            handle_request(
                {
                    "method": "tools/call",
                    "id": 1,
                    "params": {
                        "name": "observe",
                        "arguments": {},
                        "_meta": {
                            "progressToken": "pt-1",
                            "runId": "run-1",
                            "missionId": "m-1",
                            "turnId": 4,
                        },
                    },
                },
                skills,
                rpc_calls,
            )
        )
    finally:
        _clear_server_trace_sink()

    types = [e.event_type for e in events]
    assert types == [EventType.TOOL_STARTED, EventType.TOOL_COMPLETED]
    started, completed = events
    assert started.invocation_id == "pt-1"
    assert started.mission_id == "m-1"
    assert started.turn_id == 4
    assert completed.status is Status.SUCCESS
    assert completed.duration_ms is not None and completed.duration_ms >= 0
    assert all(e.source is EventSource.SERVER for e in events)


def test_server_emits_tool_failed_on_skill_result_failure() -> None:
    from dimos.agents.skill_result import SkillResult

    schema = '{"type": "object", "properties": {}}'
    skills = [SkillInfo(class_name="S", func_name="grab", args_schema=schema)]
    rpc_calls = {"grab": MagicMock(return_value=SkillResult.fail("EXECUTION_FAILED", "no object"))}
    events = _server_trace_sink()
    try:
        asyncio.run(
            handle_request(
                {
                    "method": "tools/call",
                    "id": 2,
                    "params": {"name": "grab", "arguments": {}, "_meta": {"progressToken": "pt-2"}},
                },
                skills,
                rpc_calls,
            )
        )
    finally:
        _clear_server_trace_sink()

    failed = events[-1]
    assert failed.event_type is EventType.TOOL_FAILED
    assert failed.status is Status.FAILURE
    assert failed.error_code == "EXECUTION_FAILED"


def test_server_emits_tool_failed_on_exception() -> None:
    schema = '{"type": "object", "properties": {}}'
    skills = [SkillInfo(class_name="S", func_name="boom", args_schema=schema)]
    rpc_calls = {"boom": MagicMock(side_effect=RuntimeError("kaboom"))}
    events = _server_trace_sink()
    try:
        asyncio.run(
            handle_request(
                {"method": "tools/call", "id": 3, "params": {"name": "boom", "arguments": {}}},
                skills,
                rpc_calls,
            )
        )
    finally:
        _clear_server_trace_sink()

    failed = events[-1]
    assert failed.event_type is EventType.TOOL_FAILED
    assert failed.status is Status.FAILURE
    assert failed.error_code == "EXCEPTION"


def test_server_trace_sink_absent_is_noop() -> None:
    # With no sink installed (the default), server behaviour is unchanged and
    # nothing is emitted — existing MCP tests remain valid.
    _clear_server_trace_sink()
    schema = '{"type": "object", "properties": {}}'
    skills = [SkillInfo(class_name="S", func_name="ping", args_schema=schema)]
    rpc_calls = {"ping": MagicMock(return_value="pong")}
    response = asyncio.run(
        handle_request(
            {"method": "tools/call", "id": 4, "params": {"name": "ping", "arguments": {}}},
            skills,
            rpc_calls,
        )
    )
    assert response is not None
    assert response["result"]["content"][0]["text"] == "pong"
