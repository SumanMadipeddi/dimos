# Copyright 2025-2026 Dimensional Inc.
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

import asyncio
from collections.abc import AsyncGenerator, Callable
import concurrent.futures
import json
import os
import time
from typing import TYPE_CHECKING, Any
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
import uvicorn

from dimos.agents.annotation import skill
from dimos.agents.capabilities import CapabilityRegistry
from dimos.agents.mcp import tool_stream
from dimos.agents.trace.types import (
    EventSource,
    EventType,
    MissionContext,
    MissionEvent,
    Status,
)
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.rpc_client import RpcCall, RPCClient
from dimos.core.stream import Out
from dimos.core.transport_factory import make_transport
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.core.module import SkillInfo

logger = setup_logger()


_SSE_KEEPALIVE_INTERVAL = 20.0  # seconds

# How long a `tools/call` waits for a capability held by a short, self-completing
# (instant) skill before refusing. Well under the MCP client's 120s HTTP timeout.
# Background holders run until stopped, so they are never waited on (see
# `_can_wait` in `_handle_tools_call`).
DEFAULT_CAP_ACQUIRE_TIMEOUT = 30.0  # seconds

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
app.state.skills = []
app.state.skills_by_name = {}
app.state.rpc_calls = {}
app.state.sse_queues = []
app.state.event_loop = None
app.state.cap_registry = CapabilityRegistry()
app.state.cap_acquire_timeout = DEFAULT_CAP_ACQUIRE_TIMEOUT
# Mission-trace sink: a callable set by `McpServer.start` to publish server-side
# `MissionEvent`s. `None` (the default) makes all trace emission a no-op, so the
# server behaves exactly as before for callers that don't wire tracing.
app.state.trace_sink = None
app.state.trace_run_id = ""
# progressToken -> (MissionContext, tool_name) for correlating background
# (tool-stream) frames back to the invocation that started them.
app.state.trace_invocations = {}


def _emit_trace(event: MissionEvent) -> None:
    """Publish a server-side mission event through the installed sink, if any."""
    sink = app.state.trace_sink
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        logger.exception("mission-trace sink failed", event_type=event.event_type)


def _context_from_meta(meta: dict[str, Any]) -> MissionContext:
    """Reconstruct the mission correlation context from a request's ``_meta``.

    Falls back to the server's run id and an ``"unknown"`` mission when a caller
    doesn't propagate the DimOS correlation keys (e.g. Claude Code / curl).
    """
    run_id = meta.get("runId") or app.state.trace_run_id or MissionContext.for_mission("").run_id
    return MissionContext(
        run_id=str(run_id),
        mission_id=str(meta.get("missionId") or "unknown"),
        turn_id=meta.get("turnId"),
        invocation_id=meta.get("progressToken"),
    )


def _jsonrpc_result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_result_text(req_id: Any, text: str) -> dict[str, Any]:
    return _jsonrpc_result(req_id, {"content": [{"type": "text", "text": text}]})


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _handle_initialize(req_id: Any) -> dict[str, Any]:
    return _jsonrpc_result(
        req_id,
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}, "logging": {}},
            "serverInfo": {"name": "dimensional", "version": "1.0.0"},
        },
    )


def _handle_tools_list(req_id: Any, skills: list[SkillInfo]) -> dict[str, Any]:
    tools = []

    for s in skills:
        schema = json.loads(s.args_schema)
        description = schema.pop("description", None)
        schema.pop("title", None)
        tool: dict[str, Any] = {"name": s.func_name, "inputSchema": schema}
        if description:
            tool["description"] = description
        if s.uses or s.lifecycle != "instant":
            tool["_meta"] = {
                "dimos/uses": list(s.uses),
                "dimos/lifecycle": s.lifecycle,
            }
        tools.append(tool)

    return _jsonrpc_result(req_id, {"tools": tools})


async def _handle_tools_call(
    req_id: Any, params: dict[str, Any], rpc_calls: dict[str, Any]
) -> dict[str, Any]:
    name = params.get("name", "")
    args: dict[str, Any] = params.get("arguments") or {}
    meta = params.get("_meta") or {}
    progress_token = meta.get("progressToken")
    ctx = _context_from_meta(meta)

    rpc_call = rpc_calls.get(name)
    if rpc_call is None:
        logger.warning("MCP tool not found", tool=name)
        # A selected tool that doesn't exist is a tool-selection failure, not a
        # skill-execution failure — surface it as such for attribution.
        _emit_trace(
            MissionEvent.create(
                EventType.TOOL_FAILED,
                EventSource.SERVER,
                ctx,
                tool_name=name,
                status=Status.FAILURE,
                error_code="TOOL_NOT_FOUND",
                summary=f"Tool not found: {name}",
            )
        )
        return _jsonrpc_result_text(req_id, f"Tool not found: {name}")

    skill_info = app.state.skills_by_name.get(name)
    uses: list[str] = list(skill_info.uses) if skill_info is not None else []
    lifecycle = skill_info.lifecycle if skill_info is not None else "instant"
    cap_registry: CapabilityRegistry = app.state.cap_registry

    # A per-invocation token scopes the capability hold, so a stale invocation's
    # teardown can't release a hold that a newer same-tool invocation took over.
    acquire_token = uuid.uuid4().hex
    if uses:

        def _can_wait(holder: str) -> bool:
            # Wait only on instant holders; they release when they return.
            # Background holders run until explicitly stopped, so refuse instead
            # of blocking until the timeout.
            info = app.state.skills_by_name.get(holder)
            return (info.lifecycle if info is not None else "instant") != "background"

        # Run the (possibly blocking) acquire off the event loop so waiting for a
        # busy capability doesn't stall the server.
        conflict = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: cap_registry.acquire(
                uses,
                tool_name=name,
                token=acquire_token,
                timeout=app.state.cap_acquire_timeout,
                can_wait=_can_wait,
            ),
        )
        if conflict is not None:
            cap, holder = conflict
            logger.info(
                "MCP tool refused (capability busy)",
                tool=name,
                cap=cap,
                holder=holder,
                snapshot=cap_registry.snapshot(),
            )
            # A background holder has a stop tool to call; an instant holder is
            # waited on above, so reaching here means it outlasted the timeout.
            holder_info = app.state.skills_by_name.get(holder)
            holder_lifecycle = holder_info.lifecycle if holder_info is not None else "instant"
            if holder_lifecycle == "background":
                advice = "Call the appropriate stop tool first, then retry."
            else:
                advice = "It is taking longer than expected; wait a moment and then retry."
            _emit_trace(
                MissionEvent.create(
                    EventType.TOOL_REFUSED,
                    EventSource.SERVER,
                    ctx,
                    tool_name=name,
                    status=Status.REFUSED,
                    error_code="CAPABILITY_BUSY",
                    summary=f"capability '{cap}' held by '{holder}'",
                    attributes={"capability": cap, "holder": holder},
                )
            )
            return _jsonrpc_result_text(
                req_id,
                f"Cannot start '{name}': capability '{cap}' is held by '{holder}'. {advice}",
            )

    logger.info("MCP tool call", tool=name, args=args, progress_token=progress_token)

    # The server is authoritative for actual execution state. Record the start
    # (and remember the context so background tool-stream frames from a
    # long-running skill can be correlated back to this invocation).
    if progress_token is not None:
        app.state.trace_invocations[progress_token] = (ctx, name)
    _emit_trace(
        MissionEvent.create(
            EventType.TOOL_STARTED,
            EventSource.SERVER,
            ctx,
            tool_name=name,
            status=Status.STARTED,
            attributes={"args": args, "lifecycle": lifecycle},
        )
    )
    t0 = time.monotonic()

    # _mcp_context is a reserved kwarg consumed by the `@skill` wrapper; it never
    # reaches the user-visible skill signature. The acquire token rides along so
    # a background skill's ToolStream can stamp it on its stop frame for release.
    call_kwargs = dict(args)
    mcp_context: dict[str, Any] = {}
    if progress_token is not None:
        mcp_context["progress_token"] = progress_token
    if uses:
        mcp_context["acquire_token"] = acquire_token
    if mcp_context:
        call_kwargs["_mcp_context"] = mcp_context

    # Track whether we still hold the caps so we can release on failure even
    # for background skills. On success the background skill keeps them until
    # its tool-stream closes.
    caps_held = bool(uses)
    try:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: rpc_call(**call_kwargs)
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.exception("MCP tool error", tool=name, duration=f"{elapsed_ms / 1000:.3f}s")
            _emit_trace(
                MissionEvent.create(
                    EventType.TOOL_FAILED,
                    EventSource.SERVER,
                    ctx,
                    tool_name=name,
                    status=Status.FAILURE,
                    duration_ms=elapsed_ms,
                    error_code="EXCEPTION",
                    summary=f"{type(e).__name__}: {e}",
                    attributes={"exception_type": type(e).__name__},
                )
            )
            if progress_token is not None:
                app.state.trace_invocations.pop(progress_token, None)
            return _jsonrpc_result_text(req_id, f"Error running tool '{name}': {e}")

        if lifecycle == "background":
            # Hand ownership of the caps off to the tool-stream lifecycle.
            caps_held = False
    finally:
        if caps_held:
            cap_registry.release_by_token(acquire_token)

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    duration = f"{elapsed_ms / 1000:.3f}s"
    response = str(result)[:200]
    _emit_tool_result(ctx, name, result, elapsed_ms)

    # A completed instant invocation won't produce a tool-stream stop frame, so
    # release its correlation entry now; background invocations keep theirs
    # until their stop frame arrives.
    if progress_token is not None and lifecycle != "background":
        app.state.trace_invocations.pop(progress_token, None)

    if hasattr(result, "agent_encode"):
        logger.info("MCP tool done", tool=name, duration=duration, response=response)
        return _jsonrpc_result(req_id, {"content": result.agent_encode()})

    logger.info("MCP tool done", tool=name, duration=duration, response=response)
    return _jsonrpc_result_text(req_id, str(result))


def _emit_tool_result(ctx: MissionContext, name: str, result: Any, elapsed_ms: float) -> None:
    """Emit ``tool_completed`` or ``tool_failed`` based on the RPC return value.

    A structured ``SkillResult`` is authoritative: ``success=False`` becomes a
    ``tool_failed`` carrying the skill's ``error_code``. Any other return value
    means the RPC returned without raising, recorded as ``tool_completed``.
    Completion reflects that the *call returned*, not that the physical task was
    verified (see the trace spec's Layer E).
    """
    if app.state.trace_sink is None:
        return
    from dimos.agents.skill_result import SkillResult

    if isinstance(result, SkillResult) and not result.success:
        _emit_trace(
            MissionEvent.create(
                EventType.TOOL_FAILED,
                EventSource.SERVER,
                ctx,
                tool_name=name,
                status=Status.FAILURE,
                duration_ms=elapsed_ms,
                error_code=result.error_code,
                summary=result.message or None,
            )
        )
        return

    _emit_trace(
        MissionEvent.create(
            EventType.TOOL_COMPLETED,
            EventSource.SERVER,
            ctx,
            tool_name=name,
            status=Status.SUCCESS,
            duration_ms=elapsed_ms,
            summary=str(result)[:200],
        )
    )


async def handle_request(
    request: dict[str, Any],
    skills: list[SkillInfo],
    rpc_calls: dict[str, Any],
) -> dict[str, Any] | None:
    """Handle a single MCP JSON-RPC request.

    Returns None for JSON-RPC notifications (no ``id``), which must not
    receive a response.
    """
    method = request.get("method", "")
    params = request.get("params", {}) or {}
    req_id = request.get("id")

    # JSON-RPC notifications have no "id" -- the server must not reply.
    if "id" not in request:
        return None

    if method == "initialize":
        return _handle_initialize(req_id)
    if method == "tools/list":
        return _handle_tools_list(req_id, skills)
    if method == "tools/call":
        return await _handle_tools_call(req_id, params, rpc_calls)
    return _jsonrpc_error(req_id, -32601, f"Unknown: {method}")


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    raw = await request.body()
    try:
        body = json.loads(raw)
    except Exception:
        logger.exception("POST /mcp JSON parse failed")
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    result = await handle_request(body, request.app.state.skills, request.app.state.rpc_calls)

    if result is None:
        return Response(status_code=204)
    return JSONResponse(result)


def _emit_background_trace(msg: dict[str, Any]) -> None:
    """Emit ``tool_progress`` / ``tool_stopped`` for background tool-stream frames.

    A background skill's initial ``tools/call`` returns immediately while its
    physical work continues, so these frames prove that *RPC completion !=
    physical-operation completion*. Progress frames carry the originating
    ``progressToken`` and are correlated back to the invocation's mission/turn;
    stop frames carry only the tool name (and an acquire token), so they are
    correlated best-effort by tool name.
    """
    if app.state.trace_sink is None:
        return
    method = msg.get("method")
    params = msg.get("params") or {}

    if method == tool_stream.NOTIFICATIONS_PROGRESS_METHOD:
        token = params.get("progressToken")
        entry = app.state.trace_invocations.get(token)
        if entry is None:
            return  # uncorrelated progress (no recorded start) — skip
        ctx, tool_name = entry
        _emit_trace(
            MissionEvent.create(
                EventType.TOOL_PROGRESS,
                EventSource.TOOL_STREAM,
                ctx,
                tool_name=tool_name,
                summary=str(params.get("message") or ""),
            )
        )
    elif method == tool_stream.TOOL_STREAM_STOPPED_METHOD:
        tool_name = params.get("tool_name")
        entry = next(
            (
                (tok, val)
                for tok, val in list(app.state.trace_invocations.items())
                if val[1] == tool_name
            ),
            None,
        )
        if entry is None:
            return
        token, (ctx, _) = entry
        app.state.trace_invocations.pop(token, None)
        _emit_trace(
            MissionEvent.create(
                EventType.TOOL_STOPPED,
                EventSource.TOOL_STREAM,
                ctx,
                tool_name=tool_name,
                status=Status.UNVERIFIED,
                summary="background tool stopped",
            )
        )


def _sse_frame(data: dict[str, Any]) -> str:
    """Format a JSON-RPC message as an SSE ``event: message`` frame."""
    return f"event: message\ndata: {json.dumps(data)}\n\n"


def _fan_out_to_sse_queues(msg: dict[str, Any]) -> None:
    """LCM subscriber callback: forward a tool-stream frame to every active SSE client.

    Also releases capabilities held by a background skill when its tool-stream
    closes (signaled by a ``dimos/tool_stopped`` frame).
    """
    _emit_background_trace(msg)
    if msg.get("method") == tool_stream.TOOL_STREAM_STOPPED_METHOD:
        params = msg.get("params") or {}
        token = params.get("token")
        if token:
            released = app.state.cap_registry.release_by_token(token)
            if released:
                logger.info(
                    "Capabilities released on tool-stream stop",
                    holder=params.get("tool_name"),
                    token=token,
                    released=released,
                )
    loop = app.state.event_loop
    if loop is None:
        return
    for queue in list(app.state.sse_queues):
        try:
            asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
        except RuntimeError:
            pass


@app.get("/mcp")
async def mcp_sse_endpoint() -> StreamingResponse:
    """Persistent server-to-client SSE channel for MCP notifications.

    This is the Streamable-HTTP transport's out-of-band channel for
    server-initiated messages.  Every tool-stream update is fanned out here,
    so the subscription is live for the full client session and independent
    of any particular ``tools/call`` request.
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    # Remember the loop so the LCM subscriber (running on an LCM thread)
    # can schedule queue.put via run_coroutine_threadsafe.
    app.state.event_loop = asyncio.get_running_loop()
    app.state.sse_queues.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Initial comment flushes the response headers and unblocks
            # any synchronous client that's waiting on iter_lines().
            yield ": connected\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=_SSE_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if msg is None:
                    return
                yield _sse_frame(msg)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                app.state.sse_queues.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


class McpServer(Module):
    # Server-authoritative mission-trace channel (tool started/refused/
    # completed/failed and background progress/stopped). Wired non-invasively:
    # the sink is installed on start and removed on stop.
    mission_trace: Out[MissionEvent]

    _uvicorn_server: uvicorn.Server | None = None
    _serve_future: concurrent.futures.Future[None] | None = None
    _tool_stream_cleanup: Callable[[], None] | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._start_server()
        app.state.trace_run_id = MissionContext.for_mission("").run_id
        app.state.trace_invocations = {}
        app.state.trace_sink = self._publish_trace_event
        self._tool_stream_cleanup = tool_stream.subscribe(_fan_out_to_sse_queues)

    def _publish_trace_event(self, event: MissionEvent) -> None:
        self.mission_trace.publish(event)

    @rpc
    def stop(self) -> None:
        app.state.trace_sink = None
        if self._tool_stream_cleanup is not None:
            self._tool_stream_cleanup()
            self._tool_stream_cleanup = None

        for queue in list(app.state.sse_queues):
            try:
                queue.put_nowait(None)
            except Exception:
                pass
        app.state.sse_queues.clear()

        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            loop = self._loop
            if loop is not None and self._serve_future is not None:
                self._serve_future.result(timeout=5.0)
            self._uvicorn_server = None
            self._serve_future = None
        super().stop()

    @rpc
    def on_system_modules(self, modules: list[RPCClient]) -> None:
        # TODO: this is a bit hacky, also not thread-safe
        assert self.rpc is not None
        app.state.skills = [
            skill_info for module in modules for skill_info in (module.get_skills() or [])
        ]
        app.state.skills_by_name = {s.func_name: s for s in app.state.skills}
        app.state.rpc_calls = {
            skill_info.func_name: RpcCall(
                None, self.rpc, skill_info.func_name, skill_info.class_name, []
            )
            for skill_info in app.state.skills
        }

    @skill
    def server_status(self) -> str:
        """Get MCP server status: main process PID, deployed modules, and skill count."""
        from dimos.core.run_registry import get_most_recent

        skills: list[SkillInfo] = app.state.skills
        modules = list(dict.fromkeys(s.class_name for s in skills))
        entry = get_most_recent()
        pid = entry.pid if entry else os.getpid()
        return json.dumps(
            {
                "pid": pid,
                "modules": modules,
                "skills": [s.func_name for s in skills],
            }
        )

    @skill
    def list_modules(self) -> str:
        """List deployed modules and their skills."""
        skills: list[SkillInfo] = app.state.skills
        modules: dict[str, list[str]] = {}
        for s in skills:
            modules.setdefault(s.class_name, []).append(s.func_name)
        return json.dumps({"modules": modules})

    @skill
    def agent_send(self, message: str) -> str:
        """Send a message to the running DimOS agent over the active transport."""
        if not message:
            raise ValueError("Message cannot be empty")

        transport = make_transport("/human_input")
        try:
            transport.start()
            transport.publish(message)
            return f"Message sent to agent: {message[:100]}"
        finally:
            transport.stop()

    def _start_server(self, port: int | None = None) -> None:
        from dimos.core.global_config import global_config

        _port = port if port is not None else global_config.mcp_port
        _host = global_config.listen_host
        config = uvicorn.Config(app, host=_host, port=_port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        loop = self._loop
        assert loop is not None
        self._serve_future = asyncio.run_coroutine_threadsafe(server.serve(), loop)
