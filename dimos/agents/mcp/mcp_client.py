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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from threading import Event, RLock, Thread
import time
from typing import Any
import uuid

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from reactivex.disposable import Disposable
import requests

from dimos.agents.mcp import tool_stream
from dimos.agents.system_prompt import SYSTEM_PROMPT
from dimos.agents.trace.types import (
    EventSource,
    EventType,
    MissionContext,
    MissionEvent,
    Status,
)
from dimos.agents.utils import pretty_print_langchain_message
from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.rpc_client import RPCClient
from dimos.core.stream import In, Out
from dimos.utils.logging_config import setup_logger
from dimos.utils.sequential_ids import SequentialIds

logger = setup_logger()


class MessageSource(Enum):
    """Provenance of a message entering the agent queue.

    Only :attr:`EXTERNAL_INPUT` opens a new mission. Tool-progress updates and
    agent continuations re-enter the same queue but must stay inside the mission
    that produced them, so they are classified separately rather than by
    ``isinstance(message, HumanMessage)`` (which would misclassify a
    ``[tool:...]`` progress update as a fresh user request).
    """

    EXTERNAL_INPUT = "external_input"
    TOOL_PROGRESS = "tool_progress"
    CONTINUATION = "continuation"
    INTERNAL = "internal"


@dataclass
class _QueuedAgentMessage:
    """Internal envelope that preserves a queued message's provenance/mission."""

    message: BaseMessage
    source: MessageSource
    mission_id: str


def _message_text(content: Any) -> str:
    """Best-effort public text of a LangChain message content (str or parts).

    Only public/visible text is extracted; image parts and other artefacts are
    summarized, never the raw payload. Never includes hidden reasoning.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(f"[{item.get('type', 'part')}]")
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(content)


_RESPONSES_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _init_model(model_name: str) -> Any:
    """Initialize a model while preserving LangChain provider resolution."""
    if ":" in model_name or not model_name.startswith(_RESPONSES_REASONING_MODEL_PREFIXES):
        return init_chat_model(model=model_name)

    return ChatOpenAI(
        model=model_name,
        use_responses_api=True,
        reasoning={"effort": "medium", "summary": "auto"},
    )


class McpClientConfig(ModuleConfig):
    system_prompt: str | None = SYSTEM_PROMPT
    model: str = "gpt-5.6-luna"
    model_fixture: str | None = None
    mcp_server_url: str = "http://localhost:9990/mcp"


class McpClient(Module):
    config: McpClientConfig
    agent: Out[BaseMessage]
    human_input: In[str]
    agent_idle: Out[bool]
    # Non-invasive mission-trace channel. Client-authoritative events (mission
    # creation, input, agent turns, model tool selection, public assistant
    # messages) are published here and picked up by the MissionTraceRecorder.
    mission_trace: Out[MissionEvent]

    _lock: RLock
    # Guards mission/turn correlation state (`_active_mission_id`,
    # `_current_ctx`, `_turn_counter`). Kept separate from `_lock` because
    # `_lock` is held by the worker thread for the whole `state_graph.stream()`
    # call, during which tool execution (on a langgraph executor thread) calls
    # back into `_mcp_tool_call`/`_enqueue`. Reusing `_lock` there would
    # deadlock (RLock is only reentrant on the owning thread); this lock is only
    # ever held briefly and never across `stream()`.
    _ctx_lock: RLock
    _state_graph: CompiledStateGraph[Any, Any, Any, Any] | None
    _message_queue: Queue[_QueuedAgentMessage]
    _tool_registry: dict[str, dict[str, Any]]
    _history: list[BaseMessage]
    _thread: Thread
    _stop_event: Event
    _http_client: requests.Session
    _seq_ids: SequentialIds
    _tool_stream_cleanup: Callable[[], None] | None
    _run_id: str
    _active_mission_id: str | None
    _turn_counter: int
    _current_ctx: MissionContext | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = RLock()
        self._ctx_lock = RLock()
        self._state_graph = None
        self._message_queue = Queue()
        self._tool_registry = {}
        self._history = []
        self._thread = Thread(
            target=self._thread_loop,
            name=f"{self.__class__.__name__}-thread",
            daemon=True,
        )
        self._stop_event = Event()
        self._http_client = requests.Session()
        self._seq_ids = SequentialIds()
        self._tool_stream_cleanup = None
        # Resolved lazily so the run id reflects DIMOS_RUN_ID set by the CLI.
        self._run_id = MissionContext.for_mission("").run_id
        self._active_mission_id = None
        self._turn_counter = 0
        self._current_ctx = None

    def __reduce__(self) -> Any:
        return (self.__class__, (), {})

    def _emit(self, event: MissionEvent) -> None:
        """Publish a mission-trace event; never let tracing break execution."""
        try:
            self.mission_trace.publish(event)
        except Exception:
            logger.exception("failed to publish mission-trace event", event_type=event.event_type)

    def _mcp_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._seq_ids.next(),
            "method": method,
        }
        if params is not None:
            body["params"] = params

        resp = self._http_client.post(self.config.mcp_server_url, json=body, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"MCP error {data['error']['code']}: {data['error']['message']}")

        result: dict[str, Any] = data.get("result")
        return result

    def _mcp_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        progress_token = str(uuid.uuid4())
        # Reuse `progressToken` as the invocation id and carry the mission/turn
        # correlation as valid MCP request metadata so the server can emit
        # execution events under the same mission. `_meta` is passed through
        # untouched by servers that don't understand these keys.
        meta: dict[str, Any] = {"progressToken": progress_token}
        with self._ctx_lock:
            ctx = self._current_ctx
        if ctx is not None:
            meta["runId"] = ctx.run_id
            meta["missionId"] = ctx.mission_id
            if ctx.turn_id is not None:
                meta["turnId"] = ctx.turn_id
        return self._mcp_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
                "_meta": meta,
            },
        )

    def _on_tool_stream_message(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        params = msg.get("params") or {}
        if method == tool_stream.NOTIFICATIONS_PROGRESS_METHOD:
            text = params.get("message") or ""
            tool_name = (params.get("_meta") or {}).get("tool_name") or "tool"
        elif method == tool_stream.NOTIFICATIONS_MESSAGE_METHOD:
            text = params.get("data") or ""
            tool_name = params.get("logger") or "tool"
        else:
            return
        if not text:
            return
        # A tool-progress update re-enters the queue but belongs to the mission
        # that started the tool — it must NOT open a new mission.
        self._enqueue(
            HumanMessage(content=f"[tool:{tool_name}] {text}"),
            MessageSource.TOOL_PROGRESS,
        )

    def _enqueue(
        self, message: BaseMessage, source: MessageSource, mission_id: str | None = None
    ) -> None:
        """Wrap a message with its provenance/mission and enqueue it.

        External input opens a fresh mission; every other source inherits the
        currently active mission (falling back to a new id only if none exists).
        """
        with self._ctx_lock:
            if source is MessageSource.EXTERNAL_INPUT:
                mid = mission_id or uuid.uuid4().hex
                self._active_mission_id = mid
            else:
                mid = mission_id or self._active_mission_id or uuid.uuid4().hex
        self._message_queue.put(_QueuedAgentMessage(message, source, mid))

    def _fetch_tools(self, timeout: float = 60.0, interval: float = 1.0) -> list[StructuredTool]:
        result = self._try_fetch_tools(timeout=timeout, interval=interval)
        if result is None:
            raise RuntimeError(
                f"Failed to fetch tools from MCP server {self.config.mcp_server_url}"
            )

        raw_tools = result.get("tools", [])
        self._tool_registry = {t["name"]: t for t in raw_tools}
        tools = [self._mcp_tool_to_langchain(t) for t in raw_tools]

        if not tools:
            logger.warning("No tools found from MCP server.")
        else:
            tool_names = [t.name for t in tools]
            logger.info("Discovered tools from MCP server.", tools=tool_names, n_tools=len(tools))

        return tools

    def _try_fetch_tools(self, timeout: float, interval: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout

        while True:
            try:
                self._mcp_request("initialize")
                break
            except requests.ConnectionError:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(interval)

        return self._mcp_request("tools/list")

    def _mcp_tool_to_langchain(self, mcp_tool: dict[str, Any]) -> StructuredTool:
        name = mcp_tool["name"]
        description = mcp_tool.get("description", "")
        input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})

        def call_tool(**kwargs: Any) -> str:
            result = self._mcp_tool_call(name, kwargs)
            content = result.get("content", [])
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            text = "\n".join(parts)

            # Images need to be added to the history separately because they
            # cannot be included in the tool response for OpenAI models and
            # probably others.
            for item in content:
                if item.get("type") != "text":
                    uuid_ = str(uuid.uuid4())
                    text += f"Tool call started with UUID: {uuid_}. You will be updated with the result soon."
                    _append_image_to_history(self, name, uuid_, item)

            return text

        return StructuredTool(
            name=name,
            description=description,
            func=call_tool,
            args_schema=input_schema,
        )

    @rpc
    def start(self) -> None:
        super().start()

        def _on_human_input(string: str) -> None:
            self._enqueue(HumanMessage(content=string), MessageSource.EXTERNAL_INPUT)

        self.register_disposable(Disposable(self.human_input.subscribe(_on_human_input)))

        # Subscribe directly over LCM rather than through the server's GET
        # /mcp SSE channel.  HTTP would add a startup race: the first few
        # updates of a short-lived stream can fire before the SSE connection
        # is established.  External clients like Claude Code still use GET
        # /mcp, which the server fans out to from the same LCM topic.
        self._tool_stream_cleanup = tool_stream.subscribe(self._on_tool_stream_message)

    @rpc
    def on_system_modules(self, _modules: list[RPCClient]) -> None:
        tools = self._fetch_tools()

        if self.config.model_fixture is not None:
            from dimos.agents.testing import MockModel

            model = MockModel(json_path=self.config.model_fixture)
        else:
            model = _init_model(self.config.model)

        with self._lock:
            self._state_graph = create_agent(
                model=model,
                tools=tools,
                system_prompt=self.config.system_prompt,
            )
            if not self._thread.is_alive():
                self._thread.start()

    @rpc
    def stop(self) -> None:
        # Unsubscribe first so no new tool-stream messages can arrive while
        # the worker thread is draining and joining.
        if self._tool_stream_cleanup is not None:
            self._tool_stream_cleanup()
            self._tool_stream_cleanup = None
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
        self._http_client.close()
        super().stop()

    @rpc
    def add_message(self, message: BaseMessage) -> None:
        self._enqueue(message, MessageSource.INTERNAL)

    @rpc
    def dispatch_continuation(
        self, continuation: dict[str, Any], continuation_context: dict[str, Any]
    ) -> None:
        """Execute a tool continuation with detection data, bypassing the LLM.

        Called by trigger tools (e.g. look_out_for) to immediately invoke a
        follow-up tool when a detection fires, without waiting for the LLM to
        reason about the next action.

        Args:
            continuation: ``{"tool": "<name>", "args": {…}}`` — the tool to
                call and its arguments.  Argument values that are strings
                starting with ``$`` are treated as template variables and
                resolved against *continuation_context* (e.g. ``"$bbox"``).
            continuation_context: runtime detection data, e.g.
                ``{"bbox": [x1, y1, x2, y2], "label": "person"}``.
        """
        tool_name = continuation.get("tool")
        if not tool_name:
            self._enqueue(
                HumanMessage(f"Continuation failed: missing 'tool' key in {continuation}"),
                MessageSource.CONTINUATION,
            )
            return

        if tool_name not in self._tool_registry:
            self._enqueue(
                HumanMessage(f"Continuation failed: tool '{tool_name}' not found"),
                MessageSource.CONTINUATION,
            )
            return

        tool_args: dict[str, Any] = dict(continuation.get("args", {}))

        # Substitute $-prefixed template variables from continuation_context
        for key, value in tool_args.items():
            if isinstance(value, str) and value.startswith("$"):
                context_key = value[1:]
                if context_key in continuation_context:
                    tool_args[key] = continuation_context[context_key]

        try:
            result = self._mcp_tool_call(tool_name, tool_args)
            content = result.get("content", [])
            parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            text = "\n".join(parts)
        except Exception as e:
            self._enqueue(
                HumanMessage(f"Continuation '{tool_name}' failed with error: {e}"),
                MessageSource.CONTINUATION,
            )
            return

        label = continuation_context.get("label", "unknown")
        self._enqueue(
            HumanMessage(
                f"Automatically executed '{tool_name}' as a continuation of lookout "
                f"detection (detected: {label}). Result: {text or 'started'}"
            ),
            MessageSource.CONTINUATION,
        )

    def _thread_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._message_queue.get(timeout=0.5)
            except Empty:
                continue

            with self._lock:
                if not self._state_graph:
                    raise ValueError("No state graph initialized")
                self._process_message(self._state_graph, item)

    def _process_message(
        self, state_graph: CompiledStateGraph[Any, Any, Any, Any], item: _QueuedAgentMessage
    ) -> None:
        message = item.message

        # One dequeued message == one agent turn. External input additionally
        # opens a mission. `_current_ctx` is read by `_mcp_tool_call` (from the
        # langgraph tool-executor thread) to stamp outgoing MCP `_meta`, so it
        # is guarded by `_ctx_lock` (never held across `stream()`).
        with self._ctx_lock:
            self._turn_counter += 1
            ctx = MissionContext(
                run_id=self._run_id, mission_id=item.mission_id, turn_id=self._turn_counter
            )
            self._current_ctx = ctx

        if item.source is MessageSource.EXTERNAL_INPUT:
            text = _message_text(message.content)
            self._emit(
                MissionEvent.create(
                    EventType.MISSION_STARTED, EventSource.CLIENT, ctx, summary=text
                )
            )
            self._emit(
                MissionEvent.create(
                    EventType.INPUT_RECEIVED,
                    EventSource.CLIENT,
                    ctx,
                    summary=text,
                    attributes={"source": item.source.value},
                )
            )

        self.agent_idle.publish(False)
        self._history.append(message)
        pretty_print_langchain_message(message)
        self.agent.publish(message)

        self._emit(
            MissionEvent.create(
                EventType.AGENT_TURN_STARTED,
                EventSource.CLIENT,
                ctx,
                attributes={"source": item.source.value},
            )
        )

        for update in state_graph.stream({"messages": self._history}, stream_mode="updates"):
            for node_output in update.values():
                for msg in node_output.get("messages", []):
                    self._history.append(msg)
                    pretty_print_langchain_message(msg)
                    self.agent.publish(msg)
                    self._emit_agent_output(msg, ctx)

        self._emit(MissionEvent.create(EventType.AGENT_TURN_COMPLETED, EventSource.CLIENT, ctx))

        if self._message_queue.empty():
            self.agent_idle.publish(True)

    def _emit_agent_output(self, msg: BaseMessage, ctx: MissionContext) -> None:
        """Emit client-authoritative events for a message produced by the graph.

        Records model tool selections and public assistant messages. Never
        inspects or persists hidden reasoning — only the visible content and the
        selected tool names/arguments.
        """
        tool_calls = getattr(msg, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            self._emit(
                MissionEvent.create(
                    EventType.TOOL_SELECTED,
                    EventSource.CLIENT,
                    ctx,
                    tool_name=name,
                    summary=name,
                    attributes={"args": args} if args else None,
                )
            )

        if isinstance(msg, AIMessage) and not tool_calls:
            text = _message_text(msg.content)
            if text:
                self._emit(
                    MissionEvent.create(
                        EventType.AGENT_MESSAGE,
                        EventSource.CLIENT,
                        ctx,
                        status=Status.UNVERIFIED,
                        summary=text,
                    )
                )


def _append_image_to_history(
    mcp_client: McpClient, func_name: str, uuid_: str, result: Any
) -> None:
    mcp_client.add_message(
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"This is the artefact for the '{func_name}' tool with UUID:={uuid_}.",
                },
                result,
            ]
        )
    )
