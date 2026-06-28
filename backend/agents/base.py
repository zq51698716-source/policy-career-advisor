"""
BaseAgent — shared ReAct loop + streaming bridge for all specialised agents.

Every agent inherits from BaseAgent and only needs to define:
  - system_prompt: str
  - tools: list[str]  (tool names registered in tools/__init__.py)

The base class handles:
  - Building the message list from query + history
  - The ReAct loop (LLM → tool calls → inject results → loop)
  - True streaming via asyncio.Queue + background thread bridge
  - Parallel tool execution via asyncio.gather
  - SSE event formatting
"""

import json
import asyncio
from typing import AsyncGenerator, Any

from anthropic import Anthropic
from anthropic.types import MessageParam

from backend.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOOL_CALLS
from backend.tools import get_tool_schemas, execute_tool

client = Anthropic(api_key=ANTHROPIC_API_KEY)


class BaseAgent:
    """Reusable agent base with ReAct loop and true streaming."""

    # --- Subclass overrides ---
    system_prompt: str = ""
    tools: list[str] = []  # Tool names registered in tools/__init__.py
    max_tool_rounds: int = min(MAX_TOOL_CALLS, 5)

    # Internals
    _model: str = CLAUDE_MODEL
    _tool_schemas: list[dict] = []

    def __init__(self):
        self._tool_schemas = get_tool_schemas(self.tools)

    # ================================================================
    # Public API
    # ================================================================
    async def chat_stream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """ReAct loop with true streaming — yields SSE-formatted strings."""
        if history is None:
            history = []

        messages: list[MessageParam] = self._build_messages(query, history)

        try:
            continue_rounds = 0
            for _round in range(self.max_tool_rounds):
                tool_uses: dict[str, dict] = {}
                stop_reason = "end_turn"
                text_content = ""  # Accumulate text for auto-continue on max_tokens

                async for event in self._stream_claude_response(messages):
                    yield event

                    # Parse for metadata
                    parsed = self._try_parse_sse(event)
                    if parsed is None:
                        continue

                    etype = parsed["type"]
                    if etype == "content":
                        text_content += parsed["data"]
                    elif etype == "tool_start":
                        tu_id = parsed["data"]["id"]
                        tool_uses[tu_id] = {
                            "id": tu_id,
                            "name": parsed["data"]["name"],
                            "input_json": "",
                        }
                    elif etype == "tool_delta":
                        tu_id = parsed["data"]["id"]
                        if tu_id in tool_uses:
                            tool_uses[tu_id]["input_json"] += parsed["data"]["delta"]
                    elif etype == "meta":
                        stop_reason = parsed["data"].get("stop_reason", "end_turn")
                    elif etype == "error":
                        return  # Fatal — stop

                # --- Auto-continue on max_tokens ---
                if stop_reason == "max_tokens" and text_content.strip() and continue_rounds < 3:
                    continue_rounds += 1
                    messages.append({"role": "assistant", "content": text_content})
                    messages.append({"role": "user", "content": "请继续完成上面的回答，从截断处接着写。"})
                    yield self._sse("thinking", "回答较长，正在继续生成...")
                    continue

                # --- Tool execution phase ---
                if stop_reason == "tool_use" and tool_uses:
                    # Build assistant blocks & tool list
                    tool_list = []
                    assistant_blocks = []

                    for tu in tool_uses.values():
                        try:
                            tu["input"] = json.loads(tu.pop("input_json", "{}"))
                        except json.JSONDecodeError:
                            tu["input"] = {}
                        tool_list.append(tu)
                        assistant_blocks.append({
                            "type": "tool_use",
                            "id": tu["id"],
                            "name": tu["name"],
                            "input": tu["input"],
                        })

                    messages.append({"role": "assistant", "content": assistant_blocks})

                    # Parallel execution
                    if len(tool_list) > 1:
                        yield self._sse("thinking", f"🔍 正在并行搜索 {len(tool_list)} 个来源...")

                    results = await asyncio.gather(
                        *[asyncio.to_thread(execute_tool, tu["name"], tu["input"]) for tu in tool_list]
                    )

                    user_blocks = []
                    for result_str, tu in zip(results, tool_list):
                        user_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tu["id"],
                            "content": result_str,
                        })
                        # Notify frontend
                        try:
                            robj = json.loads(result_str)
                            n = len(robj.get("results", []))
                            yield self._sse("tool_result", {
                                "tool_name": tu["name"],
                                "result_count": n,
                                "summary": f"搜索完成，获得 {n} 条结果",
                            })
                        except json.JSONDecodeError:
                            pass

                    messages.append({"role": "user", "content": user_blocks})
                    yield self._sse("thinking", "正在综合搜索结果...")
                    continue

                # No more tool calls — done
                yield self._sse("done", "")
                return

            # Max rounds reached
            yield self._sse("done", "")

        except Exception as e:
            yield self._sse("error", str(e))

    # ================================================================
    # Streaming bridge (thread → asyncio)
    # ================================================================
    async def _stream_claude_response(
        self,
        messages: list[MessageParam],
    ) -> AsyncGenerator[str, None]:
        """
        Stream Claude's response through an asyncio.Queue bridge.

        Claude SDK's sync streaming runs in a background thread;
        call_soon_threadsafe pushes SSE events into an asyncio.Queue
        consumed by this async generator — zero polling latency.
        """
        aq: asyncio.Queue[str] = asyncio.Queue(maxsize=1024)
        loop = asyncio.get_event_loop()

        loop.run_in_executor(
            None,
            self._run_claude_stream,
            messages,
            aq,
            loop,
        )

        while True:
            msg = await aq.get()
            if msg == "__END__":
                break
            yield msg

    def _run_claude_stream(
        self,
        messages: list[MessageParam],
        aq: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ):
        """Background thread: Claude streaming → thread-safe queue injection."""
        block_ids: dict[int, str] = {}

        def put(msg: str):
            # Use run_coroutine_threadsafe instead of call_soon_threadsafe + put_nowait.
            # put_nowait raises QueueFull when the queue hits maxsize (256), which
            # kills the stream mid-output.  run_coroutine_threadsafe schedules an
            # async put() that waits gracefully when the consumer is slow.
            asyncio.run_coroutine_threadsafe(aq.put(msg), loop)

        try:
            with client.messages.stream(
                model=self._model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self._tool_schemas,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            put(self._sse("content", delta.text))
                        elif delta.type == "input_json_delta":
                            bid = block_ids.get(event.index, "unknown")
                            put(self._sse("tool_delta", {
                                "id": bid,
                                "delta": delta.partial_json,
                            }))

                    elif event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            block_ids[event.index] = block.id
                            put(self._sse("tool_start", {
                                "id": block.id,
                                "name": block.name,
                            }))

                final = stream.get_final_message()
                put(self._sse("meta", {"stop_reason": final.stop_reason}))

        except Exception as e:
            put(self._sse("error", str(e)))
        finally:
            put("__END__")

    # ================================================================
    # Message building
    # ================================================================
    def _build_messages(
        self,
        query: str,
        history: list[dict[str, str]],
    ) -> list[MessageParam]:
        messages: list[MessageParam] = []
        for msg in history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif role == "user":
                messages.append({"role": "user", "content": content})

        if messages and messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "请继续"})

        messages.append({"role": "user", "content": query})
        return messages

    # ================================================================
    # SSE encoding / decoding
    # ================================================================
    @staticmethod
    def _sse(event_type: str, data: Any) -> str:
        """Format an SSE event string."""
        payload = {"type": event_type, "data": data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _try_parse_sse(sse_msg: str) -> dict | None:
        """Attempt to parse an SSE message into {type, data}. Returns None on failure."""
        if not sse_msg.startswith("data: "):
            return None
        try:
            return json.loads(sse_msg.split("data: ", 1)[1])
        except (json.JSONDecodeError, IndexError):
            return None
