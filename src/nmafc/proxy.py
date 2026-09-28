"""OpenAI-compatible client wrapper with transparent memory.

`MemoryOpenAI` wraps an `openai.OpenAI` or `openai.AsyncOpenAI` client and
interposes only on `chat.completions.create`:

    from nmafc.proxy import MemoryOpenAI
    from nmafc.wrapper import NeuromorphicMemory
    from openai import AsyncOpenAI

    memory = await NeuromorphicMemory.from_env()  # or from_config(...)
    client = MemoryOpenAI(memory, client=AsyncOpenAI())

    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "..."}],
    )

The interposition is a `recall -> inject -> forward -> remember` pipeline:

1. **recall** — the last user message is recalled against the store
   (read-only, so a lookup never mutates memory).
2. **inject** — the bounded context is prepended as a `system` message.
3. **forward** — the request is passed through untouched otherwise.
4. **remember** — the assistant reply is stored extract-only afterwards. With
   `stream=True` the stream is forwarded as-is and extraction runs on the
   fully reassembled transcript once the stream is exhausted, so nothing is
   stored mid-stream.

Everything else on the client is delegated through `__getattr__`, so
`client.models.list()`, `client.embeddings.create()`, custom headers, etc.
keep working without knowledge of memory.
"""

from __future__ import annotations

import os
import traceback

from nmafc.wrapper import NeuromorphicMemory, SyncNeuromorphicMemory

DEFAULT_PREAMBLE = """## Relevant memories
{context}"""


def _pluck(kwargs: dict, name: str, default=None):
    """Pop a memory-only kwarg so it never reaches the underlying client."""
    return kwargs.pop(name, default)


def _last_text(messages: list[dict]) -> str:
    """Text of the most recent user message, or '' if none exists."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content") or ""
            if isinstance(content, list):
                # Multimodal content arrays: keep only text parts.
                return "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content) if content else ""
    return ""


def _response_text(response) -> str:
    """Extract assistant text from a non-stream response object."""
    try:
        choices = response.choices
    except AttributeError:
        return ""
    parts: list[str] = []
    for choice in choices:
        msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                parts.append(str(content))
    return "".join(parts)


def _chunk_text(chunk) -> str:
    """Extract the incremental text carried by one stream chunk."""
    try:
        choices = getattr(chunk, "choices", None)
    except AttributeError:
        return ""
    if not choices:
        return ""
    choice = choices[0]
    msg = getattr(choice, "delta", None) or getattr(choice, "message", None)
    if msg is None:
        return ""
    content = getattr(msg, "content", None)
    return str(content) if content else ""


class MemoryOpenAI:
    """OpenAI client wrapper that adds recall/remember around chat completions.

    Args:
        memory: A `NeuromorphicMemory` (async client) or
            `SyncNeuromorphicMemory` (sync client).
        client: An existing `OpenAI`/`AsyncOpenAI` instance to wrap, or None to
            build one from `openai_kwargs` (async iff `memory` is async).
        system_preamble: Template wrapping `{context}`; injected as a system
            message when recall finds anything. Set to None to disable recall
            injection entirely.
    """

    def __init__(
        self,
        memory: NeuromorphicMemory | SyncNeuromorphicMemory,
        *,
        client=None,
        system_preamble: str | None = DEFAULT_PREAMBLE,
        **openai_kwargs,
    ) -> None:
        self._memory = memory
        self._async = isinstance(memory, NeuromorphicMemory)
        self._preamble = system_preamble
        if client is None:
            import openai

            if self._async:
                client = openai.AsyncOpenAI(**openai_kwargs)
            else:
                client = openai.OpenAI(**openai_kwargs)
        self._client = client

    def __getattr__(self, name: str):
        # Anything that is not the chat completions seam passes through.
        # Guard with object.__getattribute__ so attribute access on the
        # wrapper itself cannot recurse into __getattr__.
        if name == "chat":
            return _ChatSeam(self)
        client = object.__getattribute__(self, "_client")
        return getattr(client, name)


class _ChatSeam:
    """Interposes `completions.create`; delegates the rest of `client.chat`."""

    def __init__(self, proxy: MemoryOpenAI) -> None:
        self._proxy = proxy
        self._chat = proxy._client.chat

    def __getattr__(self, name: str):
        if name == "completions":
            return _CompletionsSeam(self._proxy, self._chat.completions)
        return getattr(self._chat, name)


class _CompletionsSeam:
    def __init__(self, proxy: MemoryOpenAI, completions) -> None:
        self._proxy = proxy
        self._completions = completions

    def __getattr__(self, name: str):
        if name == "create":
            if self._proxy._async:
                return self._create_async
            return self._create_sync
        return getattr(self._completions, name)

    # ── recall / inject / forward / remember ────────────────────────────

    def _extract_params(self, kwargs):
        """Pull memory-only kwargs and the original transcript out of the call.

        Returns (original_messages, query, agent_id, conversation_id,
        remember_on, recall_on). The original list is returned immutable --
        callers copy it forward after injection.
        """
        messages = list(kwargs.get("messages") or [])
        if not messages:
            raise TypeError("chat.completions.create missing required `messages`")
        return (
            messages,
            _last_text(messages),
            _pluck(kwargs, "memory_agent_id"),
            _pluck(kwargs, "memory_conversation_id"),
            _pluck(kwargs, "memory_turn"),
            _pluck(kwargs, "memory_remember", True),
            _pluck(kwargs, "memory_recall", True),
        )

    def _inject(self, kwargs, messages, context: str) -> None:
        if context:
            kwargs["messages"] = [
                {"role": "system", "content": self._proxy._preamble.format(context=context)},
                *messages,
            ]
        else:
            kwargs["messages"] = messages

    def _post_remember(self, original: list[dict], reply: str, agent_id, conversation_id) -> None:
        """Sync-path remember: used by sync clients and the sync stream tail."""
        if not reply:
            return
        transcript = self._transcript(original, reply)
        try:
            self._proxy._memory.remember(
                transcript, agent_id=agent_id, conversation_id=conversation_id,
            )
        except Exception:
            self._report_memory_error()

    async def _apost_remember(
        self, original: list[dict], reply: str, agent_id, conversation_id
    ) -> None:
        """Async-path remember: used by async clients and the async stream tail."""
        if not reply:
            return
        transcript = self._transcript(original, reply)
        try:
            await self._proxy._memory.remember(
                transcript, agent_id=agent_id, conversation_id=conversation_id,
            )
        except Exception:
            self._report_memory_error()

    @staticmethod
    def _transcript(original: list[dict], reply: str) -> list[dict]:
        return [
            {"role": m.get("role") or "user", "content": m.get("content") or ""}
            for m in original
            if m.get("role") and m.get("content")
        ] + [{"role": "assistant", "content": reply}]

    @staticmethod
    def _report_memory_error() -> None:
        # Memory must never break the caller's LLM call. Log the traceback
        # through the provider side is too noisy; emit to stderr only when
        # NMAFC_PROXY_DEBUG is set, and remember failures are silent otherwise.
        if os.environ.get("NMAFC_PROXY_DEBUG"):
            traceback.print_exc()

    def _create_sync(self, *args, **kwargs):
        messages, query, agent_id, conversation_id, turn, remember_on, recall_on = (
            self._extract_params(kwargs)
        )
        context = ""
        if recall_on and self._proxy._preamble and query:
            result = self._proxy._memory.recall(
                query, agent_id=agent_id, conversation_id=conversation_id, turn=turn
            )
            if result is not None and result.context:
                context = result.context
        self._inject(kwargs, messages, context)
        stream = kwargs.get("stream", False)

        if stream:
            def _iter():
                reply_parts: list[str] = []
                try:
                    for chunk in self._completions.create(*args, **kwargs):
                        text = _chunk_text(chunk)
                        if text:
                            reply_parts.append(text)
                        yield chunk
                finally:
                    if remember_on:
                        self._post_remember(
                            messages, "".join(reply_parts), agent_id, conversation_id
                        )
            return _iter()

        response = self._completions.create(*args, **kwargs)
        if remember_on:
            self._post_remember(messages, _response_text(response), agent_id, conversation_id)
        return response

    async def _create_async(self, *args, **kwargs):
        messages, query, agent_id, conversation_id, turn, remember_on, recall_on = (
            self._extract_params(kwargs)
        )
        context = ""
        if recall_on and self._proxy._preamble and query:
            result = await self._proxy._memory.recall(
                query, agent_id=agent_id, conversation_id=conversation_id, turn=turn
            )
            if result is not None and result.context:
                context = result.context
        self._inject(kwargs, messages, context)
        stream = kwargs.get("stream", False)

        if stream:
            async def _aiter():
                reply_parts: list[str] = []
                try:
                    async for chunk in await self._completions.create(*args, **kwargs):
                        text = _chunk_text(chunk)
                        if text:
                            reply_parts.append(text)
                        yield chunk
                finally:
                    if remember_on:
                        await self._apost_remember(
                            messages, "".join(reply_parts), agent_id, conversation_id
                        )
            return _aiter()

        response = await self._completions.create(*args, **kwargs)
        if remember_on:
            await self._apost_remember(
                messages, _response_text(response), agent_id, conversation_id
            )
        return response


__all__ = ["MemoryOpenAI", "DEFAULT_PREAMBLE"]
