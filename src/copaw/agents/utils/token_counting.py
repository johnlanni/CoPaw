# -*- coding: utf-8 -*-
"""Token counting utilities for managing context windows.

This module provides token counting functionality for estimating
message token usage with Qwen tokenizer.

Uses the lightweight ``tokenizers`` library (Rust-based, ~20 MB RSS) instead
of the full ``transformers`` stack (~200 MB RSS).  The public API is
unchanged so that MemoryManager / ReMeCopaw / CoPawInMemoryMemory can
consume the returned counter transparently.
"""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_token_counter = None


# ---------------------------------------------------------------------------
# Lightweight token counter (drop-in for HuggingFaceTokenCounter)
# ---------------------------------------------------------------------------

class _AutoTokenizerCompat:
    """Thin adapter that gives a ``tokenizers.Tokenizer`` the same
    surface API as ``transformers.AutoTokenizer`` (encode, chat_template,
    apply_chat_template) so that downstream code (reme, agentscope) keeps
    working without importing transformers."""

    def __init__(self, raw_tokenizer: Any, chat_template: str = "") -> None:
        self._tok = raw_tokenizer
        self.chat_template = chat_template

    def encode(self, text: str, **_kwargs: Any) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], **_kwargs: Any) -> str:
        return self._tok.decode(ids)

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_kwargs: Any,
    ) -> Any:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            elif isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("text")
                ]
                parts.append(
                    f"<|im_start|>{role}\n{''.join(text_parts)}<|im_end|>"
                )
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "\n".join(parts)
        if tokenize:
            return [self._tok.encode(text).ids]
        return text


class LightweightTokenCounter:
    """Token counter backed by the ``tokenizers`` library.

    API-compatible with ``agentscope.token.HuggingFaceTokenCounter``:
    * ``.tokenizer.encode(text) -> list[int]``
    * ``await .count(messages) -> int``
    """

    def __init__(self, tokenizer_dir: str) -> None:
        from tokenizers import Tokenizer as _Tokenizer

        json_path = Path(tokenizer_dir) / "tokenizer.json"
        raw = _Tokenizer.from_file(str(json_path))

        chat_template = ""
        config_path = Path(tokenizer_dir) / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as fh:
                chat_template = json.load(fh).get("chat_template", "")

        self.tokenizer = _AutoTokenizerCompat(raw, chat_template)

    async def count(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **_kwargs: Any,
    ) -> int:
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )[0]
        return len(tokenized)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

def _get_token_counter():
    """Get or initialize the global token counter instance.

    Prefers the bundled Qwen tokenizer files shipped with copaw.  Falls
    back to ``HuggingFaceTokenCounter`` (which pulls in ``transformers``)
    only when the local tokenizer is missing.
    """
    global _token_counter
    if _token_counter is not None:
        return _token_counter

    local_tokenizer_path = Path(__file__).parent.parent.parent / "tokenizer"

    if (
        local_tokenizer_path.exists()
        and (local_tokenizer_path / "tokenizer.json").exists()
    ):
        logger.info(
            "Using lightweight Qwen tokenizer from %s", local_tokenizer_path
        )
        _token_counter = LightweightTokenCounter(str(local_tokenizer_path))
    else:
        # Fallback: use HuggingFaceTokenCounter (loads transformers)
        from agentscope.token import HuggingFaceTokenCounter

        logger.info(
            "Local tokenizer not found, falling back to "
            "HuggingFaceTokenCounter (requires transformers)",
        )
        _token_counter = HuggingFaceTokenCounter(
            pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            use_mirror=True,
            use_fast=True,
            trust_remote_code=True,
        )

    logger.debug("Token counter initialized")
    return _token_counter


def _extract_text_from_messages(messages: list[dict]) -> str:
    """Extract text content from messages and concatenate into a string.
    NOTE: This code is deprecated and will be removed in the future.

    Handles various message formats:
    - Simple string content: {"role": "user", "content": "hello"}
    - List content with text blocks:
      {"role": "user", "content": [{"type": "text", "text": "hello"}]}

    Args:
        messages: List of message dictionaries in chat format.

    Returns:
        str: Concatenated text content from all messages.
    """
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    # Support {"type": "text", "text": "..."} format
                    text = block.get("text") or block.get("content", "")
                    if text:
                        parts.append(str(text))
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


# pylint: disable=too-many-branches,too-many-nested-blocks
def _extract_text_from_messages_v2(
    messages: list[dict],
) -> str:
    """Extract text content from messages and concatenate into a string.

    Handles various message formats:
    - Simple string content: {"role": "user", "content": "hello"}
    - List content with text blocks:
      {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    - List content with tool_result blocks:
      {"role": "user", "content": [{"type": "tool_result", "output": "..."}]}

    Args:
        messages: List of message dictionaries in chat format.

    Returns:
        str: Concatenated text content from all messages.
    """
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "tool_result":
                        output = block.get("output", "")
                        if isinstance(output, str) and output:
                            parts.append(output)
                        elif isinstance(output, list):
                            for sub in output:
                                if isinstance(sub, dict):
                                    sub_text = sub.get("text") or sub.get(
                                        "content",
                                        "",
                                    )
                                    if sub_text:
                                        parts.append(str(sub_text))
                    else:
                        text = block.get("text") or block.get("content", "")
                        if text:
                            parts.append(str(text))
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts)


async def count_message_tokens(
    messages: list[dict],
) -> int:
    """Count tokens in messages using the tokenizer.

    Extracts text content from messages and uses the tokenizer to
    count tokens. This approach is more robust across different model
    types than using apply_chat_template directly.

    Args:
        messages: List of message dictionaries in chat format.

    Returns:
        int: The estimated number of tokens in the messages.

    Raises:
        RuntimeError: If token counter fails to initialize.
    """
    token_counter = _get_token_counter()
    text = _extract_text_from_messages_v2(messages)
    token_ids = token_counter.tokenizer.encode(text)
    token_count = len(token_ids)
    logger.debug(
        "Counted %d tokens in %d messages",
        token_count,
        len(messages),
    )
    return token_count


async def safe_count_message_tokens(
    messages: list[dict],
) -> int:
    """Safely count tokens in messages with fallback estimation.

    This is a wrapper around count_message_tokens that catches exceptions
    and falls back to a character-based estimation (len // 4) if the
    tokenizer fails.

    Args:
        messages: List of message dictionaries in chat format.

    Returns:
        int: The estimated number of tokens in the messages.
    """
    try:
        return await count_message_tokens(messages)
    except Exception as e:
        # Fallback to character-based estimation
        text = _extract_text_from_messages_v2(messages)
        estimated_tokens = len(text) // 4
        logger.warning(
            "Failed to count tokens: %s, using estimated_tokens=%d",
            e,
            estimated_tokens,
        )
        return estimated_tokens


def safe_count_str_tokens(text: str) -> int:
    """Safely count tokens in a string with fallback estimation.

    Uses the tokenizer to count tokens in the given text. If the tokenizer
    fails, falls back to a character-based estimation (len // 4).

    Args:
        text: The string to count tokens for.

    Returns:
        int: The estimated number of tokens in the string.
    """
    try:
        token_counter = _get_token_counter()
        token_ids = token_counter.tokenizer.encode(text)
        token_count = len(token_ids)
        logger.debug(
            "Counted %d tokens in string of length %d",
            token_count,
            len(text),
        )
        return token_count
    except Exception as e:
        # Fallback to character-based estimation
        estimated_tokens = len(text) // 4
        logger.warning(
            "Failed to count string tokens: %s, using estimated_tokens=%d",
            e,
            estimated_tokens,
        )
        return estimated_tokens
