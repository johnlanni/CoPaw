# -*- coding: utf-8 -*-
"""Runner module with chat manager for coordinating repository.

Sub-modules are lazily imported so that ``from copaw.app.runner import
AgentRunner`` no longer forces the heavy ``api`` / ``manager`` / ``repo``
modules (which pull in FastAPI, agentscope session, etc.) to load.
"""
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import router
    from .manager import ChatManager
    from .models import ChatHistory, ChatsFile, ChatSpec
    from .repo import BaseChatRepository, JsonChatRepository
    from .runner import AgentRunner

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "AgentRunner": (".runner", "AgentRunner"),
    "router": (".api", "router"),
    "ChatManager": (".manager", "ChatManager"),
    "ChatSpec": (".models", "ChatSpec"),
    "ChatHistory": (".models", "ChatHistory"),
    "ChatsFile": (".models", "ChatsFile"),
    "BaseChatRepository": (".repo", "BaseChatRepository"),
    "JsonChatRepository": (".repo", "JsonChatRepository"),
}

__all__ = list(_LAZY_MAP)


def __getattr__(name: str):
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(mod_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
