# -*- coding: utf-8 -*-
#
# Lazy-loading tool index.
#
# Each tool function is loaded on first access so that heavyweight modules
# (e.g. browser_control ≈ 2 600 lines, desktop_screenshot with mss, …) are
# never imported unless actually referenced.

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentscope.tool import execute_python_code, view_text_file, write_text_file

    from .browser_control import browser_use
    from .desktop_screenshot import desktop_screenshot
    from .file_io import append_file, edit_file, read_file, write_file
    from .file_search import glob_search, grep_search
    from .get_current_time import get_current_time
    from .memory_search import create_memory_search_tool
    from .send_file import send_file_to_user
    from .shell import execute_shell_command

# (module_relative_path, attribute_name)
_LAZY_MAP: dict[str, tuple[str, str]] = {
    "execute_python_code": ("agentscope.tool", "execute_python_code"),
    "view_text_file": ("agentscope.tool", "view_text_file"),
    "write_text_file": ("agentscope.tool", "write_text_file"),
    "read_file": (".file_io", "read_file"),
    "write_file": (".file_io", "write_file"),
    "edit_file": (".file_io", "edit_file"),
    "append_file": (".file_io", "append_file"),
    "grep_search": (".file_search", "grep_search"),
    "glob_search": (".file_search", "glob_search"),
    "execute_shell_command": (".shell", "execute_shell_command"),
    "send_file_to_user": (".send_file", "send_file_to_user"),
    "browser_use": (".browser_control", "browser_use"),
    "desktop_screenshot": (".desktop_screenshot", "desktop_screenshot"),
    "create_memory_search_tool": (".memory_search", "create_memory_search_tool"),
    "get_current_time": (".get_current_time", "get_current_time"),
}

__all__ = list(_LAZY_MAP)


def __getattr__(name: str):
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(mod_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
