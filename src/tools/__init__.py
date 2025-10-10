"""
OpenWebUI-compatible tool system.

This module provides automatic discovery and integration of OpenWebUI-format tools
for use with LLM providers (Gemini, Ollama).
"""

from .executor import ToolExecutor
from .loader import ToolLoader
from .manager import ToolManager
from .registry import ToolRegistry

__all__ = ["ToolLoader", "ToolRegistry", "ToolExecutor", "ToolManager"]
