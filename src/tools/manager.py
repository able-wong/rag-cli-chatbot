"""
Tool manager for orchestrating the tool system.
"""

import logging
from typing import Any, Dict, List

from .executor import ToolExecutor
from .loader import ToolLoader
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolManager:
    """Orchestrates tool system - high-level API for CLI."""

    def __init__(self, config: Dict[str, Any], llm_provider: str):
        """
        Initialize tool system.

        Args:
            config: tools section from config.yaml
            llm_provider: "gemini" or "ollama" for schema conversion
        """
        self.config = config
        self.llm_provider = llm_provider.lower()
        self.tools_directory = config.get("directory", "./tools")

        self.loader = ToolLoader(self.tools_directory)
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

        self._initialized = False

    def initialize(self):
        """
        Load and register all tools.

        Called during RAGCLI initialization.
        Logs summary: "Loaded N tools: tool1, tool2, ..."
        """
        if self._initialized:
            logger.warning("ToolManager already initialized")
            return

        # Load tools
        valves_config = self.config.get("valves", {})
        loaded_tools = self.loader.load_all_tools(valves_config)

        # Register tools
        for module_name, tools_instance in loaded_tools.items():
            self.registry.register_tool_instance(module_name, tools_instance)

        # Log summary
        tool_names = self.registry.get_tool_names()
        if tool_names:
            logger.info(f"✓ Loaded {len(tool_names)} tools: {', '.join(tool_names)}")
        else:
            logger.warning("⚠️  No tools loaded from directory")

        self._initialized = True

    def get_tools_for_llm(self) -> List[Any]:
        """
        Get tool schemas for LLM provider.

        Returns:
            Gemini or Ollama format based on provider
        """
        if not self._initialized:
            raise RuntimeError("ToolManager not initialized. Call initialize() first.")

        if self.llm_provider == "gemini":
            return self.registry.to_gemini_tools()
        elif self.llm_provider == "ollama":
            return self.registry.to_ollama_tools()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute tool calls and return results.

        Args:
            tool_calls: List of tool call dicts

        Returns:
            List of execution results
        """
        return self.executor.execute_tool_calls(tool_calls)

    def has_tools(self) -> bool:
        """
        Check if any tools are registered.

        Returns:
            True if tools are registered, False otherwise
        """
        return len(self.registry.get_tool_names()) > 0
