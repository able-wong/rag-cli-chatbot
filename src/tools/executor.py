"""
Tool executor for running tool calls with error handling.
"""

import asyncio
import inspect
import logging
from typing import Any, Dict, List

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tool calls with error handling."""

    def __init__(self, registry: ToolRegistry):
        """
        Initialize executor with tool registry.

        Args:
            registry: ToolRegistry instance with registered tools
        """
        self.registry = registry

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute single tool call.

        Args:
            tool_name: Name of method to call
            arguments: Dict of parameter name → value

        Returns:
            {"success": bool, "result": str, "error": str}

        Handles:
        - Tool not found
        - Argument validation
        - Execution exceptions
        - Async tool methods (run with asyncio.run())
        - Special parameters (__event_emitter__, etc.) - inject None
        """
        try:
            # Get tool method
            method = self.registry.get_tool_method(tool_name)
            if method is None:
                return {
                    "success": False,
                    "result": "",
                    "error": f"Tool not found: {tool_name}",
                }

            # Filter out special OpenWebUI parameters and inject None
            filtered_args = {}
            sig = inspect.signature(method)

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Handle special OpenWebUI parameters
                if param_name in ["__event_emitter__", "__user__", "__metadata__"]:
                    filtered_args[param_name] = None
                elif param_name in arguments:
                    filtered_args[param_name] = arguments[param_name]

            # Execute method (handle both sync and async)
            if inspect.iscoroutinefunction(method):
                result = asyncio.run(method(**filtered_args))
            else:
                result = method(**filtered_args)

            return {"success": True, "result": str(result), "error": ""}

        except TypeError as e:
            logger.error(f"Invalid arguments for {tool_name}: {e}")
            return {
                "success": False,
                "result": "",
                "error": f"Invalid arguments: {e}",
            }
        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            return {"success": False, "result": "", "error": str(e)}

    def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute multiple tool calls (sequential for now).

        Args:
            tool_calls: List of tool call dicts with format:
                       [{"function": {"name": str, "arguments": dict}}]

        Returns:
            List of execution results
        """
        results = []
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name", "")
            arguments = function_info.get("arguments", {})

            result = self.execute_tool(tool_name, arguments)
            results.append(result)

        return results
