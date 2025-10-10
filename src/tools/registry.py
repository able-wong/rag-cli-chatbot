"""
Tool registry for converting OpenWebUI tools to LLM-compatible schemas.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Converts OpenWebUI tool methods to LLM-compatible function schemas."""

    def __init__(self):
        """Initialize empty tool registry."""
        self.tools = {}  # {tool_name: {"instance": obj, "method": callable, "module": str}}

    def register_tool_instance(self, module_name: str, tools_instance: object):
        """
        Extract public methods from Tools instance and register them.

        Skips:
        - __init__ and other dunder methods
        - Private methods (starting with _)
        - Valves and UserValves attributes
        - Non-callable attributes

        Args:
            module_name: Name of the module (for namespacing)
            tools_instance: Instance of Tools class
        """
        methods_registered = 0

        for attr_name in dir(tools_instance):
            # Skip private, dunder, and special attributes
            if attr_name.startswith("_"):
                continue
            if attr_name in ["Valves", "UserValves", "valves", "user_valves"]:
                continue

            attr = getattr(tools_instance, attr_name)

            # Only register callable methods
            if not callable(attr):
                continue

            # Use method name as tool name (can be prefixed with module for uniqueness)
            tool_name = attr_name
            self.tools[tool_name] = {
                "instance": tools_instance,
                "method": attr,
                "module": module_name,
            }
            methods_registered += 1
            logger.debug(f"Registered tool: {tool_name} from {module_name}")

        logger.info(
            f"Registered {methods_registered} methods from {module_name}"
        )

    def get_function_schema(self, method_name: str) -> Dict[str, Any]:
        """
        Extract function schema from method using inspect module.

        Returns OpenAPI-compatible schema with:
        - name: method name
        - description: from docstring
        - parameters: {type, properties, required} from type hints

        Args:
            method_name: Name of the registered method

        Returns:
            Dict with function schema

        Raises:
            KeyError: If method not found in registry
        """
        if method_name not in self.tools:
            raise KeyError(f"Tool method not found: {method_name}")

        tool_info = self.tools[method_name]
        method = tool_info["method"]

        # Extract description from docstring
        docstring = inspect.getdoc(method) or f"Call {method_name}"
        description = docstring.split("\n")[0]  # First line only

        # Extract parameters from signature
        sig = inspect.signature(method)
        type_hints = get_type_hints(method) if hasattr(method, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip self parameter
            if param_name == "self":
                continue

            # Skip special OpenWebUI parameters
            if param_name in ["__event_emitter__", "__user__", "__metadata__"]:
                continue

            # Get type hint
            param_type = type_hints.get(param_name, Any)
            json_type = self._python_type_to_json_type(param_type)

            # Build parameter schema
            param_schema = {"type": json_type["type"]}
            if "items" in json_type:
                param_schema["items"] = json_type["items"]

            # Extract parameter description from docstring
            param_desc = self._extract_param_description(docstring, param_name)
            if param_desc:
                param_schema["description"] = param_desc

            properties[param_name] = param_schema

            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": method_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _python_type_to_json_type(self, python_type: Any) -> Dict[str, Any]:
        """
        Convert Python type hint to JSON Schema type.

        Args:
            python_type: Python type from type hints

        Returns:
            Dict with 'type' and optionally 'items' for arrays
        """
        # Handle string representation of types
        if isinstance(python_type, str):
            python_type = python_type.lower()
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            return {"type": type_map.get(python_type, "string")}

        # Get the origin type (e.g., list from List[str])
        origin = getattr(python_type, "__origin__", None)

        # Handle basic types
        if python_type is str:
            return {"type": "string"}
        elif python_type is int:
            return {"type": "integer"}
        elif python_type is float:
            return {"type": "number"}
        elif python_type is bool:
            return {"type": "boolean"}
        elif python_type is dict or origin is dict:
            return {"type": "object"}
        elif python_type is list or origin is list:
            # Handle List[ItemType]
            args = getattr(python_type, "__args__", None)
            if args:
                item_type = self._python_type_to_json_type(args[0])
                return {"type": "array", "items": item_type}
            return {"type": "array", "items": {"type": "string"}}

        # Handle Optional (Union with None)
        if origin is type(Optional[str]).__origin__:  # Union type
            args = getattr(python_type, "__args__", ())
            # Get first non-None type
            for arg in args:
                if arg is not type(None):
                    return self._python_type_to_json_type(arg)

        # Default to string for unknown types
        return {"type": "string"}

    def _extract_param_description(self, docstring: str, param_name: str) -> str:
        """
        Extract parameter description from docstring.

        Handles multi-line parameter descriptions by collecting all indented
        continuation lines after the :param line.

        Looks for patterns like:
        :param param_name: description
                          continuation line
                          another line
        :return: description

        Args:
            docstring: Method docstring
            param_name: Parameter name to find

        Returns:
            Parameter description or empty string
        """
        if not docstring:
            return ""

        lines = docstring.split("\n")
        description_parts = []
        in_param = False
        param_indent = None

        for line in lines:
            stripped = line.strip()

            # Check if this is the start of our parameter
            if stripped.startswith(f":param {param_name}:"):
                in_param = True
                # Extract the description after the colon
                desc = stripped.split(":", 2)[2].strip()
                if desc:
                    description_parts.append(desc)
                # Calculate indentation level for continuation lines
                param_indent = len(line) - len(line.lstrip())
                continue
            elif stripped.startswith(f":param {param_name} "):
                in_param = True
                desc = stripped.split(param_name, 1)[1].strip()
                if desc:
                    description_parts.append(desc)
                param_indent = len(line) - len(line.lstrip())
                continue

            # If we're in a param and see another :param or :return, stop
            if in_param and (stripped.startswith(":param ") or stripped.startswith(":return")):
                break

            # If we're in a param, collect continuation lines (must be indented)
            if in_param:
                # Check if line is empty or indented (continuation)
                if not stripped:
                    # Empty line might separate parts, but keep collecting
                    continue
                elif param_indent is not None and len(line) - len(line.lstrip()) > param_indent:
                    # This is an indented continuation line
                    description_parts.append(stripped)
                elif stripped and not stripped.startswith(":"):
                    # Non-indented, non-empty, non-directive line - likely end of param
                    break

        return " ".join(description_parts)

    def to_gemini_tools(self) -> List[Any]:
        """
        Convert all registered tools to Gemini FunctionDeclaration objects.

        Returns:
            List of google.generativeai.types.FunctionDeclaration

        Raises:
            ImportError: If google.generativeai not installed
        """
        try:
            from google.generativeai.types import FunctionDeclaration
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )

        function_declarations = []
        for method_name in self.tools:
            schema = self.get_function_schema(method_name)
            func_decl = FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
            )
            function_declarations.append(func_decl)

        return function_declarations

    def to_ollama_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all registered tools to Ollama tool schema.

        Returns:
            List of {"type": "function", "function": {...}}
        """
        ollama_tools = []
        for method_name in self.tools:
            schema = self.get_function_schema(method_name)
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
            ollama_tools.append(ollama_tool)

        return ollama_tools

    def get_tool_method(self, tool_name: str) -> Optional[Callable]:
        """
        Get callable method by name for execution.

        Args:
            tool_name: Name of the tool method

        Returns:
            Callable method or None if not found
        """
        tool_info = self.tools.get(tool_name)
        return tool_info["method"] if tool_info else None

    def get_tool_names(self) -> List[str]:
        """
        Get list of all registered tool names.

        Returns:
            List of tool method names
        """
        return list(self.tools.keys())
