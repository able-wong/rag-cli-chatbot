"""
Tool loader for automatic discovery and import of OpenWebUI-format tools.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class ToolLoader:
    """Discovers and loads OpenWebUI-format tools from a directory."""

    def __init__(self, tools_directory: str):
        """
        Initialize loader with tools directory path.

        Args:
            tools_directory: Path to directory containing tool modules
        """
        self.tools_directory = Path(tools_directory)
        if not self.tools_directory.exists():
            logger.warning(f"Tools directory does not exist: {tools_directory}")
            self.tools_directory.mkdir(parents=True, exist_ok=True)

    def discover_tools(self) -> List[str]:
        """
        Find all .py files in tools directory (excluding __pycache__, __init__.py).

        Returns:
            List of module names (without .py extension)
        """
        if not self.tools_directory.exists():
            return []

        tool_files = []
        for file_path in self.tools_directory.glob("*.py"):
            # Skip __init__.py and private files
            if file_path.name.startswith("_"):
                continue
            tool_files.append(file_path.stem)

        logger.debug(f"Discovered {len(tool_files)} tool files: {tool_files}")
        return tool_files

    def load_tool(self, module_name: str, valves: Dict = None) -> object:
        """
        Import module and return Tools class instance.

        Args:
            module_name: Name of the module to load (without .py)
            valves: Optional configuration dict for tool Valves

        Returns:
            Instance of Tools class from the module

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If module doesn't have Tools class or invalid structure
        """
        module_path = self.tools_directory / f"{module_name}.py"
        if not module_path.exists():
            raise ImportError(f"Tool module not found: {module_path}")

        # Import module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec for: {module_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Validate module has Tools class
        if not hasattr(module, "Tools"):
            raise AttributeError(f"Module {module_name} does not have a 'Tools' class")

        tools_class = getattr(module, "Tools")

        # Instantiate Tools class
        try:
            tools_instance = tools_class()
            # Configure valves if provided
            if valves and hasattr(tools_instance, "valves"):
                for key, value in valves.items():
                    setattr(tools_instance.valves, key, value)
            return tools_instance
        except Exception as e:
            raise AttributeError(
                f"Failed to instantiate Tools class from {module_name}: {e}"
            )

    def load_all_tools(self, valves_config: Dict[str, Dict] = None) -> Dict[str, object]:
        """
        Load all discovered tools.

        Args:
            valves_config: Optional dict mapping module names to valve configurations

        Returns:
            Dict mapping module_name to Tools instance
            Logs warnings for tools that fail to load, continues with others.
        """
        valves_config = valves_config or {}
        discovered = self.discover_tools()
        loaded_tools = {}

        for module_name in discovered:
            try:
                valves = valves_config.get(module_name, {})
                tools_instance = self.load_tool(module_name, valves)
                loaded_tools[module_name] = tools_instance
                logger.info(f"Successfully loaded tool: {module_name}")
            except Exception as e:
                logger.warning(f"Failed to load tool '{module_name}': {e}")
                continue

        return loaded_tools
