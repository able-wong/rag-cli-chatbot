"""
Unit tests for the tools system (loader, registry, executor, manager).
"""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from tools.executor import ToolExecutor
from tools.loader import ToolLoader
from tools.manager import ToolManager
from tools.registry import ToolRegistry


# Sample tool class for testing
class MockTools:
    """Mock Tools class for testing."""

    class Valves(BaseModel):
        api_key: str = "default_key"

    def __init__(self):
        self.valves = self.Valves()

    def sample_method(self, text: str) -> str:
        """
        Sample method for testing.

        :param text: Input text
        :return: Processed text
        """
        return f"Processed: {text}"

    def method_with_int(self, count: int) -> str:
        """
        Method with integer parameter.

        :param count: Number to process
        :return: Result string
        """
        return f"Count: {count}"

    def method_with_list(self, items: List[str]) -> str:
        """
        Method with list parameter.

        :param items: List of items
        :return: Joined items
        """
        return ", ".join(items)

    def _private_method(self) -> str:
        """Should not be registered."""
        return "private"


class TestToolLoader:
    """Test cases for ToolLoader."""

    def test_init_creates_directory(self):
        """Test that ToolLoader creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir) / "nonexistent"
            ToolLoader(str(tools_dir))
            assert tools_dir.exists()

    def test_discover_tools_finds_py_files(self):
        """Test that discover_tools finds .py files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir)
            (tools_dir / "tool1.py").write_text("# tool 1")
            (tools_dir / "tool2.py").write_text("# tool 2")
            (tools_dir / "__init__.py").write_text("# init")

            loader = ToolLoader(str(tools_dir))
            discovered = loader.discover_tools()

            assert "tool1" in discovered
            assert "tool2" in discovered
            assert "__init__" not in discovered

    def test_discover_tools_empty_directory(self):
        """Test discover_tools with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ToolLoader(str(tmpdir))
            discovered = loader.discover_tools()
            assert discovered == []

    def test_load_tool_success(self):
        """Test loading a valid tool module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir)
            tool_content = '''
from pydantic import BaseModel

class Tools:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def test_method(self) -> str:
        return "test"
'''
            (tools_dir / "testtool.py").write_text(tool_content)

            loader = ToolLoader(str(tools_dir))
            tools_instance = loader.load_tool("testtool")

            assert tools_instance is not None
            assert hasattr(tools_instance, "test_method")
            assert tools_instance.test_method() == "test"

    def test_load_tool_missing_tools_class(self):
        """Test loading module without Tools class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir)
            (tools_dir / "invalid.py").write_text("# no Tools class")

            loader = ToolLoader(str(tools_dir))
            with pytest.raises(AttributeError, match="does not have a 'Tools' class"):
                loader.load_tool("invalid")

    def test_load_tool_nonexistent_file(self):
        """Test loading non-existent tool file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ToolLoader(str(tmpdir))
            with pytest.raises(ImportError, match="not found"):
                loader.load_tool("nonexistent")

    def test_load_tool_with_valves(self):
        """Test loading tool with valve configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir)
            tool_content = '''
from pydantic import BaseModel

class Tools:
    class Valves(BaseModel):
        api_key: str = "default"

    def __init__(self):
        self.valves = self.Valves()

    def get_key(self) -> str:
        return self.valves.api_key
'''
            (tools_dir / "configtool.py").write_text(tool_content)

            loader = ToolLoader(str(tools_dir))
            tools_instance = loader.load_tool("configtool", valves={"api_key": "custom_key"})

            assert tools_instance.get_key() == "custom_key"

    def test_load_all_tools_success(self):
        """Test loading all tools successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir)
            tool_template = '''
from pydantic import BaseModel

class Tools:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def method(self) -> str:
        return "{name}"
'''
            (tools_dir / "tool1.py").write_text(tool_template.format(name="tool1"))
            (tools_dir / "tool2.py").write_text(tool_template.format(name="tool2"))

            loader = ToolLoader(str(tools_dir))
            loaded = loader.load_all_tools()

            assert len(loaded) == 2
            assert "tool1" in loaded
            assert "tool2" in loaded

    def test_load_all_tools_handles_failures(self):
        """Test that load_all_tools continues when some tools fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tools_dir = Path(tmpdir)
            # Valid tool
            valid_tool = '''
from pydantic import BaseModel

class Tools:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def method(self) -> str:
        return "valid"
'''
            (tools_dir / "valid.py").write_text(valid_tool)
            # Invalid tool (no Tools class)
            (tools_dir / "invalid.py").write_text("# no Tools class")

            loader = ToolLoader(str(tools_dir))
            loaded = loader.load_all_tools()

            # Should load valid tool and skip invalid
            assert len(loaded) == 1
            assert "valid" in loaded
            assert "invalid" not in loaded


class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def test_register_tool_instance(self):
        """Test registering tool instance and extracting methods."""
        registry = ToolRegistry()
        mock_tools = MockTools()

        registry.register_tool_instance("mock", mock_tools)

        # Should register public methods
        assert "sample_method" in registry.tools
        assert "method_with_int" in registry.tools
        assert "method_with_list" in registry.tools

        # Should skip private methods
        assert "_private_method" not in registry.tools

        # Should skip Valves
        assert "Valves" not in registry.tools

    def test_get_function_schema_basic(self):
        """Test extracting function schema from simple method."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        schema = registry.get_function_schema("sample_method")

        assert schema["name"] == "sample_method"
        assert "Sample method for testing" in schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert "text" in schema["parameters"]["properties"]
        assert schema["parameters"]["properties"]["text"]["type"] == "string"
        assert "text" in schema["parameters"]["required"]

    def test_get_function_schema_with_int_parameter(self):
        """Test schema extraction for method with integer parameter."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        schema = registry.get_function_schema("method_with_int")

        assert "count" in schema["parameters"]["properties"]
        assert schema["parameters"]["properties"]["count"]["type"] == "integer"

    def test_get_function_schema_with_list_parameter(self):
        """Test schema extraction for method with List[str] parameter."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        schema = registry.get_function_schema("method_with_list")

        assert "items" in schema["parameters"]["properties"]
        items_schema = schema["parameters"]["properties"]["items"]
        assert items_schema["type"] == "array"
        assert items_schema["items"]["type"] == "string"

    def test_get_function_schema_not_found(self):
        """Test getting schema for non-existent method."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_function_schema("nonexistent")

    def test_python_type_to_json_type_conversions(self):
        """Test type conversion for various Python types."""
        registry = ToolRegistry()

        assert registry._python_type_to_json_type(str) == {"type": "string"}
        assert registry._python_type_to_json_type(int) == {"type": "integer"}
        assert registry._python_type_to_json_type(float) == {"type": "number"}
        assert registry._python_type_to_json_type(bool) == {"type": "boolean"}
        assert registry._python_type_to_json_type(dict) == {"type": "object"}

        # List type
        list_type = registry._python_type_to_json_type(list)
        assert list_type["type"] == "array"

    def test_to_ollama_tools(self):
        """Test conversion to Ollama tool schema."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        ollama_tools = registry.to_ollama_tools()

        assert len(ollama_tools) > 0
        # Find sample_method in tools
        sample_tool = next(
            t for t in ollama_tools if t["function"]["name"] == "sample_method"
        )
        assert sample_tool["type"] == "function"
        assert "description" in sample_tool["function"]
        assert "parameters" in sample_tool["function"]

    def test_get_tool_method(self):
        """Test retrieving callable method by name."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        method = registry.get_tool_method("sample_method")

        assert method is not None
        assert callable(method)
        assert method("test") == "Processed: test"

    def test_get_tool_method_not_found(self):
        """Test getting non-existent tool method."""
        registry = ToolRegistry()

        method = registry.get_tool_method("nonexistent")
        assert method is None

    def test_get_tool_names(self):
        """Test getting list of registered tool names."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        names = registry.get_tool_names()

        assert "sample_method" in names
        assert "method_with_int" in names
        assert len(names) >= 3

    def test_extract_multiline_param_description(self):
        """Test extraction of multi-line parameter descriptions from docstrings."""
        registry = ToolRegistry()

        # Create a tool with multi-line parameter description
        class MultiLineTools:
            def complex_method(self, city: str) -> str:
                """
                Get timezone for a city.

                :param city: City name (e.g., 'Toronto', 'New York').
                             Can include province/state for disambiguation
                             (e.g., 'Toronto Ontario', 'Portland Oregon').
                             Accepts comma-separated format: "Toronto, ON".
                :return: Timezone identifier
                """
                return f"Timezone for {city}"

        multiline_tools = MultiLineTools()
        registry.register_tool_instance("multiline", multiline_tools)

        schema = registry.get_function_schema("complex_method")

        # Check that the full multi-line description is captured
        city_desc = schema["parameters"]["properties"]["city"]["description"]

        # All parts of the description should be present
        assert "City name" in city_desc
        assert "Toronto" in city_desc
        assert "New York" in city_desc
        assert "province/state" in city_desc
        assert "disambiguation" in city_desc
        assert "Portland Oregon" in city_desc
        assert "comma-separated" in city_desc
        assert "Toronto, ON" in city_desc

        # Should not contain the :return line
        assert ":return" not in city_desc
        assert "Timezone identifier" not in city_desc


class TestToolRegistryGemini:
    """Test Gemini-specific tool conversion."""

    def test_to_gemini_tools(self):
        """Test conversion to Gemini FunctionDeclaration format."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        # Mock the FunctionDeclaration import inside the method
        with patch("google.generativeai.types.FunctionDeclaration") as mock_func_decl:
            mock_func_decl.return_value = MagicMock()

            registry.to_gemini_tools()

            # Should have called FunctionDeclaration for each method
            assert mock_func_decl.call_count >= 3  # At least 3 public methods

            # Verify one of the calls
            calls = mock_func_decl.call_args_list
            sample_call = next(
                c for c in calls if c.kwargs.get("name") == "sample_method"
            )
            assert "description" in sample_call.kwargs
            assert "parameters" in sample_call.kwargs

    def test_to_gemini_tools_import_error(self):
        """Test handling of missing google-generativeai package."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        # Mock the import to fail
        import sys
        original_modules = sys.modules.copy()

        # Remove google.generativeai from sys.modules if it exists
        if "google.generativeai.types" in sys.modules:
            del sys.modules["google.generativeai.types"]
        if "google.generativeai" in sys.modules:
            del sys.modules["google.generativeai"]
        if "google" in sys.modules:
            del sys.modules["google"]

        # Add fake module that raises ImportError
        sys.modules["google.generativeai.types"] = None

        with pytest.raises(ImportError, match="google-generativeai not installed"):
            registry.to_gemini_tools()

        # Restore modules
        sys.modules.clear()
        sys.modules.update(original_modules)


class TestToolExecutor:
    """Test cases for ToolExecutor."""

    def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        executor = ToolExecutor(registry)
        result = executor.execute_tool("sample_method", {"text": "hello"})

        assert result["success"] is True
        assert result["result"] == "Processed: hello"
        assert result["error"] == ""

    def test_execute_tool_not_found(self):
        """Test execution of non-existent tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)

        result = executor.execute_tool("nonexistent", {})

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_execute_tool_invalid_args(self):
        """Test execution with invalid arguments."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        executor = ToolExecutor(registry)
        # Missing required 'text' argument
        result = executor.execute_tool("sample_method", {})

        assert result["success"] is False
        assert "Invalid arguments" in result["error"]

    def test_execute_tool_exception(self):
        """Test handling of tool that raises exception."""
        registry = ToolRegistry()

        # Create a tool that raises an exception
        class ErrorTools:
            def error_method(self) -> str:
                raise ValueError("Test error")

        error_tools = ErrorTools()
        registry.register_tool_instance("error", error_tools)

        executor = ToolExecutor(registry)
        result = executor.execute_tool("error_method", {})

        assert result["success"] is False
        assert "Test error" in result["error"]

    def test_execute_tool_calls_multiple(self):
        """Test executing multiple tool calls."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        executor = ToolExecutor(registry)
        tool_calls = [
            {"function": {"name": "sample_method", "arguments": {"text": "test1"}}},
            {"function": {"name": "method_with_int", "arguments": {"count": 42}}},
        ]

        results = executor.execute_tool_calls(tool_calls)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert "test1" in results[0]["result"]
        assert results[1]["success"] is True
        assert "42" in results[1]["result"]


class TestToolManager:
    """Test cases for ToolManager."""

    def test_initialization(self):
        """Test ToolManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"directory": tmpdir, "enabled": True}
            manager = ToolManager(config, "ollama")

            assert manager.llm_provider == "ollama"
            assert not manager._initialized

    def test_initialize_loads_tools(self):
        """Test that initialize() loads tools successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple tool
            tool_content = '''
from pydantic import BaseModel

class Tools:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    def test_method(self) -> str:
        """Test method."""
        return "test"
'''
            Path(tmpdir) / "testtool.py"
            Path(tmpdir).joinpath("testtool.py").write_text(tool_content)

            config = {"directory": tmpdir, "enabled": True}
            manager = ToolManager(config, "ollama")
            manager.initialize()

            assert manager._initialized
            assert manager.has_tools()
            assert "test_method" in manager.registry.get_tool_names()

    def test_get_tools_for_llm_ollama(self):
        """Test getting tools in Ollama format."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"directory": tmpdir}
            manager = ToolManager(config, "ollama")
            manager.registry = registry
            manager._initialized = True

            tools = manager.get_tools_for_llm()

            assert len(tools) > 0
            assert tools[0]["type"] == "function"
            assert "name" in tools[0]["function"]

    def test_get_tools_for_llm_not_initialized(self):
        """Test that get_tools_for_llm raises error if not initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"directory": tmpdir}
            manager = ToolManager(config, "ollama")

            with pytest.raises(RuntimeError, match="not initialized"):
                manager.get_tools_for_llm()

    def test_execute_tool_calls(self):
        """Test executing tool calls through manager."""
        registry = ToolRegistry()
        mock_tools = MockTools()
        registry.register_tool_instance("mock", mock_tools)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"directory": tmpdir}
            manager = ToolManager(config, "ollama")
            manager.registry = registry
            manager.executor = ToolExecutor(registry)

            tool_calls = [
                {"function": {"name": "sample_method", "arguments": {"text": "hello"}}}
            ]
            results = manager.execute_tool_calls(tool_calls)

            assert len(results) == 1
            assert results[0]["success"] is True

    def test_has_tools(self):
        """Test has_tools() method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"directory": tmpdir}
            manager = ToolManager(config, "ollama")

            assert not manager.has_tools()

            # Add a tool
            registry = ToolRegistry()
            mock_tools = MockTools()
            registry.register_tool_instance("mock", mock_tools)
            manager.registry = registry

            assert manager.has_tools()
