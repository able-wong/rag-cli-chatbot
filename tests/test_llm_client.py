import unittest
from unittest.mock import Mock, patch
import json
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import LLMClient


class TestLLMClient(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'provider': 'ollama',
            'model': 'test-model',
            'base_url': 'http://localhost:11434',
            'timeout': 120,
            'temperature': 0.7,
            'max_tokens': 1024
        }
        
    def test_strip_thinking_blocks_single_block(self):
        """Test removal of single thinking block."""
        client = LLMClient(self.test_config)
        
        text = """<think>
This is my thinking process about the problem.
I need to analyze the user's request carefully.
</think>

{
  "response": "actual content"
}"""
        
        result = client._strip_thinking_blocks(text)
        expected = '{\n  "response": "actual content"\n}'
        
        self.assertEqual(result, expected)
        self.assertNotIn('<think>', result)
        self.assertNotIn('thinking process', result)

    def test_strip_thinking_blocks_multiple_blocks(self):
        """Test removal of multiple thinking blocks."""
        client = LLMClient(self.test_config)
        
        text = """<think>First thought</think>
Some content here
<think>Second thought about something else</think>
Final content"""
        
        result = client._strip_thinking_blocks(text)
        expected = 'Some content here\nFinal content'
        
        self.assertEqual(result, expected)
        self.assertNotIn('<think>', result)
        self.assertNotIn('First thought', result)
        self.assertNotIn('Second thought', result)

    def test_strip_thinking_blocks_no_blocks(self):
        """Test that text without thinking blocks is unchanged."""
        client = LLMClient(self.test_config)
        
        text = '{"key": "value", "number": 42}'
        result = client._strip_thinking_blocks(text)
        
        self.assertEqual(result, text)

    def test_strip_thinking_blocks_case_insensitive(self):
        """Test case insensitive matching of thinking blocks."""
        client = LLMClient(self.test_config)
        
        text = """<THINK>Uppercase thinking</THINK>
<Think>Mixed case thinking</Think>
<think>Lowercase thinking</think>
Final result"""
        
        result = client._strip_thinking_blocks(text)
        expected = 'Final result'
        
        self.assertEqual(result, expected)
        self.assertNotIn('thinking', result.lower())

    def test_strip_thinking_blocks_whitespace_cleanup(self):
        """Test proper whitespace cleanup after thinking block removal."""
        client = LLMClient(self.test_config)
        
        text = """<think>Some thinking</think>


Multiple empty lines here


Final content"""
        
        result = client._strip_thinking_blocks(text)
        expected = 'Multiple empty lines here\nFinal content'
        
        self.assertEqual(result, expected)
        # Should not have multiple consecutive newlines
        self.assertNotIn('\n\n\n', result)

    def test_extract_json_from_response_thinking_plus_markdown(self):
        """Test extraction with both thinking blocks and markdown wrappers."""
        client = LLMClient(self.test_config)
        
        text = """<think>
I need to generate a JSON response with the following structure...
</think>

```json
{
  "key": "value",
  "array": [1, 2, 3]
}
```"""
        
        result = client._extract_json_from_response(text)
        expected = '{\n  "key": "value",\n  "array": [1, 2, 3]\n}'
        
        self.assertEqual(result, expected)

    def test_extract_json_from_response_thinking_only(self):
        """Test extraction with just thinking blocks, no markdown."""
        client = LLMClient(self.test_config)
        
        text = """<think>Let me think about this...</think>
{"result": "success", "data": {"items": [1, 2, 3]}}"""
        
        result = client._extract_json_from_response(text)
        expected = '{"result": "success", "data": {"items": [1, 2, 3]}}'
        
        self.assertEqual(result, expected)

    def test_extract_json_from_response_markdown_only(self):
        """Test extraction with just markdown wrappers, no thinking."""
        client = LLMClient(self.test_config)
        
        text = """```json
{
  "status": "complete",
  "message": "Operation successful"
}
```"""
        
        result = client._extract_json_from_response(text)
        expected = '{\n  "status": "complete",\n  "message": "Operation successful"\n}'
        
        self.assertEqual(result, expected)

    def test_extract_json_from_response_plain_json(self):
        """Test extraction with plain JSON (no processing needed)."""
        client = LLMClient(self.test_config)
        
        text = '{"plain": "json", "no_processing": true}'
        result = client._extract_json_from_response(text)
        
        self.assertEqual(result, text)

    @patch('llm_client.LLMClient.get_llm_response')
    def test_get_json_response_with_thinking_blocks(self, mock_get_llm_response):
        """Test end-to-end JSON response with thinking blocks."""
        client = LLMClient(self.test_config)
        
        # Mock LLM response with thinking blocks
        mock_response = """<think>
The user wants a JSON response. I need to structure this properly.
Let me think about the required fields...
</think>

{
  "search_rag": true,
  "embedding_texts": {
    "rewrite": "test query",
    "hyde": ["response 1", "response 2", "response 3"]
  },
  "llm_query": "test instruction"
}"""
        
        mock_get_llm_response.return_value = mock_response
        
        messages = [{"role": "user", "content": "test query"}]
        result = client.get_json_response(messages)
        
        expected = {
            "search_rag": True,
            "embedding_texts": {
                "rewrite": "test query",
                "hyde": ["response 1", "response 2", "response 3"]
            },
            "llm_query": "test instruction"
        }
        
        self.assertEqual(result, expected)
        mock_get_llm_response.assert_called_once_with(messages, None, None)

    @patch('llm_client.LLMClient.get_llm_response')
    def test_get_json_response_with_thinking_blocks_and_markdown(self, mock_get_llm_response):
        """Test JSON response with both thinking blocks and markdown wrappers."""
        client = LLMClient(self.test_config)
        
        mock_response = """<think>Complex reasoning here...</think>

```json
{
  "processed": true,
  "data": {"test": "value"}
}
```"""
        
        mock_get_llm_response.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        result = client.get_json_response(messages)
        
        expected = {
            "processed": True,
            "data": {"test": "value"}
        }
        
        self.assertEqual(result, expected)

    @patch('llm_client.LLMClient.get_llm_response')
    def test_get_json_response_invalid_json_with_debug_info(self, mock_get_llm_response):
        """Test that invalid JSON includes raw response in error for debugging."""
        client = LLMClient(self.test_config)
        
        mock_response = """<think>Thinking...</think>
{invalid json: missing quotes}"""
        
        mock_get_llm_response.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        
        with self.assertRaises(ValueError) as context:
            client.get_json_response(messages)
        
        # Check that the error includes debugging information
        error = context.exception
        self.assertTrue(hasattr(error, 'raw_response'))
        self.assertEqual(error.raw_response, mock_response)
        self.assertTrue(hasattr(error, 'cleaned_response'))
        self.assertEqual(error.cleaned_response, '{invalid json: missing quotes}')


if __name__ == '__main__':
    unittest.main()