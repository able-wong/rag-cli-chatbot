"""
Unit tests for LLMClient JSON response parsing functionality.
Tests the get_json_response() method and markdown wrapper handling.
"""

import sys
import os
from unittest.mock import patch
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import LLMClient


class TestLLMClientJSONParsing:
    """Test LLMClient JSON response parsing and markdown wrapper handling."""
    
    def create_mock_llm_client(self, provider: str = "ollama") -> LLMClient:
        """Create a mock LLMClient for testing."""
        config = {
            'provider': provider,
            'model': 'test-model',
            'base_url': 'http://localhost:11434' if provider == 'ollama' else None,
            'gemini': {'api_key': 'test-key', 'model': 'test-model'} if provider == 'gemini' else None
        }
        
        with patch.object(LLMClient, '_initialize_client'):
            return LLMClient(config)
    
    def test_get_json_response_valid_json_no_wrapper(self):
        """Test parsing valid JSON without markdown wrapper."""
        client = self.create_mock_llm_client()
        
        json_response = '{"search_rag": true, "query": "test", "filters": {}}'
        
        with patch.object(client, 'get_llm_response', return_value=json_response):
            result = client.get_json_response([{"role": "user", "content": "test"}])
            
            assert result == {"search_rag": True, "query": "test", "filters": {}}
    
    def test_get_json_response_with_json_markdown_wrapper(self):
        """Test parsing JSON with ```json``` markdown wrapper."""
        client = self.create_mock_llm_client()
        
        json_response = '''```json
{
    "search_rag": true,
    "embedding_source_text": "machine learning",
    "llm_query": "Based on the provided context...",
    "filters": {"author": "Smith"}
}
```'''
        
        with patch.object(client, 'get_llm_response', return_value=json_response):
            result = client.get_json_response([{"role": "user", "content": "test"}])
            
            expected = {
                "search_rag": True,
                "embedding_source_text": "machine learning", 
                "llm_query": "Based on the provided context...",
                "filters": {"author": "Smith"}
            }
            assert result == expected
    
    def test_get_json_response_with_plain_markdown_wrapper(self):
        """Test parsing JSON with plain ``` markdown wrapper (no json specifier)."""
        client = self.create_mock_llm_client()
        
        json_response = '''```
{"search_rag": false, "embedding_source_text": "", "llm_query": "Tell me about Python"}
```'''
        
        with patch.object(client, 'get_llm_response', return_value=json_response):
            result = client.get_json_response([{"role": "user", "content": "test"}])
            
            expected = {
                "search_rag": False,
                "embedding_source_text": "",
                "llm_query": "Tell me about Python"
            }
            assert result == expected
    
    def test_get_json_response_with_quad_backticks(self):
        """Test parsing JSON with ````json```` wrapper (rare but possible)."""
        client = self.create_mock_llm_client()
        
        json_response = '''````json
{"test": "value", "number": 42}
````'''
        
        with patch.object(client, 'get_llm_response', return_value=json_response):
            result = client.get_json_response([{"role": "user", "content": "test"}])
            
            assert result == {"test": "value", "number": 42}
    
    def test_get_json_response_single_line_wrapper(self):
        """Test parsing JSON with single-line markdown wrapper."""
        client = self.create_mock_llm_client()
        
        json_response = '```json{"status": "ok", "data": [1, 2, 3]}```'
        
        with patch.object(client, 'get_llm_response', return_value=json_response):
            result = client.get_json_response([{"role": "user", "content": "test"}])
            
            assert result == {"status": "ok", "data": [1, 2, 3]}
    
    def test_get_json_response_malformed_json(self):
        """Test error handling for malformed JSON."""
        client = self.create_mock_llm_client()
        
        # Missing closing brace
        json_response = '{"search_rag": true, "query": "test"'
        
        with patch.object(client, 'get_llm_response', return_value=json_response):
            with pytest.raises(ValueError, match="Invalid JSON response from LLM"):
                client.get_json_response([{"role": "user", "content": "test"}])
    
    def test_get_json_response_empty_response(self):
        """Test error handling for empty LLM response.""" 
        client = self.create_mock_llm_client()
        
        with patch.object(client, 'get_llm_response', return_value=""):
            with pytest.raises(ValueError, match="LLM returned empty response"):
                client.get_json_response([{"role": "user", "content": "test"}])
    
    def test_get_json_response_whitespace_only_response(self):
        """Test error handling for whitespace-only response."""
        client = self.create_mock_llm_client()
        
        with patch.object(client, 'get_llm_response', return_value="   \n\t  "):
            with pytest.raises(ValueError, match="LLM returned empty response"):
                client.get_json_response([{"role": "user", "content": "test"}])
    
    def test_get_json_response_llm_error_propagation(self):
        """Test that LLM errors are properly propagated."""
        client = self.create_mock_llm_client()
        
        with patch.object(client, 'get_llm_response', side_effect=Exception("LLM connection failed")):
            with pytest.raises(Exception, match="LLM connection failed"):
                client.get_json_response([{"role": "user", "content": "test"}])
    
    def test_strip_markdown_json_wrapper_no_wrapper(self):
        """Test _strip_markdown_json_wrapper with plain JSON."""
        client = self.create_mock_llm_client()
        
        plain_json = '{"test": "value"}'
        result = client._strip_markdown_json_wrapper(plain_json)
        
        assert result == '{"test": "value"}'
    
    def test_strip_markdown_json_wrapper_with_json_specifier(self):
        """Test _strip_markdown_json_wrapper with ```json``` wrapper."""
        client = self.create_mock_llm_client()
        
        wrapped_json = '''```json
{"search_rag": true, "query": "test"}
```'''
        
        result = client._strip_markdown_json_wrapper(wrapped_json)
        assert result == '{"search_rag": true, "query": "test"}'
    
    def test_strip_markdown_json_wrapper_without_json_specifier(self):
        """Test _strip_markdown_json_wrapper with plain ``` wrapper."""
        client = self.create_mock_llm_client()
        
        wrapped_json = '''```
{"status": "ok"}
```'''
        
        result = client._strip_markdown_json_wrapper(wrapped_json)
        assert result == '{"status": "ok"}'
    
    def test_strip_markdown_json_wrapper_with_extra_whitespace(self):
        """Test _strip_markdown_json_wrapper handles extra whitespace."""
        client = self.create_mock_llm_client()
        
        wrapped_json = '''   ```json   
   {"test": "value"}   
   ```   '''
        
        result = client._strip_markdown_json_wrapper(wrapped_json)
        assert result == '{"test": "value"}'
    
    def test_get_json_response_with_parameters(self):
        """Test get_json_response passes parameters correctly to get_llm_response."""
        client = self.create_mock_llm_client()
        
        json_response = '{"result": "success"}'
        messages = [{"role": "user", "content": "test"}]
        
        with patch.object(client, 'get_llm_response', return_value=json_response) as mock_get_response:
            result = client.get_json_response(messages, temperature=0.5, max_tokens=100)
            
            # Verify parameters were passed through
            mock_get_response.assert_called_once_with(messages, 0.5, 100)
            assert result == {"result": "success"}
    
    def test_get_json_response_complex_nested_json(self):
        """Test parsing complex nested JSON structure."""
        client = self.create_mock_llm_client()
        
        complex_json = '''```json
{
    "search_rag": true,
    "embedding_source_text": "machine learning algorithms",
    "llm_query": "Based on the provided context, explain machine learning algorithms.",
    "filters": {
        "author": "Dr. Smith",
        "tags": ["machine learning", "algorithms", "AI"],
        "publication_date": "2023",
        "metadata": {
            "confidence": 0.95,
            "source_type": "academic"
        }
    }
}
```'''
        
        with patch.object(client, 'get_llm_response', return_value=complex_json):
            result = client.get_json_response([{"role": "user", "content": "test"}])
            
            expected = {
                "search_rag": True,
                "embedding_source_text": "machine learning algorithms",
                "llm_query": "Based on the provided context, explain machine learning algorithms.",
                "filters": {
                    "author": "Dr. Smith",
                    "tags": ["machine learning", "algorithms", "AI"],
                    "publication_date": "2023",
                    "metadata": {
                        "confidence": 0.95,
                        "source_type": "academic"
                    }
                }
            }
            assert result == expected