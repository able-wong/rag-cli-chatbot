"""
Unit tests for QueryRewriter class.
Tests query transformation logic with mocked LLM responses.
"""

import sys
import os
import json
from unittest.mock import Mock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter

def create_mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_client = Mock()
    return mock_client

def test_query_rewriter_initialization():
    """Test QueryRewriter initialization."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'temperature': 0.1,
        'max_tokens': 512
    }
    
    rewriter = QueryRewriter(mock_llm, config)
    
    assert rewriter.llm_client == mock_llm
    assert rewriter.trigger_phrase == '@knowledgebase'
    assert rewriter.temperature == 0.1
    assert rewriter.max_tokens == 512

def test_trigger_detection_in_transformation():
    """Test trigger phrase detection in query transformation."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock valid JSON response with trigger detected
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "Python programming language",
        "llm_query": "What is Python programming language?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("@knowledgebase What is Python?")
    
    assert result['search_rag'] is True
    assert 'Python' in result['embedding_source_text']
    assert result['llm_query'] == "What is Python programming language?"
    mock_llm.get_llm_response.assert_called_once()

def test_no_trigger_detection():
    """Test query without trigger phrase."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock valid JSON response without trigger
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "weather today",
        "llm_query": "What is the weather today?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("What is the weather today?")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == "weather today"
    assert result['llm_query'] == "What is the weather today?"

def test_json_parsing_markdown_wrapped():
    """Test parsing JSON wrapped in markdown code blocks."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response with JSON in markdown code block
    mock_response = '''```json
{
    "search_rag": true,
    "embedding_source_text": "machine learning algorithms",
    "llm_query": "Explain machine learning algorithms"
}
```'''
    mock_llm.get_llm_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase Explain ML algorithms")
    
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "machine learning algorithms"
    assert result['llm_query'] == "Explain machine learning algorithms"

def test_json_parsing_with_extra_text():
    """Test parsing JSON when response includes extra text."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response with JSON embedded in text
    mock_response = '''Here is the analysis:
{"search_rag": false, "embedding_source_text": "cats behavior", "llm_query": "Why do cats purr?"}
Hope this helps!'''
    mock_llm.get_llm_response.return_value = mock_response
    
    result = rewriter.transform_query("Why do cats purr?")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == "cats behavior"
    assert result['llm_query'] == "Why do cats purr?"

def test_malformed_json_fallback():
    """Test fallback behavior when LLM returns malformed JSON."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock malformed JSON response
    mock_llm.get_llm_response.return_value = "This is not valid JSON at all!"
    
    result = rewriter.transform_query("@knowledgebase What is AI?")
    
    # Should fall back to simple logic
    assert result['search_rag'] is True  # Trigger detected
    assert result['embedding_source_text'] == "What is AI?"  # Trigger removed
    assert result['llm_query'] == "@knowledgebase What is AI?"  # Original query

def test_missing_fields_fallback():
    """Test fallback when JSON is missing required fields."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock incomplete JSON response
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "incomplete response"
        # Missing llm_query field
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("@knowledgebase Test query")
    
    # Should use fallback
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "Test query"
    assert result['llm_query'] == "@knowledgebase Test query"

def test_trigger_detection_mismatch_correction():
    """Test correction when LLM incorrectly detects trigger phrase."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response where LLM incorrectly says no trigger when there is one
    mock_response = {
        "search_rag": False,  # LLM says no trigger
        "embedding_source_text": "artificial intelligence",
        "llm_query": "What is artificial intelligence?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("@knowledgebase What is AI?")  # Actually has trigger
    
    # Should correct the search_rag field
    assert result['search_rag'] is True  # Corrected to True
    assert result['embedding_source_text'] == "artificial intelligence"
    assert result['llm_query'] == "What is artificial intelligence?"

def test_case_insensitive_trigger_detection():
    """Test case-insensitive trigger phrase detection."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "database design",
        "llm_query": "How to design databases?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    # Test with different cases
    test_queries = [
        "@KNOWLEDGEBASE How to design databases?",
        "@KnowledgeBase How to design databases?",
        "Tell me @knowledgebase about databases"
    ]
    
    for query in test_queries:
        result = rewriter.transform_query(query)
        assert result['search_rag'] is True

def test_empty_or_invalid_string_fields():
    """Test handling of empty or invalid string fields."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response with empty embedding_source_text
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "",  # Empty string
        "llm_query": "Valid query"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("@knowledgebase Test")
    
    # Should use fallback due to empty embedding_source_text
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "Test"  # Fallback value
    assert result['llm_query'] == "@knowledgebase Test"

def test_non_boolean_search_rag_conversion():
    """Test conversion of non-boolean search_rag values."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test string boolean conversion
    test_cases = [
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False)
    ]
    
    for string_val, expected_bool in test_cases:
        mock_response = {
            "search_rag": string_val,
            "embedding_source_text": "test content",
            "llm_query": "test query"
        }
        mock_llm.get_llm_response.return_value = json.dumps(mock_response)
        
        # Use query that matches the expected boolean value
        query = "@knowledgebase test query" if expected_bool else "test query"
        result = rewriter.transform_query(query)
        assert result['search_rag'] == expected_bool

def test_llm_exception_handling():
    """Test handling of LLM client exceptions."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock LLM client to raise exception
    mock_llm.get_llm_response.side_effect = Exception("LLM service unavailable")
    
    result = rewriter.transform_query("@knowledgebase What is Python?")
    
    # Should use fallback
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "What is Python?"
    assert result['llm_query'] == "@knowledgebase What is Python?"

def test_connection_test_success():
    """Test successful connection test."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock successful response
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "artificial intelligence",
        "llm_query": "What is artificial intelligence?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    success = rewriter.test_connection()
    assert success is True

def test_connection_test_failure():
    """Test connection test failure scenarios."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test with exception
    mock_llm.get_llm_response.side_effect = Exception("Connection failed")
    success = rewriter.test_connection()
    assert success is False
    
    # Test with invalid response structure
    mock_llm.get_llm_response.side_effect = None
    mock_llm.get_llm_response.return_value = json.dumps({"invalid": "response"})
    success = rewriter.test_connection()
    assert success is False

def test_custom_trigger_phrase():
    """Test QueryRewriter with custom trigger phrase."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@search'}
    rewriter = QueryRewriter(mock_llm, config)
    
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "custom search query",
        "llm_query": "Custom search query"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    # Should detect custom trigger
    result = rewriter.transform_query("@search custom query")
    assert result['search_rag'] is True
    
    # Should not detect default trigger
    result = rewriter.transform_query("@knowledgebase default query")
    # The actual detection is done by the fallback logic checking actual trigger
    assert result['search_rag'] is False  # No @search trigger present

def test_query_cleaning_in_fallback():
    """Test query cleaning in fallback scenarios."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Force fallback with exception
    mock_llm.get_llm_response.side_effect = Exception("Test exception")
    
    result = rewriter.transform_query("@knowledgebase    What is machine learning?   ")
    
    # Should clean the trigger phrase and whitespace
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "What is machine learning?"
    assert result['llm_query'] == "@knowledgebase    What is machine learning?   "

def test_empty_query_after_trigger_fallback():
    """Test fallback behavior when query contains only trigger phrase."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Force fallback with exception
    mock_llm.get_llm_response.side_effect = Exception("Test exception")
    
    # Test with just trigger phrase
    result = rewriter.transform_query("@knowledgebase")
    
    # Should detect trigger but not trigger RAG search due to empty content
    assert result['search_rag'] is False  # Should not trigger RAG with empty content
    assert result['embedding_source_text'] == ""  # Empty after cleaning
    assert result['llm_query'] == "@knowledgebase"  # Original query preserved
    
    # Test with trigger phrase and whitespace only
    result = rewriter.transform_query("@knowledgebase   ")
    
    assert result['search_rag'] is False  # Should not trigger RAG with empty content
    assert result['embedding_source_text'] == ""  # Empty after cleaning and stripping
    assert result['llm_query'] == "@knowledgebase   "  # Original query preserved

def test_conversational_follow_up_detection():
    """Test detection of conversational follow-up queries."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response for conversational follow-up
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "",
        "llm_query": "Tell me more about the automation benefits based on context in previous conversation."
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("Tell me more about the automation benefits")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == ""
    assert "based on context in previous conversation" in result['llm_query']

def test_referential_query_detection():
    """Test detection of queries with referential language."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test various referential patterns
    test_cases = [
        ("Can you elaborate on that approach?", "elaborate on that approach based on context in previous conversation"),
        ("What about the other method you mentioned?", "explain the other method based on context in previous conversation"),
        ("How does it compare to traditional methods?", "compare it to traditional methods based on context in previous conversation")
    ]
    
    for query, expected_context_ref in test_cases:
        mock_response = {
            "search_rag": False,
            "embedding_source_text": "",
            "llm_query": expected_context_ref.capitalize() + "."
        }
        mock_llm.get_llm_response.return_value = json.dumps(mock_response)
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is False
        assert "based on context in previous conversation" in result['llm_query']

def test_general_question_no_context_reference():
    """Test that general questions don't reference context."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response for general question
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "",
        "llm_query": "What is the capital of France?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("What is the capital of France?")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == ""
    assert "context" not in result['llm_query']

def test_rag_query_still_uses_provided_context():
    """Test that RAG queries still use 'provided context' instruction."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response for RAG query
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "machine learning benefits applications",
        "llm_query": "Explain machine learning benefits based on the provided context."
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response)
    
    result = rewriter.transform_query("@knowledgebase What are the benefits of machine learning?")
    
    assert result['search_rag'] is True
    assert len(result['embedding_source_text']) > 0
    assert "based on the provided context" in result['llm_query']