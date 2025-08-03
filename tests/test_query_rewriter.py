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
    # Ensure both methods are available for backward compatibility during transition
    mock_client.get_llm_response = Mock()
    mock_client.get_json_response = Mock()
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
    assert rewriter.retrieval_strategy == 'rewrite'  # Default strategy
    assert rewriter.temperature == 0.1
    assert rewriter.max_tokens == 512

def test_query_rewriter_hyde_initialization():
    """Test QueryRewriter initialization with HyDE strategy."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde',
        'temperature': 0.1,
        'max_tokens': 512
    }
    
    rewriter = QueryRewriter(mock_llm, config)
    
    assert rewriter.llm_client == mock_llm
    assert rewriter.trigger_phrase == '@knowledgebase'
    assert rewriter.retrieval_strategy == 'hyde'
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
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase What is Python?")
    
    assert result['search_rag'] is True
    assert 'Python' in result['embedding_source_text']
    assert result['llm_query'] == "What is Python programming language?"
    mock_llm.get_json_response.assert_called_once()

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
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("What is the weather today?")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == "weather today"
    assert result['llm_query'] == "What is the weather today?"

def test_json_parsing_markdown_wrapped():
    """Test that QueryRewriter works with LLMClient's JSON parsing (including markdown handling)."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock JSON response (LLMClient handles markdown parsing internally)
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "machine learning algorithms",
        "llm_query": "Explain machine learning algorithms"
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase Explain ML algorithms")
    
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "machine learning algorithms"
    assert result['llm_query'] == "Explain machine learning algorithms"

def test_json_parsing_with_extra_text():
    """Test that QueryRewriter works with LLMClient's JSON parsing (including extraction from text)."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock JSON response (LLMClient handles text extraction internally)
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "cats behavior",
        "llm_query": "Why do cats purr?"
    }
    mock_llm.get_json_response.return_value = mock_response
    
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
    mock_llm.get_json_response.return_value = mock_response
    
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
    mock_llm.get_json_response.return_value = mock_response
    
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
    mock_llm.get_json_response.return_value = mock_response
    
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
    mock_llm.get_json_response.return_value = mock_response
    
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
        mock_llm.get_json_response.return_value = mock_response
        
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
    mock_llm.get_json_response.side_effect = Exception("LLM service unavailable")
    
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
        "llm_query": "What is artificial intelligence?",
        "hard_filters": {}
    }
    mock_llm.get_json_response.return_value = mock_response
    
    success = rewriter.test_connection()
    assert success is True

def test_connection_test_failure():
    """Test connection test failure scenarios."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test with exception
    mock_llm.get_json_response.side_effect = Exception("Connection failed")
    success = rewriter.test_connection()
    assert success is False
    
    # Test with invalid response structure
    mock_llm.get_json_response.side_effect = None
    mock_llm.get_json_response.return_value = {"invalid": "response"}
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
    mock_llm.get_json_response.return_value = mock_response
    
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
    mock_llm.get_json_response.side_effect = Exception("Test exception")
    
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
    mock_llm.get_json_response.side_effect = Exception("Test exception")
    
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
    mock_llm.get_json_response.return_value = mock_response
    
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
        mock_llm.get_json_response.return_value = mock_response
        
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
    mock_llm.get_json_response.return_value = mock_response
    
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
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase What are the benefits of machine learning?")
    
    assert result['search_rag'] is True
    assert len(result['embedding_source_text']) > 0
    assert "based on the provided context" in result['llm_query']

# ============================================================================
# HyDE Strategy Tests
# ============================================================================

def test_hyde_strategy_system_prompt_selection():
    """Test that HyDE strategy uses the correct system prompt."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    
    rewriter = QueryRewriter(mock_llm, config)
    
    # Check that the system prompt contains HyDE-specific instructions
    assert "HyDE (Hypothetical Document Embeddings)" in rewriter.system_prompt
    assert "hypothetical document" in rewriter.system_prompt

def test_rewrite_strategy_system_prompt_selection():
    """Test that rewrite strategy uses the correct system prompt."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'rewrite'
    }
    
    rewriter = QueryRewriter(mock_llm, config)
    
    # Check that the system prompt contains rewrite-specific instructions
    assert "Core topic keywords only, ignoring instruction words" in rewriter.system_prompt
    assert "HyDE" not in rewriter.system_prompt

def test_hyde_strategy_hypothetical_document_generation():
    """Test HyDE strategy generates hypothetical documents."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock HyDE response with hypothetical document
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes called neurons that process information through weighted connections. During training, networks learn by adjusting these weights through backpropagation, a process that calculates errors and propagates them backward through the network to optimize performance.",
        "llm_query": "Explain how neural networks work based on the provided context."
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase How do neural networks work?")
    
    assert result['search_rag'] is True
    # HyDE should generate a comprehensive hypothetical document
    assert len(result['embedding_source_text']) > 50  # Much longer than keywords
    assert "neural networks" in result['embedding_source_text'].lower()
    assert "training" in result['embedding_source_text'].lower()
    assert "based on the provided context" in result['llm_query']

def test_hyde_vs_rewrite_embedding_text_difference():
    """Test that HyDE and rewrite strategies produce different embedding text."""
    mock_llm = create_mock_llm_client()
    
    # Test rewrite strategy
    rewrite_config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'rewrite'
    }
    rewrite_rewriter = QueryRewriter(mock_llm, rewrite_config)
    
    rewrite_response = {
        "search_rag": True,
        "embedding_source_text": "machine learning algorithms applications",  # Keywords
        "llm_query": "Explain machine learning based on the provided context."
    }
    mock_llm.get_json_response.return_value = rewrite_response
    
    rewrite_result = rewrite_rewriter.transform_query("@knowledgebase What is machine learning?")
    
    # Test HyDE strategy
    hyde_config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    hyde_rewriter = QueryRewriter(mock_llm, hyde_config)
    
    hyde_response = {
        "search_rag": True,
        "embedding_source_text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or classifications on new, unseen data.",  # Hypothetical document
        "llm_query": "Explain machine learning based on the provided context."
    }
    mock_llm.get_json_response.return_value = hyde_response
    
    hyde_result = hyde_rewriter.transform_query("@knowledgebase What is machine learning?")
    
    # Compare results
    assert rewrite_result['search_rag'] is True
    assert hyde_result['search_rag'] is True
    
    # HyDE should produce much longer, more descriptive text
    assert len(hyde_result['embedding_source_text']) > len(rewrite_result['embedding_source_text'])
    assert len(hyde_result['embedding_source_text']) > 100  # Should be document-like
    assert len(rewrite_result['embedding_source_text']) < 50  # Should be keyword-like

def test_hyde_strategy_non_rag_queries():
    """Test that HyDE strategy handles non-RAG queries correctly."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response for non-RAG query
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "",
        "llm_query": "What is the weather today?"
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("What is the weather today?")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == ""
    assert result['llm_query'] == "What is the weather today?"

def test_hyde_strategy_conversational_queries():
    """Test that HyDE strategy handles conversational follow-ups correctly."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response for conversational follow-up
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "",
        "llm_query": "Tell me more about that approach based on context in previous conversation."
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("Tell me more about that approach")
    
    assert result['search_rag'] is False
    assert result['embedding_source_text'] == ""
    assert "based on context in previous conversation" in result['llm_query']

def test_hyde_strategy_fallback_behavior():
    """Test that HyDE strategy fallback works correctly."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock LLM exception to trigger fallback
    mock_llm.get_json_response.side_effect = Exception("LLM service unavailable")
    
    result = rewriter.transform_query("@knowledgebase What is Python?")
    
    # Should use fallback logic (same for both strategies)
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "What is Python?"
    assert result['llm_query'] == "@knowledgebase What is Python?"

def test_hyde_strategy_connection_test():
    """Test connection test works with HyDE strategy."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock successful response
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses various subfields including machine learning, natural language processing, and computer vision.",
        "llm_query": "What is artificial intelligence based on the provided context?",
        "hard_filters": {}
    }
    mock_llm.get_json_response.return_value = mock_response
    
    success = rewriter.test_connection()
    assert success is True

# ============================================================================
# Hybrid Search / Metadata Filtering Tests
# ============================================================================

def test_filters_field_initialization():
    """Test that hard_hard_filters field is properly initialized in responses."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock response without hard_hard_filters field
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "Python programming",
        "llm_query": "Explain Python programming based on the provided context."
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase What is Python?")
    
    # Should add empty hard_hard_filters field
    assert 'hard_filters' in result
    assert result['hard_filters'] == {}

def test_author_filter_extraction():
    """Test extraction of author filters from natural language."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test various author filter patterns
    test_cases = [
        ("@knowledgebase papers by John Smith", {"author": "John Smith"}),
        ("@knowledgebase articles by Jane Doe", {"author": "Jane Doe"}),
        ("@knowledgebase research from Dr. Wilson", {"author": "Dr. Wilson"}),
        ("@knowledgebase Smith's work on AI", {"author": "Smith"}),
    ]
    
    for query, expected_filters in test_cases:
        mock_response = {
            "search_rag": True,
            "embedding_source_text": "research topic",
            "llm_query": "Based on the provided context, provide information about the research.",
            "hard_filters": expected_filters
        }
        mock_llm.get_json_response.return_value = mock_response
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is True
        assert result['hard_filters'] == expected_filters
        assert result['hard_filters']['author'] == expected_filters['author']

def test_publication_date_filter_extraction():
    """Test extraction of publication date filters from natural language."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test various date filter patterns
    test_cases = [
        ("@knowledgebase articles from 2023", {"publication_date": "2023"}),
        ("@knowledgebase papers published in 2024", {"publication_date": "2024"}),
        ("@knowledgebase research from March 2025", {"publication_date": "2025-03"}),
        ("@knowledgebase documents from last year", {"publication_date": "2023"}),
    ]
    
    for query, expected_filters in test_cases:
        mock_response = {
            "search_rag": True,
            "embedding_source_text": "research topic",
            "llm_query": "Based on the provided context, provide information about the research.",
            "hard_filters": expected_filters
        }
        mock_llm.get_json_response.return_value = mock_response
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is True
        assert result['hard_filters'] == expected_filters
        assert result['hard_filters']['publication_date'] == expected_filters['publication_date']

def test_explicit_tag_syntax_single_tag():
    """Test explicit tag syntax with single tag."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test explicit tag syntax patterns
    test_cases = [
        ("@knowledgebase documents with tag Python", {"tags": ["python"]}),
        ("@knowledgebase articles tagged as machine learning", {"tags": ["machine learning"]}),
        ("@knowledgebase search with tag AI", {"tags": ["ai"]}),
    ]
    
    for query, expected_filters in test_cases:
        mock_response = {
            "search_rag": True,
            "embedding_source_text": "research topic",
            "llm_query": "Based on the provided context, provide information about the research.",
            "hard_filters": expected_filters
        }
        mock_llm.get_json_response.return_value = mock_response
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is True
        assert result['hard_filters'] == expected_filters
        assert result['hard_filters']['tags'] == expected_filters['tags']

def test_explicit_tag_syntax_multiple_tags():
    """Test explicit tag syntax with multiple tags."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test multiple explicit tags
    test_cases = [
        ("@knowledgebase articles with tags python, AI", {"tags": ["python", "ai"]}),
        ("@knowledgebase research with tags machine learning, deep learning", {"tags": ["machine learning", "deep learning"]}),
    ]
    
    for query, expected_filters in test_cases:
        mock_response = {
            "search_rag": True,
            "embedding_source_text": "research topic",
            "llm_query": "Based on the provided context, provide information about the research.",
            "hard_filters": expected_filters
        }
        mock_llm.get_json_response.return_value = mock_response
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is True
        assert result['hard_filters'] == expected_filters
        assert result['hard_filters']['tags'] == expected_filters['tags']

def test_no_auto_tag_extraction():
    """Test that topics don't auto-extract as tags (broader search)."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test queries that should NOT extract tags automatically
    test_cases = [
        ("@knowledgebase documents about Python", {}),
        ("@knowledgebase articles on machine learning", {}),
        ("@knowledgebase research about AI and robotics", {}),
        ("@knowledgebase papers on web development", {}),
        ("@knowledgebase search on vibe coding", {}),
    ]
    
    for query, expected_filters in test_cases:
        mock_response = {
            "search_rag": True,
            "embedding_source_text": "research topic",
            "llm_query": "Based on the provided context, provide information about the research.",
            "hard_filters": expected_filters
        }
        mock_llm.get_json_response.return_value = mock_response
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is True
        assert result['hard_filters'] == expected_filters
        # Verify no tags were extracted
        assert 'tags' not in result['hard_filters']

def test_mixed_explicit_and_auto_filters():
    """Test explicit tags with auto author/date extraction."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test combining explicit tags with auto-extracted author/date
    test_cases = [
        ("@knowledgebase papers by John Smith with tag machine learning", 
         {"author": "John Smith", "tags": ["machine learning"]}),
        ("@knowledgebase articles with tags python, AI from 2023", 
         {"tags": ["python", "ai"], "publication_date": "2023"}),
        ("@knowledgebase search by Jane Doe with tag robotics from March 2024",
         {"author": "Jane Doe", "tags": ["robotics"], "publication_date": "2024-03"}),
    ]
    
    for query, expected_filters in test_cases:
        mock_response = {
            "search_rag": True,
            "embedding_source_text": "research topic",
            "llm_query": "Based on the provided context, provide information about the research.",
            "hard_filters": expected_filters
        }
        mock_llm.get_json_response.return_value = mock_response
        
        result = rewriter.transform_query(query)
        
        assert result['search_rag'] is True
        assert result['hard_filters'] == expected_filters

def test_combined_filters_extraction():
    """Test extraction of multiple filters from single query."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test complex query with multiple filters
    query = "@knowledgebase Smith's papers about AI from 2023"
    expected_filters = {
        "author": "Smith",
        "tags": ["ai"],
        "publication_date": "2023"
    }
    
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "artificial intelligence research",
        "llm_query": "Based on the provided context, provide information about Smith's AI research from 2023.",
        "hard_filters": expected_filters
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query(query)
    
    assert result['search_rag'] is True
    assert result['hard_filters'] == expected_filters
    assert result['hard_filters']['author'] == "Smith"
    assert result['hard_filters']['tags'] == ["ai"]
    assert result['hard_filters']['publication_date'] == "2023"

def test_complex_natural_language_query():
    """Test the complex example from the user: vibe coding from John Wong published in March 2025."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    query = "@knowledgebase search on vibe coding from John Wong published in March 2025, then explain what is vibe coding, and pros/cons"
    expected_filters = {
        "author": "John Wong",
        "tags": ["vibe coding"],
        "publication_date": "2025-03"
    }
    
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "vibe coding programming approach",
        "llm_query": "Based on the provided context, explain what vibe coding is, including its pros and cons, and cite sources.",
        "hard_filters": expected_filters
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query(query)
    
    assert result['search_rag'] is True
    assert result['hard_filters'] == expected_filters
    assert result['hard_filters']['author'] == "John Wong"
    assert result['hard_filters']['tags'] == ["vibe coding"]
    assert result['hard_filters']['publication_date'] == "2025-03"

def test_filters_field_validation():
    """Test validation of hard_filters field structure."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test with invalid filters type (not a dict)
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "test content",
        "llm_query": "test query",
        "hard_filters": "invalid_string"  # Should be dict
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase test query")
    
    # Should reset to empty dict
    assert result['hard_filters'] == {}

def test_filters_with_hyde_strategy():
    """Test that filters work correctly with HyDE strategy."""
    mock_llm = create_mock_llm_client()
    config = {
        'trigger_phrase': '@knowledgebase',
        'retrieval_strategy': 'hyde'
    }
    rewriter = QueryRewriter(mock_llm, config)
    
    query = "@knowledgebase papers by John Smith about machine learning"
    expected_filters = {
        "author": "John Smith",
        "tags": ["machine learning"]
    }
    
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns, make predictions, and improve performance through experience.",
        "llm_query": "Based on the provided context, provide information about machine learning from John Smith's papers.",
        "hard_filters": expected_filters
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query(query)
    
    assert result['search_rag'] is True
    assert result['hard_filters'] == expected_filters
    assert len(result['embedding_source_text']) > 100  # HyDE should generate longer text

def test_no_filters_for_non_rag_queries():
    """Test that non-RAG queries still get empty filters."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "",
        "llm_query": "What is the weather today?",
        "hard_filters": {}
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("What is the weather today?")
    
    assert result['search_rag'] is False
    assert result['hard_filters'] == {}

def test_filters_fallback_behavior():
    """Test that fallback logic includes empty hard_filters field."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Force fallback with exception
    mock_llm.get_json_response.side_effect = Exception("LLM service unavailable")
    
    result = rewriter.transform_query("@knowledgebase What is Python?")
    
    # Should use fallback and include empty filters
    assert result['search_rag'] is True
    assert result['embedding_source_text'] == "What is Python?"
    assert result['llm_query'] == "@knowledgebase What is Python?"
    assert 'hard_filters' in result
    assert result['hard_filters'] == {}

def test_empty_filters_handling():
    """Test handling of empty or null filters in response."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test with null filters
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "test content",
        "llm_query": "test query",
        "hard_filters": None
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("@knowledgebase test query")
    
    # Should convert None to empty dict
    assert result['hard_filters'] == {}

def test_partial_filters_extraction():
    """Test queries that only extract some types of filters."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Test query with only author filter
    query = "@knowledgebase latest papers by Dr. Johnson"
    expected_filters = {"author": "Dr. Johnson"}
    
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "latest research papers",
        "llm_query": "Based on the provided context, provide information about Dr. Johnson's latest papers.",
        "hard_filters": expected_filters
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query(query)
    
    assert result['search_rag'] is True
    assert result['hard_filters'] == expected_filters
    assert 'tags' not in result['hard_filters']
    assert 'publication_date' not in result['hard_filters']

def test_filters_with_conversational_queries():
    """Test that conversational queries don't extract filters."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    mock_response = {
        "search_rag": False,
        "embedding_source_text": "",
        "llm_query": "Tell me more about those techniques based on context in previous conversation.",
        "hard_filters": {}
    }
    mock_llm.get_json_response.return_value = mock_response
    
    result = rewriter.transform_query("Tell me more about those techniques")
    
    assert result['search_rag'] is False
    assert result['hard_filters'] == {}
    assert "based on context in previous conversation" in result['llm_query']

def test_filters_in_connection_test():
    """Test that connection test handles hard_filters field correctly."""
    mock_llm = create_mock_llm_client()
    config = {'trigger_phrase': '@knowledgebase'}
    rewriter = QueryRewriter(mock_llm, config)
    
    # Mock successful response with filters
    mock_response = {
        "search_rag": True,
        "embedding_source_text": "artificial intelligence",
        "llm_query": "What is artificial intelligence?",
        "hard_filters": {"tags": ["ai"]}
    }
    mock_llm.get_json_response.return_value = mock_response
    
    success = rewriter.test_connection()
    assert success is True
    
    # Test that connection test works even without hard_filters field
    mock_response_no_filters = {
        "search_rag": True,
        "embedding_source_text": "artificial intelligence",
        "llm_query": "What is artificial intelligence?"
    }
    mock_llm.get_llm_response.return_value = json.dumps(mock_response_no_filters)
    
    success = rewriter.test_connection()
    assert success is True