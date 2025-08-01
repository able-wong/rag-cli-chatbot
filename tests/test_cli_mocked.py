"""
Unit tests for CLI logic with mocked dependencies.
Tests the core RAG logic without requiring external services.
"""

import sys
import os
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli import RAGCLI

class MockScoredPoint:
    """Mock for Qdrant ScoredPoint."""
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload

def test_query_analysis_with_trigger():
    """Test query analysis with RAG trigger phrase."""
    
    # Mock all dependencies
    with patch('cli.ConfigManager') as mock_config, \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.QueryRewriter') as mock_query_rewriter, \
         patch('cli.setup_logging'):
        
        # Setup config mock
        mock_config.return_value.get.return_value = {'enabled': True}
        
        cli = RAGCLI.__new__(RAGCLI)  # Create without calling __init__
        cli.config_manager = mock_config.return_value
        cli.query_rewriter = mock_query_rewriter.return_value
        
        # Mock QueryRewriter response for RAG query
        mock_query_rewriter.return_value.transform_query.return_value = {
            'search_rag': True,
            'embedding_source_text': 'Python programming language',
            'llm_query': 'Explain Python programming based on the provided context.'
        }
        
        result = cli._analyze_and_transform_query("@knowledgebase what is Python?")
        
        assert result['search_rag'] is True
        assert 'Python' in result['embedding_source_text']
        assert 'context' in result['llm_query']

def test_query_analysis_without_trigger():
    """Test query analysis without RAG trigger phrase."""
    
    # Mock all dependencies
    with patch('cli.ConfigManager') as mock_config, \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.QueryRewriter') as mock_query_rewriter, \
         patch('cli.setup_logging'):
        
        # Setup config mock
        mock_config.return_value.get.return_value = {'enabled': True}
        
        cli = RAGCLI.__new__(RAGCLI)  # Create without calling __init__
        cli.config_manager = mock_config.return_value
        cli.query_rewriter = mock_query_rewriter.return_value
        
        # Mock QueryRewriter response for non-RAG query
        mock_query_rewriter.return_value.transform_query.return_value = {
            'search_rag': False,
            'embedding_source_text': 'weather today',
            'llm_query': 'What is the weather today?'
        }
        
        result = cli._analyze_and_transform_query("What is the weather today?")
        
        assert result['search_rag'] is False
        assert result['embedding_source_text'] == 'weather today'
        assert 'context' not in result['llm_query']

def test_rag_context_decision():
    """Test decision logic for using RAG context."""
    
    with patch('cli.ConfigManager'), \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.min_score = 0.7
        
        # Test with high score results
        high_score_results = [
            MockScoredPoint(0.85, {"content": "Test content"}),
            MockScoredPoint(0.75, {"content": "More content"})
        ]
        assert cli._should_use_rag_context(high_score_results)
        
        # Test with low score results
        low_score_results = [
            MockScoredPoint(0.65, {"content": "Test content"}),
            MockScoredPoint(0.55, {"content": "More content"})
        ]
        assert not cli._should_use_rag_context(low_score_results)
        
        # Test with empty results
        assert not cli._should_use_rag_context([])

def test_conversation_history_management():
    """Test conversation history length management."""
    
    with patch('cli.ConfigManager'), \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.max_history_length = 2  # Keep only 2 user-assistant pairs
        
        # Create a conversation history that exceeds the limit
        cli.conversation_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Message 4"},
            {"role": "assistant", "content": "Response 4"},
        ]
        
        cli._manage_conversation_history()
        
        # Should keep system prompt + last 2 pairs (4 messages) = 5 total
        assert len(cli.conversation_history) == 5
        assert cli.conversation_history[0]["role"] == "system"
        assert cli.conversation_history[-1]["content"] == "Response 4"

def test_rag_context_building():
    """Test RAG context string building."""
    
    with patch('cli.ConfigManager'), \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.max_context_length = 200  # Small limit for testing
        
        results = [
            MockScoredPoint(0.9, {
                "content": "Short content", 
                "source": "doc1.txt"
            }),
            MockScoredPoint(0.8, {
                "content": "This is a longer piece of content that might exceed the context length limit when combined with other results",
                "source": "doc2.txt"
            })
        ]
        
        context = cli._build_rag_context(results)
        
        # Should include at least the first result
        assert "Short content" in context
        assert "Source 1" in context
        assert len(context) <= cli.max_context_length + 100  # Some tolerance

def test_prompt_building():
    """Test prompt building with new structured approach."""
    
    with patch('cli.ConfigManager'), \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.QueryRewriter'), \
         patch('cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        
        # Test RAG prompt building with structured LLM query
        llm_query = "Explain Python programming based on the provided context."
        context = "Python is a programming language."
        
        rag_prompt = cli._build_prompt_with_context(llm_query, context)
        assert "Context from knowledge base:" in rag_prompt
        assert "Python is a programming language" in rag_prompt
        assert "Task: Explain Python programming based on the provided context." in rag_prompt
        
        # Test no-answer prompt building
        no_answer_prompt = cli._build_no_answer_prompt(llm_query)
        assert "couldn't find relevant information" in no_answer_prompt

def test_configurable_system_prompt():
    """Test that CLI uses configurable system prompt from config."""
    
    # Mock all dependencies
    with patch('cli.ConfigManager') as mock_config, \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.QueryRewriter'), \
         patch('cli.setup_logging'):
        
        # Setup config mock with custom system prompt
        custom_prompt = "You are John's writing assistant. Your job is helping John research and write blog posts. Follow the task instructions carefully and use the specified context source as directed. If you don't know the answer based on the specified context or from conversation history, you can say you don't know."
        
        config_instance = mock_config.return_value
        config_instance.get_rag_config.return_value = {'trigger_phrase': '@knowledgebase', 'top_k': 5, 'min_score': 0.7, 'max_context_length': 8000}
        config_instance.get_vector_db_config.return_value = {}
        config_instance.get_embedding_config.return_value = {}
        config_instance.get_llm_config.return_value = {}
        config_instance.get_documents_config.return_value = {}
        config_instance.get_logging_config.return_value = {}
        config_instance.get.side_effect = lambda key, default=None: {
            'cli': {'system_prompt': custom_prompt, 'max_history_length': 20},
            'query_rewriter': {'enabled': True}
        }.get(key, default)
        
        # Initialize CLI
        cli = RAGCLI()
        
        # Check that conversation history uses custom system prompt
        assert len(cli.conversation_history) == 1
        assert cli.conversation_history[0]['role'] == 'system'
        assert cli.conversation_history[0]['content'] == custom_prompt
        assert "John's writing assistant" in cli.conversation_history[0]['content']

def test_default_system_prompt_fallback():
    """Test that CLI falls back to default system prompt if not configured."""
    
    # Mock all dependencies
    with patch('cli.ConfigManager') as mock_config, \
         patch('cli.EmbeddingClient'), \
         patch('cli.QdrantDB'), \
         patch('cli.LLMClient'), \
         patch('cli.QueryRewriter'), \
         patch('cli.setup_logging'):
        
        # Setup config mock without custom system prompt
        config_instance = mock_config.return_value
        config_instance.get_rag_config.return_value = {'trigger_phrase': '@knowledgebase', 'top_k': 5, 'min_score': 0.7, 'max_context_length': 8000}
        config_instance.get_vector_db_config.return_value = {}
        config_instance.get_embedding_config.return_value = {}
        config_instance.get_llm_config.return_value = {}
        config_instance.get_documents_config.return_value = {}
        config_instance.get_logging_config.return_value = {}
        config_instance.get.side_effect = lambda key, default=None: {
            'cli': {'max_history_length': 20},  # No system_prompt key
            'query_rewriter': {'enabled': True}
        }.get(key, default)
        
        # Initialize CLI
        cli = RAGCLI()
        
        # Check that conversation history uses default system prompt
        assert len(cli.conversation_history) == 1
        assert cli.conversation_history[0]['role'] == 'system'
        default_prompt = "You are a helpful AI assistant. Follow the task instructions carefully and use the specified context source as directed. If you don't know the answer based on the specified context or from conversation history, you can say you don't know."
        assert cli.conversation_history[0]['content'] == default_prompt

