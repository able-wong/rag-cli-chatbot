"""
Integration tests for QueryRewriter with real LLM providers.
Minimal tests to confirm actual LLM behavior that can't be mocked.
"""

import sys
import os
from typing import Dict, Any

import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter
from llm_client import LLMClient
from config_manager import ConfigManager

class TestQueryRewriterIntegration:
    """Integration tests for QueryRewriter with real LLM."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with real configuration."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        cls.config_manager = ConfigManager(config_path)
        
        # Initialize real LLM client
        llm_config = cls.config_manager.get_llm_config()
        cls.llm_client = LLMClient(llm_config)
        
        # Initialize QueryRewriter with real dependencies
        query_rewriter_config = cls.config_manager.get('query_rewriter', {})
        query_rewriter_config['trigger_phrase'] = cls.config_manager.get('rag.trigger_phrase', '@knowledgebase')
        cls.query_rewriter = QueryRewriter(cls.llm_client, query_rewriter_config)
    
    def validate_json_structure(self, result: Dict[str, Any]) -> bool:
        """Validate that result has expected structure."""
        required_fields = ['search_rag', 'embedding_source_text', 'llm_query']
        
        if not all(field in result for field in required_fields):
            return False
        
        if not isinstance(result['search_rag'], bool):
            return False
        
        if not isinstance(result['embedding_source_text'], str) or not result['embedding_source_text'].strip():
            return False
        
        if not isinstance(result['llm_query'], str) or not result['llm_query'].strip():
            return False
        
        return True
    
    @pytest.mark.integration
    def test_rag_query_with_real_llm(self):
        """Test RAG query transformation with real LLM - confirms JSON parsing and context instruction."""
        user_query = "@knowledgebase What are the benefits of machine learning?"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure (most critical - confirms JSON parsing works with real LLM)
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        
        # Validate trigger detection
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate embedding text doesn't contain trigger phrase
        assert '@knowledgebase' not in result['embedding_source_text'].lower(), "Embedding text should not contain trigger phrase"
        
        # Validate RAG query includes context reference (critical for RAG functionality)
        llm_query = result['llm_query'].lower()
        context_indicators = ['based on', 'context', 'provided', 'from the']
        has_context_ref = any(indicator in llm_query for indicator in context_indicators)
        assert has_context_ref, f"RAG query should reference context. Got: {result['llm_query']}"
        
        print("RAG Integration Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
    
    @pytest.mark.integration
    def test_non_rag_query_with_real_llm(self):
        """Test non-RAG query transformation with real LLM - confirms general instruction format."""
        user_query = "What is machine learning?"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        
        # Validate trigger detection
        assert result['search_rag'] is False, "Should not detect RAG trigger"
        
        # Validate non-RAG query doesn't mention context (important distinction)
        llm_query = result['llm_query'].lower()
        context_indicators = ['based on', 'context', 'provided', 'from the']
        has_context_ref = any(indicator in llm_query for indicator in context_indicators)
        assert not has_context_ref, f"Non-RAG query should not reference context. Got: {result['llm_query']}"
        
        print("Non-RAG Integration Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
    
    @pytest.mark.integration
    def test_real_llm_connection_and_error_handling(self):
        """Test that real LLM connection works and error handling is robust."""
        # Test connection (confirms real LLM is accessible)
        connection_success = self.query_rewriter.test_connection()
        assert connection_success, "QueryRewriter should successfully connect to real LLM"
        
        # Test that the system can handle real LLM responses consistently
        # This is important because real LLMs may return slightly different formats
        test_query = "@knowledgebase How does neural network training work?"
        
        # Run same query twice to check consistency in parsing
        result1 = self.query_rewriter.transform_query(test_query)
        result2 = self.query_rewriter.transform_query(test_query)
        
        # Both should have valid structure (tests robustness of JSON parsing)
        assert self.validate_json_structure(result1), "First result should have valid structure"
        assert self.validate_json_structure(result2), "Second result should have valid structure"
        
        # Both should detect trigger correctly (tests consistency)
        assert result1['search_rag'] is True, "First result should detect trigger"
        assert result2['search_rag'] is True, "Second result should detect trigger"
        
        print(f"Connection Test: {connection_success}")
        print("Consistency Test - both results have valid structure and trigger detection")