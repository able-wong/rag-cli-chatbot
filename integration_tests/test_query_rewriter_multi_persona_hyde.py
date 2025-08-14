#!/usr/bin/env python3
"""
Integration test for QueryRewriter multi-persona HyDE functionality.
Tests the new unified prompt that returns both rewrite and hyde formats.
"""

import sys
import os
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter
from llm_client import LLMClient
from config_manager import ConfigManager


@pytest.mark.integration
class TestQueryRewriterMultiPersonaHyde:
    """Integration tests for QueryRewriter multi-persona HyDE functionality."""

    @pytest.fixture(scope="class")
    def query_rewriter(self):
        """Create QueryRewriter instance for testing."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config_manager = ConfigManager(config_path)
        
        # Initialize LLM client
        llm_config = config_manager.get_llm_config()
        llm_client = LLMClient(llm_config)
        
        # Create query rewriter config
        query_rewriter_config = config_manager.get('query_rewriter', {})
        query_rewriter_config['trigger_phrase'] = config_manager.get('rag.trigger_phrase', '@knowledgebase')
        
        return QueryRewriter(llm_client, query_rewriter_config)

    def assert_llm_response_structure(self, result, user_query):
        """Assert that result came from LLM and has proper multi-persona structure."""
        # Check source field
        assert result.get('source') == 'llm', f"Expected LLM response, got {result.get('source')} for query: {user_query}"
        
        # Check required fields
        required_fields = ['search_rag', 'embedding_texts', 'llm_query', 'hard_filters', 'negation_filters', 'soft_filters', 'source']
        for field in required_fields:
            assert field in result, f"Missing required field '{field}' in result for query: {user_query}"
        
        # Check embedding_texts structure
        embedding_texts = result['embedding_texts']
        assert isinstance(embedding_texts, dict), f"embedding_texts should be dict for query: {user_query}"
        assert 'rewrite' in embedding_texts, f"Missing 'rewrite' in embedding_texts for query: {user_query}"
        assert 'hyde' in embedding_texts, f"Missing 'hyde' in embedding_texts for query: {user_query}"
        
        # Check hyde is array of 3 strings
        hyde = embedding_texts['hyde']
        assert isinstance(hyde, list), f"hyde should be list for query: {user_query}"
        assert len(hyde) == 3, f"hyde should have 3 items, got {len(hyde)} for query: {user_query}"
        
        # Check rewrite format
        rewrite = embedding_texts['rewrite']
        assert isinstance(rewrite, str), f"rewrite should be string for query: {user_query}"
        
        # For RAG queries, validate multi-persona structure
        if result['search_rag']:
            assert len(rewrite.strip()) > 0, f"rewrite should not be empty for RAG query: {user_query}"
            
            # Check that hyde texts are different (indicating different personas)
            unique_hyde = set(hyde)
            assert len(unique_hyde) >= 2, f"hyde texts should be different (multi-persona) for query: {user_query}"
            
            # Check that each hyde text is substantial
            for i, hyde_text in enumerate(hyde):
                assert isinstance(hyde_text, str), f"hyde[{i}] should be string for query: {user_query}"
                assert len(hyde_text.strip()) > 20, f"hyde[{i}] should be substantial text for query: {user_query}"

    def test_neural_network_science_topic(self, query_rewriter):
        """Test multi-persona HyDE generation for science topic."""
        query = "@knowledgebase How does neural network training work?"
        result = query_rewriter.transform_query(query)
        
        self.assert_llm_response_structure(result, query)
        
        # Check specific expectations for neural network topic
        assert result['search_rag'] is True
        assert 'neural network' in result['embedding_texts']['rewrite'].lower()
        
        # Verify different personas in hyde texts
        hyde_texts = result['embedding_texts']['hyde']
        combined_text = ' '.join(hyde_texts).lower()
        
        # Should contain technical terms (professor perspective)
        # Should contain educational terms (teacher perspective)  
        # Should contain learning-focused language (student perspective)
        assert any(term in combined_text for term in ['algorithm', 'process', 'training', 'learning']), \
            "Hyde texts should contain relevant technical terms for neural networks"

    def test_business_automation_topic(self, query_rewriter):
        """Test multi-persona HyDE generation for business topic."""
        query = "@knowledgebase what are the benefits of automation in business"
        result = query_rewriter.transform_query(query)
        
        self.assert_llm_response_structure(result, query)
        
        # Check specific expectations for business topic
        assert result['search_rag'] is True
        assert 'automation' in result['embedding_texts']['rewrite'].lower()
        assert 'business' in result['embedding_texts']['rewrite'].lower()
        
        # Verify business-oriented language in hyde texts
        hyde_texts = result['embedding_texts']['hyde']
        combined_text = ' '.join(hyde_texts).lower()
        
        # Should contain business terms from different perspectives
        assert any(term in combined_text for term in ['cost', 'efficiency', 'productivity', 'benefit']), \
            "Hyde texts should contain relevant business terms for automation"

    def test_programming_general_topic(self, query_rewriter):
        """Test multi-persona HyDE generation for general programming topic."""
        query = "@knowledgebase What is Python programming?"
        result = query_rewriter.transform_query(query)
        
        self.assert_llm_response_structure(result, query)
        
        # Check specific expectations for programming topic
        assert result['search_rag'] is True
        assert 'python' in result['embedding_texts']['rewrite'].lower()
        
        # Verify programming-oriented language in hyde texts
        hyde_texts = result['embedding_texts']['hyde']
        combined_text = ' '.join(hyde_texts).lower()
        
        # Should contain programming terms
        assert any(term in combined_text for term in ['programming', 'language', 'code', 'developer']), \
            "Hyde texts should contain relevant programming terms for Python"

    def test_search_with_filters_query(self, query_rewriter):
        """Test multi-persona HyDE with metadata filters."""
        query = "@knowledgebase find papers by John Smith about machine learning"
        result = query_rewriter.transform_query(query)
        
        self.assert_llm_response_structure(result, query)
        
        # Check that it's a search query
        assert result['search_rag'] is True
        assert 'machine learning' in result['embedding_texts']['rewrite'].lower()
        
        # Should still generate diverse hyde texts even for search queries
        hyde_texts = result['embedding_texts']['hyde']
        assert len(set(hyde_texts)) >= 2, "Search queries should still generate diverse hyde texts"

    def test_non_rag_conversational_query(self, query_rewriter):
        """Test multi-persona structure for non-RAG queries."""
        query = "Tell me more about the benefits"
        result = query_rewriter.transform_query(query)
        
        self.assert_llm_response_structure(result, query)
        
        # Check non-RAG expectations
        assert result['search_rag'] is False
        
        # For non-RAG queries, texts may be empty or minimal
        embedding_texts = result['embedding_texts']
        assert isinstance(embedding_texts['rewrite'], str)
        assert isinstance(embedding_texts['hyde'], list)
        assert len(embedding_texts['hyde']) == 3

    def test_unified_embedding_structure(self, query_rewriter):
        """Test that the unified embedding structure works correctly."""
        query = "@knowledgebase explain REST vs GraphQL APIs"
        result = query_rewriter.transform_query(query)
        
        self.assert_llm_response_structure(result, query)
        
        # Check unified structure
        assert 'embedding_texts' in result, "Should have embedding_texts structure"
        assert 'rewrite' in result['embedding_texts'], "Should have rewrite text"
        assert 'hyde' in result['embedding_texts'], "Should have hyde texts"
        
        # For API topic, should contain relevant terms
        combined_text = (result['embedding_texts']['rewrite'] + ' ' + ' '.join(result['embedding_texts']['hyde'])).lower()
        assert any(term in combined_text for term in ['api', 'rest', 'graphql']), \
            "Should contain relevant API terms"

    def test_connection_and_consistency(self, query_rewriter):
        """Test LLM connection and consistency across multiple calls."""
        query = "@knowledgebase papers by Smith"
        
        results = []
        for i in range(2):
            result = query_rewriter.transform_query(query)
            self.assert_llm_response_structure(result, query)
            results.append(result)
        
        # All results should come from LLM
        for result in results:
            assert result['source'] == 'llm', "All calls should use LLM, not fallback"
        
        # Results should be consistent in structure
        for result in results:
            assert result['search_rag'] is True
            assert 'embedding_texts' in result
            assert len(result['embedding_texts']['hyde']) == 3


def main():
    """Run as standalone script for manual testing."""
    import pytest
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()