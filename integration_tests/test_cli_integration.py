"""
Integration test for CLI with QueryRewriter integration.
Tests the complete flow with real LLM providers.
"""

import sys
import os
from unittest.mock import patch

import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli import RAGCLI

@pytest.fixture
def cli():
    """Provides an initialized RAGCLI instance for tests."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    return RAGCLI(config_path)

class TestCLIIntegration:
    """Integration tests for CLI with QueryRewriter."""
    
    @pytest.mark.integration
    def test_cli_initialization_with_query_rewriter(self, cli: RAGCLI):
        """Test that CLI initializes successfully with QueryRewriter."""
        # Verify QueryRewriter is initialized
        assert cli.query_rewriter is not None
        assert hasattr(cli.query_rewriter, 'transform_query')
        
        # Test QueryRewriter connection
        connection_success = cli.query_rewriter.test_connection()
        assert connection_success, "QueryRewriter should connect successfully"
        
        print("CLI initialized successfully with QueryRewriter")
    
    @pytest.mark.integration 
    def test_query_analysis_integration(self, cli: RAGCLI):
        """Test query analysis with real LLM in CLI context."""
        # Test RAG query analysis
        rag_query = "@knowledgebase What are the benefits of machine learning?"
        result = cli._analyze_and_transform_query(rag_query)
        
        assert result['search_rag'] is True, "Should detect RAG trigger"
        assert 'embedding_texts' in result, "Should have embedding_texts structure"
        assert len(result['embedding_texts']['rewrite']) > 0, "Should have rewrite embedding text"
        assert len(result['llm_query']) > 0, "Should have LLM query"
        assert 'context' in result['llm_query'].lower(), "RAG query should reference context"
        
        # Test non-RAG query analysis
        general_query = "What is the weather today?"
        result = cli._analyze_and_transform_query(general_query)
        
        assert result['search_rag'] is False, "Should not detect RAG trigger"
        assert 'embedding_texts' in result, "Should have embedding_texts structure"
        assert isinstance(result['embedding_texts']['rewrite'], str), "Should have rewrite embedding text field"
        assert len(result['llm_query']) > 0, "Should have LLM query"
        assert 'context' not in result['llm_query'].lower(), "Non-RAG query should not reference context"
        
        print("Query analysis integration working correctly")
    
    @pytest.mark.integration
    def test_prompt_building_integration(self, cli: RAGCLI):
        """Test new prompt building methods work correctly."""
        # Test context-based prompt building
        llm_query = "Explain machine learning benefits based on the provided context."
        context = "Machine learning provides automation and pattern recognition capabilities."
        
        prompt = cli._build_prompt_with_context(llm_query, context)
        assert "Context from knowledge base:" in prompt
        assert context in prompt
        assert llm_query in prompt
        
        # Test no-answer prompt building
        no_answer_prompt = cli._build_no_answer_prompt(llm_query)
        assert "couldn't find relevant information" in no_answer_prompt
        
        print("Prompt building integration working correctly")
    
    @pytest.mark.integration
    def test_fallback_behavior(self, cli: RAGCLI):
        """Test fallback behavior when QueryRewriter fails."""
        # Mock QueryRewriter to raise exception
        with patch.object(cli.query_rewriter, 'transform_query', side_effect=Exception("Test error")):
            
            # Test fallback with trigger phrase
            result = cli._analyze_and_transform_query("@knowledgebase What is Python?")
            assert result['search_rag'] is True, "Fallback should detect trigger"
            assert result['embedding_texts']['rewrite'] == "What is Python?", "Should clean trigger phrase"
            
            # Test fallback without trigger phrase
            result = cli._analyze_and_transform_query("What is Python?")
            assert result['search_rag'] is False, "Fallback should not detect trigger"
            assert result['embedding_texts']['rewrite'] == "What is Python?", "Should use original query"
            
            # Test edge case: trigger phrase only (should not trigger RAG)
            result = cli._analyze_and_transform_query("@knowledgebase")
            assert result['search_rag'] is False, "Empty query after trigger should not trigger RAG"
            assert result['embedding_texts']['rewrite'] == "", "Should have empty embedding text"
            assert result['llm_query'] == "@knowledgebase", "Should preserve original input"
            
            # Test edge case: trigger phrase with only whitespace
            result = cli._analyze_and_transform_query("@knowledgebase   ")
            assert result['search_rag'] is False, "Whitespace-only query after trigger should not trigger RAG"
            assert result['embedding_texts']['rewrite'] == "", "Should have empty embedding text after stripping"
        
        print("Fallback behavior working correctly")