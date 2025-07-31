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

class TestCLIIntegration:
    """Integration tests for CLI with QueryRewriter."""
    
    @pytest.mark.integration
    def test_cli_initialization_with_query_rewriter(self):
        """Test that CLI initializes successfully with QueryRewriter."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        # Test CLI initialization
        cli = RAGCLI(config_path)
        
        # Verify QueryRewriter is initialized
        assert cli.query_rewriter is not None
        assert hasattr(cli.query_rewriter, 'transform_query')
        
        # Test QueryRewriter connection
        connection_success = cli.query_rewriter.test_connection()
        assert connection_success, "QueryRewriter should connect successfully"
        
        print("CLI initialized successfully with QueryRewriter")
    
    @pytest.mark.integration 
    def test_query_analysis_integration(self):
        """Test query analysis with real LLM in CLI context."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        cli = RAGCLI(config_path)
        
        # Test RAG query analysis
        rag_query = "@knowledgebase What are the benefits of machine learning?"
        result = cli._analyze_and_transform_query(rag_query)
        
        assert result['search_rag'] is True, "Should detect RAG trigger"
        assert len(result['embedding_source_text']) > 0, "Should have embedding text"
        assert len(result['llm_query']) > 0, "Should have LLM query"
        assert 'context' in result['llm_query'].lower(), "RAG query should reference context"
        
        # Test non-RAG query analysis
        general_query = "What is the weather today?"
        result = cli._analyze_and_transform_query(general_query)
        
        assert result['search_rag'] is False, "Should not detect RAG trigger"
        assert isinstance(result['embedding_source_text'], str), "Should have embedding text field"
        assert len(result['llm_query']) > 0, "Should have LLM query"
        assert 'context' not in result['llm_query'].lower(), "Non-RAG query should not reference context"
        
        print("Query analysis integration working correctly")
    
    @pytest.mark.integration
    def test_prompt_building_integration(self):
        """Test new prompt building methods work correctly."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        cli = RAGCLI(config_path)
        
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
    def test_fallback_behavior(self):
        """Test fallback behavior when QueryRewriter fails."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        cli = RAGCLI(config_path)
        
        # Mock QueryRewriter to raise exception
        with patch.object(cli.query_rewriter, 'transform_query', side_effect=Exception("Test error")):
            
            # Test fallback with trigger phrase
            result = cli._analyze_and_transform_query("@knowledgebase What is Python?")
            assert result['search_rag'] is True, "Fallback should detect trigger"
            assert result['embedding_source_text'] == "What is Python?", "Should clean trigger phrase"
            
            # Test fallback without trigger phrase
            result = cli._analyze_and_transform_query("What is Python?")
            assert result['search_rag'] is False, "Fallback should not detect trigger"
            assert result['embedding_source_text'] == "What is Python?", "Should use original query"
        
        print("Fallback behavior working correctly")