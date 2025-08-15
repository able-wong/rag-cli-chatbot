"""
Integration tests for QueryRewriter improvements.

Tests the intent-based filter detection and LLM query cleaning functionality
introduced to address GitHub issue #20.
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter
from llm_client import LLMClient


class TestQueryRewriterIntegration:
    """Integration tests for QueryRewriter improvements."""

    @pytest.fixture(scope="class")
    def query_rewriter(self):
        """Create QueryRewriter instance for testing."""
        try:
            # Load config using ConfigManager (handles environment variables properly)
            from config_manager import ConfigManager
            
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.sample.yaml')
            
            config_manager = ConfigManager(config_path)
            
            # Initialize LLM client and QueryRewriter
            llm_client = LLMClient(config_manager.get_llm_config())
            query_rewriter = QueryRewriter(llm_client, config_manager.get('query_rewriter', {}))
            
            # Test connection to ensure LLM is working
            if not query_rewriter.test_connection():
                pytest.skip("LLM connection failed - skipping integration test")
            
            return query_rewriter
            
        except Exception as e:
            pytest.skip(f"QueryRewriter setup failed: {e}")

    def test_intent_based_filter_detection_and_query_cleaning(self, query_rewriter):
        """
        Test that intent-based filter detection and LLM query cleaning work correctly.
        
        This test validates both requirements from GitHub issue #20:
        1. Hard filter detection works with lowercase "only" (intent-based)
        2. LLM query is properly cleaned of search instructions and filters
        """
        # The problematic query from the GitHub issue
        test_query = "search @knowledgebase on what is vibe coding and its pros and cons, only authored by John Wong"
        
        result = query_rewriter.transform_query(test_query)
        
        # Validate the result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'hard_filters' in result, "Result should contain hard_filters"
        assert 'soft_filters' in result, "Result should contain soft_filters"
        assert 'negation_filters' in result, "Result should contain negation_filters"
        assert 'llm_query' in result, "Result should contain llm_query"
        assert 'source' in result, "Result should contain source field"
        
        # Requirement 1: Hard filter detection with lowercase "only"
        assert 'author' in result['hard_filters'], "Should detect 'only authored by John Wong' as hard filter"
        assert result['hard_filters']['author'] == 'John Wong', "Hard filter should contain correct author"
        assert 'author' not in result['soft_filters'], "Author should NOT be in soft filters when detected as hard filter"
        
        # Requirement 2: LLM query cleaning
        llm_query = result['llm_query']
        
        # Verify original search instructions are removed (but template text like "search terms" is OK)
        original_search_parts = ['search @knowledgebase on', 'find @knowledgebase', '@knowledgebase']
        for term in original_search_parts:
            assert term not in llm_query.lower(), f"LLM query should not contain original search instruction: {term}"
        
        # Verify filter clauses are removed
        filter_terms = ['only authored by', 'John Wong', 'authored by John Wong']
        for term in filter_terms:
            assert term not in llm_query, f"LLM query should not contain filter clause: {term}"
        
        # This query is classified as Pattern 1 (Pure search) by the LLM, so it should use SEARCH_SUMMARY_MODE
        # which gets converted to the document summary instructions
        expected_in_summary_mode = ["Document Summary", "Key Topics", "Question Suggestions"]
        if any(phrase in llm_query for phrase in expected_in_summary_mode):
            # Pattern 1: Pure search → SEARCH_SUMMARY_MODE
            print("✅ Query correctly classified as Pattern 1 (Pure search) → SEARCH_SUMMARY_MODE")
        else:
            # Pattern 2: Search + action → clean question
            assert llm_query.startswith("Based on the provided context"), "Pattern 2 queries should start with proper context instruction"
            assert 'vibe coding' in llm_query, "LLM query should contain the core topic 'vibe coding'"
            assert 'pros and cons' in llm_query, "LLM query should contain the specific question 'pros and cons'"
            print("✅ Query correctly classified as Pattern 2 (Search + action) → clean question")
        
        # Verify source is from LLM (not fallback)
        assert result['source'] == 'llm', "Result should come from LLM, not fallback"
        
        print(f"✅ Test passed - Query: {test_query}")
        print(f"✅ Hard filter detected: {result['hard_filters']}")
        print(f"✅ Clean LLM query: {llm_query}")

    def test_intent_based_negation_detection(self, query_rewriter):
        """
        Test that intent-based negation filter detection works correctly.
        """
        test_query = "@knowledgebase explain machine learning algorithms, not by Johnson"
        
        result = query_rewriter.transform_query(test_query)
        
        # Validate negation filter detection
        assert 'author' in result['negation_filters'], "Should detect 'not by Johnson' as negation filter"
        assert result['negation_filters']['author'] == 'Johnson', "Negation filter should contain correct author"
        assert 'author' not in result['soft_filters'], "Author should NOT be in soft filters when detected as negation"
        assert 'author' not in result['hard_filters'], "Author should NOT be in hard filters when detected as negation"
        
        # Verify LLM query cleaning for negation case
        llm_query = result['llm_query']
        assert 'machine learning algorithms' in llm_query, "LLM query should contain core topic"
        assert 'not by Johnson' not in llm_query, "LLM query should not contain negation filter clause"
        
        print(f"✅ Negation test passed - Query: {test_query}")
        print(f"✅ Negation filter detected: {result['negation_filters']}")
        print(f"✅ Clean LLM query: {llm_query}")

    def test_soft_filter_default_behavior(self, query_rewriter):
        """
        Test that mentions without restrictive/exclusion intent go to soft filters.
        """
        test_query = "@knowledgebase papers by Smith from 2024"
        
        result = query_rewriter.transform_query(test_query)
        
        # Should be soft filters (default behavior)
        assert 'author' in result['soft_filters'], "Should detect 'by Smith' as soft filter"
        assert result['soft_filters']['author'] == 'Smith', "Soft filter should contain correct author"
        assert 'author' not in result['hard_filters'], "Author should NOT be in hard filters without restrictive intent"
        assert 'author' not in result['negation_filters'], "Author should NOT be in negation filters"
        
        print(f"✅ Soft filter test passed - Query: {test_query}")
        print(f"✅ Soft filters detected: {result['soft_filters']}")

    def test_pattern_2_query_cleaning(self, query_rewriter):
        """
        Test Pattern 2 (Search + action) query cleaning behavior.
        
        This test demonstrates that query cleaning works for explicit action queries.
        """
        test_query = "search @knowledgebase on vibe coding, only authored by John Wong, and explain what it is and its pros and cons"
        
        result = query_rewriter.transform_query(test_query)
        
        # Should detect hard filter
        assert 'author' in result['hard_filters'], "Should detect hard filter"
        assert result['hard_filters']['author'] == 'John Wong', "Hard filter should contain correct author"
        
        # Should classify as Pattern 2 and clean the query
        llm_query = result['llm_query']
        
        # Verify original search instructions are removed
        original_search_parts = ['search @knowledgebase on', 'find @knowledgebase', '@knowledgebase']
        for term in original_search_parts:
            assert term not in llm_query.lower(), f"LLM query should not contain original search instruction: {term}"
        
        # Verify filter clauses are removed
        filter_terms = ['only authored by', 'John Wong']
        for term in filter_terms:
            assert term not in llm_query, f"LLM query should not contain filter clause: {term}"
        
        # Should be Pattern 2: clean action query
        assert llm_query.startswith("Based on the provided context"), "Pattern 2 queries should start with proper context instruction"
        assert 'vibe coding' in llm_query, "LLM query should contain the core topic 'vibe coding'"
        assert 'pros and cons' in llm_query, "LLM query should contain the specific question 'pros and cons'"
        
        print(f"✅ Pattern 2 test passed - Query: {test_query}")
        print(f"✅ Clean LLM query: {llm_query}")