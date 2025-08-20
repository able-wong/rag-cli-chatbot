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

    def test_intent_based_hard_filter_detection_pattern1(self, query_rewriter):
        """
        Test hard filter detection with deterministic Pattern 1 (Pure Search) query.
        
        Validates GitHub issue #20 requirement 1: Hard filter detection works with lowercase "only"
        Uses unambiguous search query that consistently classifies as Pattern 1.
        """
        # Unambiguous Pattern 1 query - pure search intent, no specific action
        test_query = "search @knowledgebase for papers only authored by John Wong"
        
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
        
        # Pattern 1: Should use SEARCH_SUMMARY_MODE (deterministic)
        llm_query = result['llm_query']
        expected_in_summary_mode = ["Document Summary", "Key Topics", "Question Suggestions"]
        assert any(phrase in llm_query for phrase in expected_in_summary_mode), "Pattern 1 should use SEARCH_SUMMARY_MODE"
        
        # Verify search instructions are removed from the LLM query
        original_search_parts = ['search @knowledgebase for', '@knowledgebase']
        for term in original_search_parts:
            assert term not in llm_query.lower(), f"LLM query should not contain original search instruction: {term}"
        
        # Verify filter clauses are removed
        filter_clauses = ['only authored by', 'only authored by John Wong']
        for term in filter_clauses:
            assert term not in llm_query, f"LLM query should not contain filter clause: {term}"
        
        # Verify source is from LLM (not fallback)
        assert result['source'] == 'llm', "Result should come from LLM, not fallback"
        
        print(f"✅ Pattern 1 test passed - Query: {test_query}")
        print(f"✅ Hard filter detected: {result['hard_filters']}")
        print("✅ SEARCH_SUMMARY_MODE used correctly")

    def test_intent_based_hard_filter_detection_pattern2(self, query_rewriter):
        """
        Test hard filter detection with deterministic Pattern 2 (Search + Action) query.
        
        Validates both GitHub issue #20 requirements:
        1. Hard filter detection works with lowercase "only" 
        2. LLM query is properly cleaned of search instructions and filters
        Uses unambiguous action query that consistently classifies as Pattern 2.
        """
        # Unambiguous Pattern 2 query - search + explicit action
        test_query = "search @knowledgebase for papers only authored by John Wong and explain their main contributions"
        
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
        
        # Pattern 2: Should use clean action query format (deterministic)
        llm_query = result['llm_query']
        assert llm_query.startswith("Based on the provided context"), "Pattern 2 queries should start with proper context instruction"
        assert 'contributions' in llm_query, "LLM query should contain the specific action 'contributions'"
        
        # Requirement 2: LLM query cleaning - verify search instructions are removed
        original_search_parts = ['search @knowledgebase for', '@knowledgebase']
        for term in original_search_parts:
            assert term not in llm_query.lower(), f"LLM query should not contain original search instruction: {term}"
        
        # Verify filter clauses are removed (but author name in action context is OK)
        filter_clauses = ['only authored by', 'only authored by John Wong']
        for term in filter_clauses:
            assert term not in llm_query, f"LLM query should not contain filter clause: {term}"
        
        # Verify source is from LLM (not fallback)
        assert result['source'] == 'llm', "Result should come from LLM, not fallback"
        
        print(f"✅ Pattern 2 test passed - Query: {test_query}")
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

    def test_filter_information_removal_from_embeddings(self, query_rewriter):
        """
        Test that filter information (author, dates) is properly removed from 
        rewrite keywords and HyDE content, while being extracted to filters.
        
        Validates that:
        1. rewrite field contains only core topic keywords 
        2. HyDE content focuses on topic without filter information
        3. Filter information is properly extracted to soft_filters
        """
        test_query = "@knowledgebase vibe coding articles by John Wong from 2025"
        
        result = query_rewriter.transform_query(test_query)
        
        # Validate basic structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        assert 'embedding_texts' in result, "Should contain embedding_texts"
        assert 'soft_filters' in result, "Should contain soft_filters"
        
        # Test 1: rewrite should only contain core topic keywords
        rewrite_text = result['embedding_texts']['rewrite']
        assert isinstance(rewrite_text, str), "Rewrite should be a string"
        
        # Should contain topic keywords
        assert 'vibe coding' in rewrite_text.lower(), "Rewrite should contain core topic 'vibe coding'"
        assert 'articles' in rewrite_text.lower(), "Rewrite should contain content type 'articles'"
        
        # Should NOT contain filter information
        assert 'john wong' not in rewrite_text.lower(), "Rewrite should not contain author name"
        assert '2025' not in rewrite_text, "Rewrite should not contain publication date"
        assert 'john' not in rewrite_text.lower(), "Rewrite should not contain author first name"
        assert 'wong' not in rewrite_text.lower(), "Rewrite should not contain author last name"
        
        # Test 2: HyDE should focus on topic only, not include filter information
        hyde_texts = result['embedding_texts']['hyde']
        assert isinstance(hyde_texts, list), "HyDE should be a list"
        assert len(hyde_texts) >= 1, "Should have at least one HyDE text"
        
        # Check each HyDE text
        for i, hyde_text in enumerate(hyde_texts):
            assert isinstance(hyde_text, str), f"HyDE text {i} should be a string"
            assert len(hyde_text.strip()) > 0, f"HyDE text {i} should not be empty"
            
            # Should focus on topic
            assert 'vibe coding' in hyde_text.lower() or 'coding' in hyde_text.lower(), f"HyDE text {i} should discuss the topic"
            
            # Should NOT contain specific filter information
            assert 'john wong' not in hyde_text.lower(), f"HyDE text {i} should not mention specific author"
            assert '2025' not in hyde_text, f"HyDE text {i} should not mention specific year"
            
            # Should not be placeholder text
            assert not hyde_text.startswith('[Replace with'), f"HyDE text {i} should not be placeholder text"
        
        # Test 3: Filter information should be properly extracted
        soft_filters = result['soft_filters']
        assert 'author' in soft_filters, "Should extract author to soft_filters"
        assert soft_filters['author'] == 'John Wong', "Should correctly extract author name"
        assert 'publication_date' in soft_filters, "Should extract publication date to soft_filters"
        
        pub_date = soft_filters['publication_date']
        assert isinstance(pub_date, dict), "Publication date should be a range object"
        assert 'gte' in pub_date, "Should have gte (greater than or equal) date"
        assert 'lt' in pub_date, "Should have lt (less than) date"
        assert '2025' in pub_date['gte'], "Should extract 2025 as start year"
        assert '2026' in pub_date['lt'], "Should set 2026 as end year"
        
        print(f"✅ Filter removal test passed - Query: {test_query}")
        print(f"  Rewrite (clean): '{rewrite_text}'")
        print(f"  HyDE texts: {len(hyde_texts)} generated")
        print(f"  Soft filters: {soft_filters}")
        print("  All filter information properly separated from embeddings")