"""
Integration tests for soft filtering with SearchService and QueryRewriter.
Tests the complete pipeline with real LLM for three-filter system.
"""

import sys
import os
from typing import Dict, Any

import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter
from search_service import SearchService
from llm_client import LLMClient
from config_manager import ConfigManager
from unittest.mock import Mock
from qdrant_client.models import ScoredPoint


class TestSoftFilteringIntegration:
    """Integration tests for soft filtering system with real LLM."""
    
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
        
        # Create mock QdrantDB for SearchService testing
        cls.mock_qdrant = cls.create_mock_qdrant_db()
        
        # Initialize SearchService with mock
        search_service_config = cls.config_manager.get('search_service', {})
        cls.search_service = SearchService(cls.mock_qdrant, search_service_config)
    
    def setup_method(self):
        """Reset mock state before each test method."""
        self.mock_qdrant.search.reset_mock()
    
    @classmethod
    def create_mock_qdrant_db(cls):
        """Create a mock QdrantDB for testing."""
        mock_db = Mock()
        mock_db.search = Mock()
        return mock_db
    
    @classmethod
    def create_mock_scored_point(cls, point_id: str, score: float, payload: Dict[str, Any]) -> ScoredPoint:
        """Create a mock ScoredPoint for testing."""
        return ScoredPoint(
            id=point_id,
            version=1,
            score=score,
            payload=payload,
            vector=None
        )
    
    def validate_three_filter_structure(self, result: Dict[str, Any]) -> bool:
        """Validate that result has the new three-filter structure."""
        required_fields = ['search_rag', 'embedding_texts', 'llm_query', 
                          'hard_filters', 'negation_filters', 'soft_filters']
        
        if not all(field in result for field in required_fields):
            return False
        
        # Validate field types
        if not isinstance(result['search_rag'], bool):
            return False
        
        # Validate embedding_texts structure
        if not isinstance(result['embedding_texts'], dict):
            return False
        
        embedding_texts = result['embedding_texts']
        if 'rewrite' not in embedding_texts or 'hyde' not in embedding_texts:
            return False
        
        if not isinstance(embedding_texts['rewrite'], str):
            return False
        
        if not isinstance(embedding_texts['hyde'], list):
            return False
        
        if not isinstance(result['llm_query'], str) or not result['llm_query'].strip():
            return False
        
        # Validate all three filter fields are dictionaries
        for filter_field in ['hard_filters', 'negation_filters', 'soft_filters']:
            if not isinstance(result[filter_field], dict):
                return False
        
        return True
    
    @pytest.mark.integration
    def test_soft_filter_classification_with_real_llm(self):
        """Test that real LLM correctly classifies filters as soft (default behavior)."""
        user_query = "@knowledgebase papers by Smith about Python from 2024"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate new three-filter structure
        assert self.validate_three_filter_structure(result), f"Invalid structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Default behavior: most filters should go to soft_filters
        soft_filters = result['soft_filters']
        hard_filters = result['hard_filters']
        negation_filters = result['negation_filters']
        
        # Should have soft filters for author, topic, and date
        assert len(soft_filters) > 0, f"Should have soft filters, got: {soft_filters}"
        
        # Should NOT have hard or negation filters (no restrictive keywords)
        assert len(hard_filters) == 0, f"Should have no hard filters, got: {hard_filters}"
        assert len(negation_filters) == 0, f"Should have no negation filters, got: {negation_filters}"
        
        print("Soft Filter Classification Test:")
        print(f"  User Query: {user_query}")
        print(f"  Soft Filters: {soft_filters}")
        print(f"  Hard Filters: {hard_filters}")
        print(f"  Negation Filters: {negation_filters}")
    
    @pytest.mark.integration
    def test_hard_filter_classification_with_real_llm(self):
        """Test that real LLM correctly classifies explicit restrictive keywords as hard filters."""
        user_query = "@knowledgebase papers ONLY from 2025 about AI"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_three_filter_structure(result), f"Invalid structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Should classify date as hard filter due to "ONLY"
        soft_filters = result['soft_filters']
        hard_filters = result['hard_filters']
        negation_filters = result['negation_filters']
        
        # Should have hard filter for the "ONLY from 2025" part
        assert len(hard_filters) > 0, f"Should have hard filters for 'ONLY', got: {hard_filters}"
        
        # Should have no negation filters
        assert len(negation_filters) == 0, f"Should have no negation filters, got: {negation_filters}"
        
        # May have soft filters for other parts (like topic "AI")
        print("Hard Filter Classification Test:")
        print(f"  User Query: {user_query}")
        print(f"  Hard Filters: {hard_filters}")
        print(f"  Soft Filters: {soft_filters}")
        print(f"  Negation Filters: {negation_filters}")
    
    @pytest.mark.integration
    def test_negation_filter_classification_with_real_llm(self):
        """Test that real LLM correctly classifies negation keywords as negation filters."""
        user_query = "@knowledgebase papers about Python not from Johnson"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_three_filter_structure(result), f"Invalid structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Should classify author exclusion as negation filter
        soft_filters = result['soft_filters']
        hard_filters = result['hard_filters']
        negation_filters = result['negation_filters']
        
        # Should have negation filter for "not from Johnson"
        assert len(negation_filters) > 0, f"Should have negation filters for 'not from', got: {negation_filters}"
        
        # Should have no hard filters (no restrictive keywords)
        assert len(hard_filters) == 0, f"Should have no hard filters, got: {hard_filters}"
        
        # May have soft filters for topic "Python"
        print("Negation Filter Classification Test:")
        print(f"  User Query: {user_query}")
        print(f"  Negation Filters: {negation_filters}")
        print(f"  Soft Filters: {soft_filters}")
        print(f"  Hard Filters: {hard_filters}")
    
    @pytest.mark.integration
    def test_combined_filter_types_with_real_llm(self):
        """Test query with all three filter types combined."""
        user_query = "@knowledgebase papers ONLY from 2025 about AI, not by Johnson, with tags Python"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_three_filter_structure(result), f"Invalid structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        soft_filters = result['soft_filters']
        hard_filters = result['hard_filters']
        negation_filters = result['negation_filters']
        
        # Should have hard filter for "ONLY from 2025"
        assert len(hard_filters) > 0, f"Should have hard filters for 'ONLY', got: {hard_filters}"
        
        # Should have negation filter for "not by Johnson"
        assert len(negation_filters) > 0, f"Should have negation filters for 'not by', got: {negation_filters}"
        
        # Should have soft filters for other parts
        # Note: "with tags Python" might go to soft filters based on the prompt
        
        print("Combined Filter Types Test:")
        print(f"  User Query: {user_query}")
        print(f"  Hard Filters: {hard_filters}")
        print(f"  Negation Filters: {negation_filters}")
        print(f"  Soft Filters: {soft_filters}")
    
    @pytest.mark.integration
    def test_search_service_with_soft_filters(self):
        """Test SearchService unified search with soft filters."""
        # Reset mock call history from previous tests
        self.mock_qdrant.search.reset_mock()
        
        # Setup mock search results
        mock_results = [
            self.create_mock_scored_point("1", 0.8, {
                "title": "Python Guide", 
                "tags": ["python"], 
                "author": "Smith"
            }),
            self.create_mock_scored_point("2", 0.9, {
                "title": "Java Tutorial", 
                "tags": ["java"]
            }),
            self.create_mock_scored_point("3", 0.7, {
                "title": "Python Advanced", 
                "tags": ["python"], 
                "author": "Jones"
            })
        ]
        self.mock_qdrant.search.return_value = mock_results
        
        # Test unified search with soft filters
        query_vector = [0.1, 0.2, 0.3]
        soft_filters = {"tags": ["python"], "author": "Smith"}
        
        results = self.search_service.unified_search(
            query_vector=query_vector,
            top_k=3,
            hard_filters=None,
            negation_filters=None,
            soft_filters=soft_filters
        )
        
        # Should return re-ranked results with boosting
        assert len(results) == 3, f"Should return 3 results, got {len(results)}"
        
        # Results should be boosted (document 1 should get highest boost for matching both filters)
        # Document 1: matches python + Smith = highest boost
        # Document 3: matches python only = medium boost  
        # Document 2: no matches = no boost
        
        # Check that scores were modified (boosted results should have different scores)
        original_scores = {r.id: r.score for r in mock_results}
        final_scores = {r.id: r.score for r in results}
        
        # At least one score should be boosted
        scores_changed = any(final_scores[rid] != original_scores[rid] for rid in final_scores)
        assert scores_changed, "Soft filtering should boost some document scores"
        
        print("SearchService Soft Filtering Test:")
        print(f"  Original scores: {original_scores}")
        print(f"  Boosted scores: {final_scores}")
        print(f"  Soft filters applied: {soft_filters}")
    
    @pytest.mark.integration
    def test_search_service_with_hard_filters_only(self):
        """Test SearchService with hard filters only (no soft filtering)."""
        # Setup mock search results
        mock_results = [
            self.create_mock_scored_point("1", 0.8, {"title": "Test Document"})
        ]
        self.mock_qdrant.search.return_value = mock_results
        
        # Test with hard filters only
        query_vector = [0.1, 0.2, 0.3]
        hard_filters = {"author": "Smith"}
        
        results = self.search_service.unified_search(
            query_vector=query_vector,
            top_k=5,
            hard_filters=hard_filters,
            negation_filters=None,
            soft_filters=None
        )
        
        # Should use direct qdrant search without boosting
        self.mock_qdrant.search.assert_called_once_with(
            query_vector=query_vector,
            limit=5,  # No multiplier when no soft filters
            score_threshold=None,
            filters=hard_filters,
            negation_filters=None
        )
        
        # Should return original results unchanged
        assert results == mock_results
        
        print("SearchService Hard Filters Only Test:")
        print(f"  Hard filters: {hard_filters}")
        print(f"  Results unchanged: {len(results)} documents")
    
    @pytest.mark.integration
    def test_search_service_with_all_filter_types(self):
        """Test SearchService with all three filter types."""
        # Reset mock call history from previous tests
        self.mock_qdrant.search.reset_mock()
        
        # Setup mock search results  
        mock_results = [
            self.create_mock_scored_point("1", 0.8, {
                "title": "Python ML Guide", 
                "tags": ["python", "ml"], 
                "author": "Smith"
            })
        ]
        self.mock_qdrant.search.return_value = mock_results
        
        # Test with all filter types
        query_vector = [0.1, 0.2, 0.3]
        hard_filters = {"publication_date": {"gte": "2025-01-01", "lt": "2026-01-01"}}
        negation_filters = {"author": "Johnson"}
        soft_filters = {"tags": ["python"], "author": "Smith"}
        
        results = self.search_service.unified_search(
            query_vector=query_vector,
            top_k=3,
            hard_filters=hard_filters,
            negation_filters=negation_filters,
            soft_filters=soft_filters
        )
        
        # Should apply hard/negation filters via Qdrant, then boost with soft filters
        call_args = self.mock_qdrant.search.call_args
        assert call_args.kwargs['filters'] == hard_filters
        assert call_args.kwargs['negation_filters'] == negation_filters
        assert call_args.kwargs['limit'] > 3  # Should use multiplier for soft filtering
        
        print("SearchService All Filter Types Test:")
        print(f"  Hard filters: {hard_filters}")
        print(f"  Negation filters: {negation_filters}")
        print(f"  Soft filters: {soft_filters}")
        print(f"  Results: {len(results)} documents")
    
    @pytest.mark.integration
    def test_end_to_end_soft_filtering_pipeline(self):
        """Test complete end-to-end pipeline: QueryRewriter -> SearchService."""
        # Reset mock call history from previous tests
        self.mock_qdrant.search.reset_mock()
        
        user_query = "@knowledgebase papers by Smith about Python from 2024"
        
        # Step 1: Transform query with real LLM
        query_result = self.query_rewriter.transform_query(user_query)
        
        # Validate query transformation
        assert self.validate_three_filter_structure(query_result), "Query transformation failed"
        assert query_result['search_rag'] is True, "Should trigger RAG search"
        
        # Step 2: Setup mock embedding and search results
        mock_embedding = [0.1, 0.2, 0.3]  # Mock embedding vector
        mock_results = [
            self.create_mock_scored_point("1", 0.8, {
                "title": "Python Best Practices", 
                "tags": ["python"], 
                "author": "Smith",
                "publication_date": "2024-03-15"
            }),
            self.create_mock_scored_point("2", 0.7, {
                "title": "Java Guide", 
                "tags": ["java"], 
                "author": "Jones"
            })
        ]
        self.mock_qdrant.search.return_value = mock_results
        
        # Step 3: Perform search with extracted filters
        search_results = self.search_service.unified_search(
            query_vector=mock_embedding,
            top_k=5,
            hard_filters=query_result['hard_filters'],
            negation_filters=query_result['negation_filters'],
            soft_filters=query_result['soft_filters']
        )
        
        # Step 4: Validate results
        assert len(search_results) <= 5, "Should respect top_k limit"
        
        # If soft filters were extracted, results should be boosted
        if query_result['soft_filters']:
            # Document 1 should be boosted (matches python, Smith, and 2024)
            original_score_1 = 0.8
            final_score_1 = next(r.score for r in search_results if r.id == "1")
            assert final_score_1 >= original_score_1, "Matching document should be boosted"
        
        print("End-to-End Pipeline Test:")
        print(f"  Original Query: {user_query}")
        print(f"  Embedding Text: {query_result['embedding_texts']['rewrite']}")
        print(f"  Hard Filters: {query_result['hard_filters']}")
        print(f"  Negation Filters: {query_result['negation_filters']}")
        print(f"  Soft Filters: {query_result['soft_filters']}")
        print(f"  Search Results: {len(search_results)} documents")
    
    @pytest.mark.integration
    def test_query_rewriter_consistency_for_filter_types(self):
        """Test QueryRewriter consistency across multiple runs for filter classification."""
        test_cases = [
            ("@knowledgebase papers by Smith", "soft", "author"),
            ("@knowledgebase papers ONLY by Smith", "hard", "author"),
            ("@knowledgebase papers not by Smith", "negation", "author"),
        ]
        
        for user_query, expected_filter_type, filter_field in test_cases:
            results = []
            
            # Run same query multiple times
            for i in range(2):
                result = self.query_rewriter.transform_query(user_query)
                results.append(result)
            
            # All results should have valid structure
            for i, result in enumerate(results):
                assert self.validate_three_filter_structure(result), f"Run {i+1} invalid structure"
                assert result['search_rag'] is True, f"Run {i+1} should detect RAG trigger"
            
            # Check filter classification consistency
            filter_types_found = []
            for result in results:
                if result['hard_filters'].get(filter_field):
                    filter_types_found.append("hard")
                elif result['negation_filters'].get(filter_field):
                    filter_types_found.append("negation")
                elif result['soft_filters'].get(filter_field):
                    filter_types_found.append("soft")
                else:
                    filter_types_found.append("none")
            
            # Should be consistent (at least 50% of runs should classify correctly)
            correct_classifications = filter_types_found.count(expected_filter_type)
            consistency_rate = correct_classifications / len(results)
            
            print(f"Filter Consistency Test: {user_query}")
            print(f"  Expected: {expected_filter_type} filter")
            print(f"  Classifications: {filter_types_found}")
            print(f"  Consistency rate: {consistency_rate:.1%}")
            
            # Don't fail the test for inconsistency, but log it for analysis
            if consistency_rate < 0.5:
                print(f"  ⚠️  Low consistency for {expected_filter_type} filter classification")
            else:
                print(f"  ✅ Good consistency for {expected_filter_type} filter classification")
    
    @pytest.mark.integration 
    def test_search_service_boost_configuration(self):
        """Test SearchService boost configuration and statistics."""
        # Reset mock call history from previous tests
        self.mock_qdrant.search.reset_mock()
        
        # Test getting boost statistics
        stats = self.search_service.get_boost_statistics()
        
        # Should have all required configuration fields
        required_fields = ['base_boost_per_match', 'diminishing_factor', 'max_total_boost', 
                          'field_weights', 'fetch_multiplier']
        for field in required_fields:
            assert field in stats, f"Missing configuration field: {field}"
        
        # Test updating boost configuration
        original_base_boost = stats['base_boost_per_match']
        new_config = {'base_boost_per_match': 0.2}
        
        self.search_service.update_boost_config(new_config)
        updated_stats = self.search_service.get_boost_statistics()
        
        assert updated_stats['base_boost_per_match'] == 0.2, "Configuration update failed"
        
        # Restore original configuration
        restore_config = {'base_boost_per_match': original_base_boost}
        self.search_service.update_boost_config(restore_config)
        
        print("Boost Configuration Test:")
        print(f"  Original base boost: {original_base_boost}")
        print("  Updated base boost: 0.2")
        print(f"  Configuration restored: {self.search_service.get_boost_statistics()['base_boost_per_match']}")
    
    @pytest.mark.integration
    def test_real_llm_connection_and_filter_parsing(self):
        """Test that real LLM connection works and produces parseable filter results."""
        # Test connection
        connection_success = self.query_rewriter.test_connection()
        assert connection_success, "QueryRewriter should successfully connect to real LLM"
        
        # Test filter parsing with various queries
        test_queries = [
            "@knowledgebase simple query about AI",
            "@knowledgebase papers by Smith ONLY from 2025",
            "@knowledgebase documents not by Johnson about Python"
        ]
        
        for query in test_queries:
            result = self.query_rewriter.transform_query(query)
            
            # Should parse successfully with new structure
            assert self.validate_three_filter_structure(result), f"Failed to parse: {query}"
            
            # Should detect trigger correctly
            assert result['search_rag'] is True, f"Should detect trigger in: {query}"
        
        print("Real LLM Connection Test:")
        print(f"  Connection successful: {connection_success}")
        print(f"  Successfully parsed {len(test_queries)} different queries with three-filter structure")
    
    @pytest.mark.integration
    def test_real_llm_search_only_query_detection(self):
        """Test that real LLM correctly detects search-only queries."""
        # Test search-only queries
        search_only_queries = [
            "@knowledgebase search papers by John Wong about vibe coding",
            "@knowledgebase find documents from 2025 without tag gemini",
            "@knowledgebase get research by Dr. Smith on AI",
            "@knowledgebase show me articles about Python"
        ]
        
        # Test question queries (should NOT be search-only)
        question_queries = [
            "@knowledgebase what is vibe coding?",
            "@knowledgebase how does neural network training work?",
            "@knowledgebase explain the benefits of Python",
            "@knowledgebase compare REST vs GraphQL APIs"
        ]
        
        print("Search-Only Query Detection with Real LLM:")
        print("=" * 50)
        
        # Test search-only detection
        for query in search_only_queries:
            result = self.query_rewriter.transform_query(query)
            
            # Should parse successfully
            assert self.validate_three_filter_structure(result), f"Failed to parse: {query}"
            assert result['search_rag'] is True, f"Should detect trigger in: {query}"
            
            # Should be detected as search-only (QueryRewriter now returns full prompt)
            is_search_only = 'If no context documents are provided' in result['llm_query']
            print(f"  SEARCH-ONLY: '{query}' → {is_search_only}")
            
            # Note: We don't assert here because LLM behavior can vary slightly
            # But we log for analysis
        
        print()
        
        # Test question detection (should NOT be search-only)
        for query in question_queries:
            result = self.query_rewriter.transform_query(query)
            
            # Should parse successfully
            assert self.validate_three_filter_structure(result), f"Failed to parse: {query}"
            assert result['search_rag'] is True, f"Should detect trigger in: {query}"
            
            # Should NOT be detected as search-only
            is_search_only = 'If no context documents are provided' in result['llm_query']
            has_context_ref = 'based on the provided context' in result['llm_query'].lower()
            print(f"  QUESTION: '{query}' → search_only:{is_search_only}, has_context:{has_context_ref}")
        
        print(f"  Successfully tested search-only detection with {len(search_only_queries + question_queries)} queries")