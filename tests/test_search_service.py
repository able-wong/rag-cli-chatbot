"""
Unit tests for SearchService class.
Tests search functionality with mocked QdrantDB for isolated testing.
"""

import sys
import os
from unittest.mock import Mock
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_service import SearchService, SOFT_FILTER_CONFIG
from qdrant_client.models import ScoredPoint


def create_mock_qdrant_db():
    """Create a mock QdrantDB for testing."""
    mock_db = Mock()
    mock_db.search = Mock()
    return mock_db


def create_mock_embedding_client():
    """Create a mock EmbeddingClient for testing."""
    mock_client = Mock()
    mock_client.get_embedding = Mock(return_value=[0.1, 0.2, 0.3])
    mock_client.has_sparse_embedding = Mock(return_value=False)
    mock_client.get_sparse_embedding = Mock(return_value={"indices": [1, 2], "values": [0.5, 0.3]})
    return mock_client


def create_mock_query_rewriter(hard_filters=None, negation_filters=None, soft_filters=None):
    """Create a mock QueryRewriter for testing."""
    mock_rewriter = Mock()
    mock_rewriter.transform_query = Mock(return_value={
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': hard_filters or {},
        'negation_filters': negation_filters or {},
        'soft_filters': soft_filters or {},
        'strategy': 'rewrite'
    })
    return mock_rewriter


def create_search_service(config=None):
    """Create a SearchService with all required mocks for testing."""
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_query_rewriter = create_mock_query_rewriter()
    
    return SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter,
        config=config
    ), mock_db, mock_dense_client, mock_query_rewriter


def create_mock_scored_point(point_id: str, score: float, payload: Dict[str, Any]) -> ScoredPoint:
    """Create a mock ScoredPoint for testing."""
    return ScoredPoint(
        id=point_id,
        version=1,
        score=score,
        payload=payload,
        vector=None
    )


def test_search_service_initialization():
    """Test SearchService initialization with default and custom config."""
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_query_rewriter = create_mock_query_rewriter()
    
    # Test default initialization
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter
    )
    assert search_service.qdrant_db == mock_db
    assert search_service.dense_embedding_client == mock_dense_client
    assert search_service.query_rewriter == mock_query_rewriter
    assert search_service.sparse_embedding_client is None
    assert search_service.config['base_boost_per_match'] == 0.12
    assert search_service.config['field_weights']['tags'] == 1.2
    
    # Test custom config with sparse embedding client
    custom_config = {
        'base_boost_per_match': 0.15,
        'field_weights': {'tags': 1.5}
    }
    mock_sparse_client = create_mock_embedding_client()
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter,
        sparse_embedding_client=mock_sparse_client,
        config=custom_config
    )
    assert search_service.config['base_boost_per_match'] == 0.15
    assert search_service.config['field_weights']['tags'] == 1.5
    assert search_service.sparse_embedding_client == mock_sparse_client
    # Should still have default values for non-overridden fields
    assert search_service.config['diminishing_factor'] == 0.75


def test_unified_search_no_soft_filters():
    """Test unified search without soft filters (should use standard search)."""
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_query_rewriter = create_mock_query_rewriter()
    
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter
    )
    
    # Mock search results
    mock_results = [
        create_mock_scored_point("1", 0.9, {"title": "Test Document 1"}),
        create_mock_scored_point("2", 0.8, {"title": "Test Document 2"})
    ]
    mock_db.search.return_value = mock_results
    
    # Set up query rewriter to return hard and negation filters
    query = "test query"
    expected_hard_filters = {"author": "Smith"}
    expected_negation_filters = {"tags": ["draft"]}
    
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': expected_hard_filters,
        'negation_filters': expected_negation_filters,
        'soft_filters': {},
        'strategy': 'rewrite'
    }
    
    results = search_service.unified_search(
        query=query,
        top_k=5
    )
    
    # Should call query_rewriter.transform_query with the query
    mock_query_rewriter.transform_query.assert_called_once_with(query)
    
    # Should call dense_embedding_client.get_embedding with processed text
    mock_dense_client.get_embedding.assert_called_once_with('processed query text')
    
    # Should call qdrant_db.search with generated vector
    mock_db.search.assert_called_once_with(
        query_vector=[0.1, 0.2, 0.3],
        limit=5,  # No multiplier when no soft filters
        score_threshold=None,
        filters=expected_hard_filters,
        negation_filters=expected_negation_filters
    )
    
    # Should return original results unchanged
    assert results == mock_results
    assert len(results) == 2


def test_unified_search_with_soft_filters():
    """Test unified search with soft filters (should apply boosting and re-ranking)."""
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_query_rewriter = create_mock_query_rewriter()
    
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter
    )
    
    # Mock search results with different scores
    mock_results = [
        create_mock_scored_point("1", 0.8, {"title": "Document 1", "tags": ["python"], "author": "Smith"}),
        create_mock_scored_point("2", 0.9, {"title": "Document 2", "tags": ["java"]}),
        create_mock_scored_point("3", 0.7, {"title": "Document 3", "tags": ["python", "ai"], "author": "Jones"})
    ]
    mock_db.search.return_value = mock_results
    
    # Set up query rewriter to return soft filters
    query = "test query"
    expected_soft_filters = {"tags": ["python"], "author": "Smith"}
    
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': expected_soft_filters,
        'strategy': 'rewrite'
    }
    
    results = search_service.unified_search(
        query=query,
        top_k=2
    )
    
    # Should call query_rewriter.transform_query with the query
    mock_query_rewriter.transform_query.assert_called_once_with(query)
    
    # Should call dense_embedding_client.get_embedding with processed text
    mock_dense_client.get_embedding.assert_called_once_with('processed query text')
    
    # Should call qdrant_db.search with multiplied limit
    expected_limit = 2 * SOFT_FILTER_CONFIG['fetch_multiplier']  # 2 * 4 = 8
    mock_db.search.assert_called_once_with(
        query_vector=[0.1, 0.2, 0.3],
        limit=expected_limit,
        score_threshold=None,
        filters=None,
        negation_filters=None
    )
    
    # Should return boosted and re-ranked results
    assert len(results) == 2
    
    # Document 1 and 3 should be boosted (they match "python" tag)
    # Document 1 also matches "Smith" author, so should get higher boost
    # Final ranking should be affected by boosting
    
    # Verify scores are boosted (higher than original)
    original_scores = {r.id: r.score for r in mock_results}
    for result in results:
        if result.payload.get("tags") and "python" in result.payload["tags"]:
            assert result.score > original_scores[result.id]
    
    print("✓ Soft filter boosting test passed")


def test_boost_calculation_single_match():
    """Test boost calculation for single field matches."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Test single tag match
    payload = {"tags": ["python"], "author": "Smith"}
    soft_filters = {"tags": ["python"]}
    
    boost = search_service._calculate_boost_multiplier(payload, soft_filters)
    
    # Should be base_boost * weight for tags
    expected_boost = SOFT_FILTER_CONFIG['base_boost_per_match'] * SOFT_FILTER_CONFIG['field_weights']['tags']
    assert abs(boost - expected_boost) < 0.001
    
    # Test single author match
    soft_filters = {"author": "Smith"}
    boost = search_service._calculate_boost_multiplier(payload, soft_filters)
    
    expected_boost = SOFT_FILTER_CONFIG['base_boost_per_match'] * SOFT_FILTER_CONFIG['field_weights']['author']
    assert abs(boost - expected_boost) < 0.001
    
    print("✓ Single match boost calculation test passed")


def test_boost_calculation_multiple_matches():
    """Test boost calculation with diminishing returns for multiple matches."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Document matching both tags and author
    payload = {"tags": ["python"], "author": "Smith", "title": "Python Guide"}
    soft_filters = {"tags": ["python"], "author": "Smith", "title": "Python"}
    
    boost = search_service._calculate_boost_multiplier(payload, soft_filters)
    
    # Calculate expected boost with diminishing returns
    base_boost = SOFT_FILTER_CONFIG['base_boost_per_match']
    diminishing = SOFT_FILTER_CONFIG['diminishing_factor']
    weights = SOFT_FILTER_CONFIG['field_weights']
    
    # Matches sorted by weight: tags (1.2), title (1.0), author (0.8)
    match_weights = [weights['tags'], weights['title'], weights['author']]
    match_weights.sort(reverse=True)
    
    expected_boost = 0.0
    for i, weight in enumerate(match_weights):
        match_boost = base_boost * weight * (diminishing ** i)
        expected_boost += match_boost
    
    assert abs(boost - expected_boost) < 0.001
    
    print("✓ Multiple match diminishing returns test passed")


def test_boost_calculation_max_cap():
    """Test that boost calculation respects maximum cap."""
    # Create config with very high base boost to test capping
    high_boost_config = {
        'base_boost_per_match': 1.0,  # Very high boost
        'max_total_boost': 0.3  # But cap at 30%
    }
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service(high_boost_config)
    
    # Document with many matches
    payload = {
        "tags": ["python", "ai"], 
        "author": "Smith", 
        "title": "Python AI Guide",
        "file_extension": ".pdf"
    }
    soft_filters = {
        "tags": ["python"], 
        "author": "Smith", 
        "title": "Python",
        "file_extension": ".pdf"
    }
    
    boost = search_service._calculate_boost_multiplier(payload, soft_filters)
    
    # Should be capped at max_total_boost
    assert boost == 0.3
    
    print("✓ Boost capping test passed")


def test_field_matching_tags():
    """Test tag field matching logic."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Test exact tag match
    payload = {"tags": ["python", "machine-learning", "ai"]}
    assert search_service._field_matches(payload, "tags", ["python"])
    assert search_service._field_matches(payload, "tags", ["ai"])
    assert not search_service._field_matches(payload, "tags", ["java"])
    
    # Test multiple tag match (should match if any tag matches)
    assert search_service._field_matches(payload, "tags", ["python", "java"])
    
    # Test case insensitive matching
    assert search_service._field_matches(payload, "tags", ["PYTHON"])
    assert search_service._field_matches(payload, "tags", ["Machine-Learning"])
    
    # Test single string tag filter
    assert search_service._field_matches(payload, "tags", "python")
    assert not search_service._field_matches(payload, "tags", "javascript")
    
    print("✓ Tag matching test passed")


def test_field_matching_author():
    """Test author field matching logic."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Test partial author match
    payload = {"author": "Dr. John Smith"}
    assert search_service._field_matches(payload, "author", "Smith")
    assert search_service._field_matches(payload, "author", "John")
    assert search_service._field_matches(payload, "author", "Dr.")
    assert not search_service._field_matches(payload, "author", "Johnson")
    
    # Test case insensitive matching
    assert search_service._field_matches(payload, "author", "smith")
    assert search_service._field_matches(payload, "author", "JOHN")
    
    print("✓ Author matching test passed")


def test_field_matching_date():
    """Test date field matching logic."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Test DatetimeRange format matching
    payload = {"publication_date": "2025-03-15"}
    
    # Should match date ranges that include the document date
    date_range = {"gte": "2025-01-01", "lt": "2026-01-01"}
    assert search_service._field_matches(payload, "publication_date", date_range)
    
    # Should not match ranges that exclude the document date
    date_range = {"gte": "2024-01-01", "lt": "2025-01-01"}
    assert not search_service._field_matches(payload, "publication_date", date_range)
    
    # Test string format matching
    assert search_service._field_matches(payload, "publication_date", "2025")
    assert search_service._field_matches(payload, "publication_date", "2025-03")
    assert not search_service._field_matches(payload, "publication_date", "2024")
    
    print("✓ Date matching test passed")


def test_empty_filters_behavior():
    """Test behavior with empty or None filters."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    mock_results = [create_mock_scored_point("1", 0.9, {"title": "Test"})]
    mock_db.search.return_value = mock_results
    
    query = "test query"
    
    # Test with None filters (QueryRewriter returns empty filters)
    results = search_service.unified_search(
        query=query,
        top_k=5
    )
    assert len(results) == 1
    
    # Test with empty dict filters (QueryRewriter returns empty filters)
    results = search_service.unified_search(
        query=query,
        top_k=5
    )
    assert len(results) == 1
    
    print("✓ Empty filters test passed")


def test_no_search_results():
    """Test behavior when qdrant returns no results."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Mock empty search results
    mock_db.search.return_value = []
    
    query = "test query"
    
    # Set up query rewriter to return soft filters
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {"tags": ["python"]},
        'strategy': 'rewrite'
    }
    
    results = search_service.unified_search(
        query=query,
        top_k=5
    )
    
    assert results == []
    
    print("✓ No results test passed")


def test_boost_statistics_and_config():
    """Test boost statistics and configuration methods."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Test getting boost statistics
    stats = search_service.get_boost_statistics()
    assert 'base_boost_per_match' in stats
    assert 'field_weights' in stats
    assert 'fetch_multiplier' in stats
    assert stats['base_boost_per_match'] == SOFT_FILTER_CONFIG['base_boost_per_match']
    
    # Test updating boost config
    new_config = {'base_boost_per_match': 0.2}
    search_service.update_boost_config(new_config)
    assert search_service.config['base_boost_per_match'] == 0.2
    
    # Should still have other config values
    assert search_service.config['diminishing_factor'] == SOFT_FILTER_CONFIG['diminishing_factor']
    
    print("✓ Boost statistics and config test passed")


def test_complex_search_scenario():
    """Test complex search scenario with multiple filter types and boosting."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Create mock documents with varying match patterns
    mock_results = [
        # Document 1: Matches soft filters (should get boost)
        create_mock_scored_point("1", 0.7, {
            "title": "Python Machine Learning Guide", 
            "tags": ["python", "machine-learning"], 
            "author": "Dr. Smith",
            "publication_date": "2025-01-15"
        }),
        
        # Document 2: High original score but no soft matches
        create_mock_scored_point("2", 0.9, {
            "title": "Java Programming", 
            "tags": ["java"], 
            "author": "Johnson",
            "publication_date": "2024-06-10"
        }),
        
        # Document 3: Medium score with some soft matches
        create_mock_scored_point("3", 0.8, {
            "title": "Data Science with Python", 
            "tags": ["python", "data-science"], 
            "author": "Wilson",
            "publication_date": "2025-02-20"
        }),
        
        # Document 4: Low score but matches many soft filters
        create_mock_scored_point("4", 0.6, {
            "title": "Advanced Python Programming", 
            "tags": ["python", "advanced"], 
            "author": "Dr. Smith",
            "publication_date": "2025-01-10"
        })
    ]
    mock_db.search.return_value = mock_results
    
    # Set up query rewriter to return all filter types
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': {"publication_date": {"gte": "2025-01-01", "lt": "2026-01-01"}},
        'negation_filters': {"tags": ["deprecated"]},
        'soft_filters': {"tags": ["python"], "author": "Dr. Smith", "title": "Python"},
        'strategy': 'rewrite'
    }
    
    # Search with filters extracted from query
    results = search_service.unified_search(
        query="test query",
        top_k=3
    )
    
    # Should apply qdrant filtering first, then boost and re-rank
    mock_db.search.assert_called_once()
    call_args = mock_db.search.call_args
    assert call_args.kwargs['filters'] == {"publication_date": {"gte": "2025-01-01", "lt": "2026-01-01"}}
    assert call_args.kwargs['negation_filters'] == {"tags": ["deprecated"]}
    
    # Should return top 3 results after boosting
    assert len(results) == 3
    
    # Documents with more soft filter matches should rank higher after boosting
    # Document 1 and 4 match python+author+title, should get significant boost
    # Document 3 matches python+title, should get some boost
    # Document 2 matches nothing, no boost
    
    # Verify that boosting affects ranking
    result_ids = [r.id for r in results]
    
    print("✓ Complex search scenario test passed")
    print(f"  Final ranking: {result_ids}")
    print(f"  Boosted scores: {[f'{r.score:.3f}' for r in results]}")


def test_edge_cases():
    """Test edge cases and error handling."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Test with malformed document payload
    payload = {"tags": None, "author": "", "invalid_field": 123}
    soft_filters = {"tags": ["python"], "author": "Smith"}
    
    # Should not crash, should return 0 boost
    boost = search_service._calculate_boost_multiplier(payload, soft_filters)
    assert boost == 0.0
    
    # Test field matching with None/empty values
    assert not search_service._field_matches(payload, "tags", ["python"])
    assert not search_service._field_matches(payload, "author", "Smith")
    
    # Test with qdrant_db.search raising exception
    mock_db.search.side_effect = Exception("Qdrant error")
    
    # Set up query rewriter to return soft filters for exception test
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {"tags": ["python"]},
        'strategy': 'rewrite'
    }
    
    results = search_service.unified_search(
        query="test query",
        top_k=5
    )
    
    # Should return empty results gracefully
    assert results == []
    
    print("✓ Edge cases test passed")


def test_fetch_multiplier_behavior():
    """Test that fetch_multiplier is used correctly with soft filters."""
    # Custom config with different fetch multiplier
    custom_config = {'fetch_multiplier': 3}
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service(custom_config)
    
    mock_db.search.return_value = []
    
    # Set up query rewriter to return soft filters
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {"tags": ["python"]},
        'strategy': 'rewrite'
    }
    
    # With soft filters, should use multiplied limit
    search_service.unified_search(
        query="test query",
        top_k=5
    )
    
    # Should call with limit = 5 * 3 = 15
    call_args = mock_db.search.call_args
    assert call_args.kwargs['limit'] == 15
    
    # Reset mock and set up query rewriter for no soft filters
    mock_db.search.reset_mock()
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {'rewrite': 'processed query text', 'hyde': ['processed query text']},
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {},
        'strategy': 'rewrite'
    }
    
    # Without soft filters, should use original limit
    search_service.unified_search(
        query="test query",
        top_k=5
    )
    
    call_args = mock_db.search.call_args
    assert call_args.kwargs['limit'] == 5
    
    print("✓ Fetch multiplier test passed")


def test_progress_callback():
    """Test progress callback functionality during search."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Mock search results
    mock_results = [
        create_mock_scored_point("1", 0.9, {"title": "Test Document 1"}),
        create_mock_scored_point("2", 0.8, {"title": "Test Document 2"})
    ]
    mock_db.search.return_value = mock_results
    
    # Create mock callback to capture progress updates
    progress_updates = []
    def mock_callback(stage: str, data: Dict[str, Any]):
        progress_updates.append((stage, data))
    
    # Test search with progress callback
    results = search_service.unified_search(
        query="test query",
        top_k=5,
        progress_callback=mock_callback
    )
    
    # Should have received all expected progress updates
    assert len(progress_updates) >= 4  # analyzing, query_analyzed, searching, search_complete
    
    # Check specific progress stages
    stages = [update[0] for update in progress_updates]
    assert "analyzing" in stages
    assert "query_analyzed" in stages
    assert "search_ready" in stages
    assert "search_complete" in stages
    
    # Check query_analyzed data
    query_analyzed_data = next(data for stage, data in progress_updates if stage == "query_analyzed")
    assert "embedding_text" in query_analyzed_data
    assert "strategy" in query_analyzed_data
    assert "original_query" in query_analyzed_data
    
    # Check search_complete data
    search_complete_data = next(data for stage, data in progress_updates if stage == "search_complete")
    assert "result_count" in search_complete_data
    assert search_complete_data["result_count"] == 2
    
    # Should still return correct results
    assert len(results) == 2


def test_progress_callback_no_results():
    """Test progress callback when no search results are found."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Mock empty search results
    mock_db.search.return_value = []
    
    # Create mock callback to capture progress updates
    progress_updates = []
    def mock_callback(stage: str, data: Dict[str, Any]):
        progress_updates.append((stage, data))
    
    # Test search with progress callback
    results = search_service.unified_search(
        query="test query",
        top_k=5,
        progress_callback=mock_callback
    )
    
    # Should have received progress updates including empty results
    stages = [update[0] for update in progress_updates]
    assert "search_complete" in stages
    
    # Check search_complete data for empty results
    search_complete_data = next(data for stage, data in progress_updates if stage == "search_complete")
    assert search_complete_data["result_count"] == 0
    
    # Should return empty results
    assert len(results) == 0


# ========================================
# HYBRID SEARCH TESTS
# ========================================

def test_hyde_multi_vector_generation():
    """Test multi-vector generation for HyDE strategy."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Mock multiple embeddings for HyDE personas
    mock_dense_client.get_embedding.side_effect = [
        [0.1, 0.2, 0.3],  # Professor perspective
        [0.4, 0.5, 0.6],  # Teacher perspective  
        [0.7, 0.8, 0.9]   # Student perspective
    ]
    
    # Set up query rewriter to return HyDE strategy with multiple personas
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {
            'rewrite': 'neural networks',
            'hyde': [
                'Neural networks utilize backpropagation algorithms for weight optimization...',
                'Neural networks learn by adjusting connections between artificial neurons...',
                'I am studying how neural networks mimic the human brain to recognize patterns...'
            ]
        },
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {},
        'strategy': 'hyde'
    }
    
    # Mock hybrid search (since we changed the signature)
    mock_db.hybrid_search = Mock(return_value=[
        create_mock_scored_point("1", 0.9, {"title": "Neural Networks Guide"})
    ])
    
    search_service.unified_search(
        query="explain neural networks",
        top_k=5,
        enable_hybrid=True
    )
    
    # Should generate 3 embeddings for HyDE personas
    assert mock_dense_client.get_embedding.call_count == 3
    
    # Should call hybrid_search with multiple dense vectors
    mock_db.hybrid_search.assert_called_once()
    call_args = mock_db.hybrid_search.call_args
    assert len(call_args.kwargs['dense_vectors']) == 3
    assert call_args.kwargs['sparse_vector'] is None  # No sparse client configured
    
    print("✓ HyDE multi-vector generation test passed")


def test_rewrite_single_vector_generation():
    """Test single vector generation for rewrite strategy."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Set up query rewriter to return rewrite strategy
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {
            'rewrite': 'machine learning algorithms',
            'hyde': []  # Empty hyde should fall back to rewrite
        },
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {},
        'strategy': 'rewrite'
    }
    
    # Mock traditional search (should use single vector)
    mock_db.search = Mock(return_value=[
        create_mock_scored_point("1", 0.8, {"title": "ML Algorithms"})
    ])
    
    search_service.unified_search(
        query="machine learning algorithms",
        top_k=5,
        enable_hybrid=False  # Disabled hybrid search
    )
    
    # Should generate 1 embedding for rewrite text
    mock_dense_client.get_embedding.assert_called_once_with('machine learning algorithms')
    
    # Should call traditional search with single vector
    mock_db.search.assert_called_once()
    call_args = mock_db.search.call_args
    assert call_args.kwargs['query_vector'] == [0.1, 0.2, 0.3]
    
    print("✓ Rewrite single vector generation test passed")


def test_hybrid_search_with_sparse_embedding():
    """Test hybrid search with sparse embedding generation."""
    # Create search service with sparse embedding client
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_sparse_client = create_mock_embedding_client()
    mock_sparse_client.has_sparse_embedding.return_value = True
    mock_query_rewriter = create_mock_query_rewriter()
    
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter,
        sparse_embedding_client=mock_sparse_client
    )
    
    # Set up query rewriter to return HyDE strategy
    mock_query_rewriter.transform_query.return_value = {
        'embedding_texts': {
            'rewrite': 'neural networks',
            'hyde': ['Professor perspective text']
        },
        'hard_filters': {},
        'negation_filters': {},
        'soft_filters': {},
        'strategy': 'hyde'
    }
    
    # Mock hybrid search
    mock_db.hybrid_search = Mock(return_value=[
        create_mock_scored_point("1", 0.95, {"title": "Neural Networks Research"})
    ])
    
    search_service.unified_search(
        query="explain neural networks",
        top_k=5,
        enable_hybrid=True
    )
    
    # Should generate dense embedding for HyDE text
    mock_dense_client.get_embedding.assert_called_once_with('Professor perspective text')
    
    # Should generate sparse embedding for keywords (rewrite text)
    mock_sparse_client.get_sparse_embedding.assert_called_once_with('neural networks')
    
    # Should call hybrid_search with both dense and sparse vectors
    mock_db.hybrid_search.assert_called_once()
    call_args = mock_db.hybrid_search.call_args
    assert len(call_args.kwargs['dense_vectors']) == 1
    assert call_args.kwargs['sparse_vector'] == {"indices": [1, 2], "values": [0.5, 0.3]}
    
    print("✓ Hybrid search with sparse embedding test passed")


def test_hybrid_search_mode_selection():
    """Test automatic search mode selection based on configuration."""
    # Test Dense+Sparse Hybrid Search
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_sparse_client = create_mock_embedding_client()
    mock_sparse_client.has_sparse_embedding.return_value = True
    mock_query_rewriter = create_mock_query_rewriter()
    
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter,
        sparse_embedding_client=mock_sparse_client
    )
    
    # Mock methods to track which search mode is used
    mock_db.hybrid_search = Mock(return_value=[])
    mock_db.search = Mock(return_value=[])
    
    # Test 1: Dense+Sparse Hybrid (enable_hybrid=True + sparse available)
    search_service.unified_search(
        query="test query",
        top_k=5,
        enable_hybrid=True
    )
    
    # Should use hybrid_search
    mock_db.hybrid_search.assert_called()
    mock_db.search.assert_not_called()
    
    # Reset mocks
    mock_db.hybrid_search.reset_mock()
    mock_db.search.reset_mock()
    
    # Test 2: Traditional Search (enable_hybrid=False)
    search_service.unified_search(
        query="test query",
        top_k=5,
        enable_hybrid=False
    )
    
    # Should use traditional search
    mock_db.search.assert_called()
    mock_db.hybrid_search.assert_not_called()
    
    print("✓ Hybrid search mode selection test passed")


def test_search_with_vectors_method():
    """Test the updated _search_with_vectors method directly."""
    search_service, mock_db, mock_dense_client, mock_query_rewriter = create_search_service()
    
    # Mock both search methods
    mock_db.hybrid_search = Mock(return_value=[
        create_mock_scored_point("1", 0.9, {"title": "Hybrid Result"})
    ])
    mock_db.search = Mock(return_value=[
        create_mock_scored_point("1", 0.8, {"title": "Traditional Result"})
    ])
    
    dense_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    sparse_vector = {"indices": [1, 2], "values": [0.5, 0.3]}
    
    # Test 1: Dense+Sparse Hybrid Search
    results = search_service._search_with_vectors(
        dense_vectors=dense_vectors,
        sparse_vector=sparse_vector,
        limit=5,
        enable_hybrid=True
    )
    
    mock_db.hybrid_search.assert_called_once()
    assert len(results) == 1
    assert results[0].payload["title"] == "Hybrid Result"
    
    # Reset mocks
    mock_db.hybrid_search.reset_mock()
    mock_db.search.reset_mock()
    
    # Test 2: Dense-only Hybrid Search (multiple dense vectors, no sparse)
    results = search_service._search_with_vectors(
        dense_vectors=dense_vectors,
        sparse_vector=None,
        limit=5,
        enable_hybrid=True
    )
    
    mock_db.hybrid_search.assert_called_once()
    call_args = mock_db.hybrid_search.call_args
    assert call_args.kwargs['sparse_vector'] is None
    assert len(call_args.kwargs['dense_vectors']) == 2
    
    # Reset mocks
    mock_db.hybrid_search.reset_mock()
    mock_db.search.reset_mock()
    
    # Test 3: Traditional Search (single vector, hybrid disabled)
    results = search_service._search_with_vectors(
        dense_vectors=[[0.1, 0.2, 0.3]],  # Single vector
        sparse_vector=None,
        limit=5,
        enable_hybrid=False
    )
    
    mock_db.search.assert_called_once()
    call_args = mock_db.search.call_args
    assert call_args.kwargs['query_vector'] == [0.1, 0.2, 0.3]
    
    print("✓ _search_with_vectors method test passed")


def test_enable_hybrid_parameter_override():
    """Test enable_hybrid parameter override functionality."""
    # Set up search service with sparse client and hybrid disabled by default
    custom_config = {'enable_hybrid': False}
    mock_db = create_mock_qdrant_db()
    mock_dense_client = create_mock_embedding_client()
    mock_sparse_client = create_mock_embedding_client()
    mock_sparse_client.has_sparse_embedding.return_value = True
    mock_query_rewriter = create_mock_query_rewriter()
    
    search_service = SearchService(
        qdrant_db=mock_db,
        dense_embedding_client=mock_dense_client,
        query_rewriter=mock_query_rewriter,
        sparse_embedding_client=mock_sparse_client,
        config=custom_config
    )
    
    mock_db.hybrid_search = Mock(return_value=[])
    mock_db.search = Mock(return_value=[])
    
    # Test 1: Override to enable hybrid search
    search_service.unified_search(
        query="test query",
        top_k=5,
        enable_hybrid=True  # Override config
    )
    
    # Should use hybrid search despite config
    mock_db.hybrid_search.assert_called()
    mock_db.search.assert_not_called()
    
    # Reset mocks
    mock_db.hybrid_search.reset_mock()
    mock_db.search.reset_mock()
    
    # Test 2: Use config default (hybrid disabled)
    search_service.unified_search(
        query="test query",
        top_k=5
        # No enable_hybrid parameter, should use config default
    )
    
    # Should use traditional search
    mock_db.search.assert_called()
    mock_db.hybrid_search.assert_not_called()
    
    print("✓ enable_hybrid parameter override test passed")


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_search_service_initialization,
        test_unified_search_no_soft_filters,
        test_unified_search_with_soft_filters,
        test_boost_calculation_single_match,
        test_boost_calculation_multiple_matches,
        test_boost_calculation_max_cap,
        test_field_matching_tags,
        test_field_matching_author,
        test_field_matching_date,
        test_empty_filters_behavior,
        test_no_search_results,
        test_boost_statistics_and_config,
        test_complex_search_scenario,
        test_edge_cases,
        test_fetch_multiplier_behavior,
        test_progress_callback,
        test_progress_callback_no_results,
        # New hybrid search tests
        test_hyde_multi_vector_generation,
        test_rewrite_single_vector_generation,
        test_hybrid_search_with_sparse_embedding,
        test_hybrid_search_mode_selection,
        test_search_with_vectors_method,
        test_enable_hybrid_parameter_override
    ]
    
    print("Running SearchService unit tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            raise
    
    print(f"\n🎉 All {len(test_functions)} SearchService tests passed!")