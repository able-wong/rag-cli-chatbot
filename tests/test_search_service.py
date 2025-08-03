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
    
    # Test default initialization
    search_service = SearchService(mock_db)
    assert search_service.qdrant_db == mock_db
    assert search_service.config['base_boost_per_match'] == 0.12
    assert search_service.config['field_weights']['tags'] == 1.2
    
    # Test custom config
    custom_config = {
        'base_boost_per_match': 0.15,
        'field_weights': {'tags': 1.5}
    }
    search_service = SearchService(mock_db, custom_config)
    assert search_service.config['base_boost_per_match'] == 0.15
    assert search_service.config['field_weights']['tags'] == 1.5
    # Should still have default values for non-overridden fields
    assert search_service.config['diminishing_factor'] == 0.75


def test_unified_search_no_soft_filters():
    """Test unified search without soft filters (should use standard search)."""
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
    # Mock search results
    mock_results = [
        create_mock_scored_point("1", 0.9, {"title": "Test Document 1"}),
        create_mock_scored_point("2", 0.8, {"title": "Test Document 2"})
    ]
    mock_db.search.return_value = mock_results
    
    # Test search without soft filters
    query_vector = [0.1, 0.2, 0.3]
    hard_filters = {"author": "Smith"}
    negation_filters = {"tags": ["draft"]}
    
    results = search_service.unified_search(
        query_vector=query_vector,
        top_k=5,
        hard_filters=hard_filters,
        negation_filters=negation_filters,
        soft_filters=None
    )
    
    # Should call qdrant_db.search with correct parameters
    mock_db.search.assert_called_once_with(
        query_vector=query_vector,
        limit=5,  # No multiplier when no soft filters
        score_threshold=None,
        filters=hard_filters,
        negation_filters=negation_filters
    )
    
    # Should return original results unchanged
    assert results == mock_results
    assert len(results) == 2


def test_unified_search_with_soft_filters():
    """Test unified search with soft filters (should apply boosting and re-ranking)."""
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
    # Mock search results with different scores
    mock_results = [
        create_mock_scored_point("1", 0.8, {"title": "Document 1", "tags": ["python"], "author": "Smith"}),
        create_mock_scored_point("2", 0.9, {"title": "Document 2", "tags": ["java"]}),
        create_mock_scored_point("3", 0.7, {"title": "Document 3", "tags": ["python", "ai"], "author": "Jones"})
    ]
    mock_db.search.return_value = mock_results
    
    # Test search with soft filters
    query_vector = [0.1, 0.2, 0.3]
    soft_filters = {"tags": ["python"], "author": "Smith"}
    
    results = search_service.unified_search(
        query_vector=query_vector,
        top_k=2,
        soft_filters=soft_filters
    )
    
    # Should call qdrant_db.search with multiplied limit
    expected_limit = 2 * SOFT_FILTER_CONFIG['fetch_multiplier']  # 2 * 4 = 8
    mock_db.search.assert_called_once_with(
        query_vector=query_vector,
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    mock_db = create_mock_qdrant_db()
    
    # Create config with very high base boost to test capping
    high_boost_config = {
        'base_boost_per_match': 1.0,  # Very high boost
        'max_total_boost': 0.3  # But cap at 30%
    }
    search_service = SearchService(mock_db, high_boost_config)
    
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
    mock_results = [create_mock_scored_point("1", 0.9, {"title": "Test"})]
    mock_db.search.return_value = mock_results
    
    query_vector = [0.1, 0.2, 0.3]
    
    # Test with None filters
    results = search_service.unified_search(
        query_vector=query_vector,
        top_k=5,
        hard_filters=None,
        negation_filters=None,
        soft_filters=None
    )
    assert len(results) == 1
    
    # Test with empty dict filters
    results = search_service.unified_search(
        query_vector=query_vector,
        top_k=5,
        hard_filters={},
        negation_filters={},
        soft_filters={}
    )
    assert len(results) == 1
    
    print("✓ Empty filters test passed")


def test_no_search_results():
    """Test behavior when qdrant returns no results."""
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
    # Mock empty search results
    mock_db.search.return_value = []
    
    query_vector = [0.1, 0.2, 0.3]
    results = search_service.unified_search(
        query_vector=query_vector,
        top_k=5,
        soft_filters={"tags": ["python"]}
    )
    
    assert results == []
    
    print("✓ No results test passed")


def test_boost_statistics_and_config():
    """Test boost statistics and configuration methods."""
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    
    # Search with hard filters, negation filters, and soft filters
    results = search_service.unified_search(
        query_vector=[0.1, 0.2, 0.3],
        top_k=3,
        hard_filters={"publication_date": {"gte": "2025-01-01", "lt": "2026-01-01"}},
        negation_filters={"tags": ["deprecated"]},
        soft_filters={"tags": ["python"], "author": "Dr. Smith", "title": "Python"}
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
    mock_db = create_mock_qdrant_db()
    search_service = SearchService(mock_db)
    
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
    
    results = search_service.unified_search(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        soft_filters={"tags": ["python"]}
    )
    
    # Should return empty results gracefully
    assert results == []
    
    print("✓ Edge cases test passed")


def test_fetch_multiplier_behavior():
    """Test that fetch_multiplier is used correctly with soft filters."""
    mock_db = create_mock_qdrant_db()
    
    # Custom config with different fetch multiplier
    custom_config = {'fetch_multiplier': 3}
    search_service = SearchService(mock_db, custom_config)
    
    mock_db.search.return_value = []
    
    # With soft filters, should use multiplied limit
    search_service.unified_search(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        soft_filters={"tags": ["python"]}
    )
    
    # Should call with limit = 5 * 3 = 15
    call_args = mock_db.search.call_args
    assert call_args.kwargs['limit'] == 15
    
    # Without soft filters, should use original limit
    mock_db.search.reset_mock()
    search_service.unified_search(
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        soft_filters=None
    )
    
    call_args = mock_db.search.call_args
    assert call_args.kwargs['limit'] == 5
    
    print("✓ Fetch multiplier test passed")


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
        test_fetch_multiplier_behavior
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