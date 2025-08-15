"""
Unit tests for QdrantDB filtering functionality.
Tests the _build_qdrant_filter method and search with filters.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qdrant_db import QdrantDB
from qdrant_client.models import Filter, FusionQuery, Fusion

def create_mock_qdrant_client(collection_exists=False):
    """Create a mock Qdrant client for testing."""
    mock_client = Mock()
    if collection_exists:
        mock_collections = Mock()
        mock_collections.collections = [Mock(name='test')]
        mock_client.get_collections.return_value = mock_collections
    else:
        mock_client.get_collections.return_value = Mock(collections=[])
    return mock_client


# ========================================
# NEGATION FILTER TESTS
# ========================================

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_positive_only(mock_qdrant_client_class):
    """Test building combined filter with only positive filters."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    filters = {"author": "John Smith"}
    negation_filters = None
    
    combined_filter = db._build_combined_filter(filters, negation_filters)
    
    assert combined_filter is not None
    assert isinstance(combined_filter, Filter)
    assert len(combined_filter.must) == 1
    assert not hasattr(combined_filter, 'must_not') or combined_filter.must_not is None
    
    condition = combined_filter.must[0]
    assert condition.key == "author"
    assert condition.match.value == "John Smith"

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_negation_only(mock_qdrant_client_class):
    """Test building combined filter with only negation filters."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    filters = None
    negation_filters = {"author": "John Wong"}
    
    combined_filter = db._build_combined_filter(filters, negation_filters)
    
    assert combined_filter is not None
    assert isinstance(combined_filter, Filter)
    assert not hasattr(combined_filter, 'must') or combined_filter.must is None
    assert len(combined_filter.must_not) == 1
    
    condition = combined_filter.must_not[0]
    assert condition.key == "author"
    assert condition.match.value == "John Wong"

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_both_positive_and_negative(mock_qdrant_client_class):
    """Test building combined filter with both positive and negative filters."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    filters = {"tags": ["python"]}
    negation_filters = {"author": "John Wong"}
    
    combined_filter = db._build_combined_filter(filters, negation_filters)
    
    assert combined_filter is not None
    assert isinstance(combined_filter, Filter)
    assert len(combined_filter.must) == 1
    assert len(combined_filter.must_not) == 1
    
    # Check positive condition
    positive_condition = combined_filter.must[0]
    assert positive_condition.key == "tags"
    assert positive_condition.match.any == ["python"]
    
    # Check negative condition
    negative_condition = combined_filter.must_not[0]
    assert negative_condition.key == "author"
    assert negative_condition.match.value == "John Wong"

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_multiple_negation_conditions(mock_qdrant_client_class):
    """Test building combined filter with multiple negation conditions."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    filters = None
    negation_filters = {
        "author": "John Wong",
        "tags": ["exclude_tag"],
        "title": "Exclude This Title"
    }
    
    combined_filter = db._build_combined_filter(filters, negation_filters)
    
    assert combined_filter is not None
    assert len(combined_filter.must_not) == 3
    
    # Verify all negation conditions are present
    keys = [condition.key for condition in combined_filter.must_not]
    assert "author" in keys
    assert "tags" in keys
    assert "title" in keys

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_negation_date_range(mock_qdrant_client_class):
    """Test building combined filter with negation date range."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    filters = None
    negation_filters = {
        "publication_date": {
            "gte": "2023-01-01",
            "lt": "2024-01-01"
        }
    }
    
    combined_filter = db._build_combined_filter(filters, negation_filters)
    
    assert combined_filter is not None
    assert len(combined_filter.must_not) == 1
    
    condition = combined_filter.must_not[0]
    assert condition.key == "publication_date"
    from qdrant_client.models import DatetimeRange
    assert isinstance(condition.range, DatetimeRange)

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_empty_filters(mock_qdrant_client_class):
    """Test building combined filter with empty filters returns None."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test with empty/None filters
    combined_filter = db._build_combined_filter(None, None)
    assert combined_filter is None
    
    combined_filter = db._build_combined_filter({}, {})
    assert combined_filter is None
    
    # Test with filters containing only empty values
    combined_filter = db._build_combined_filter({"author": ""}, {"tags": []})
    assert combined_filter is None

@patch('qdrant_db.QdrantClient')
def test_build_combined_filter_error_handling(mock_qdrant_client_class):
    """Test error handling in combined filter building."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Mock an exception in filter condition building
    with patch.object(db, '_build_filter_conditions', side_effect=Exception("Filter error")):
        combined_filter = db._build_combined_filter({"author": "Test"}, {"tags": ["python"]})
        
        # Should return None on error
        assert combined_filter is None

@patch('qdrant_db.QdrantClient')
def test_search_with_negation_filters_only(mock_qdrant_client_class):
    """Test search method with only negation filters."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock query_points response for hybrid search
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        query_vector = [0.1, 0.2, 0.3]
        negation_filters = {"author": "John Wong"}
        
        db.search(
            query_vector=query_vector,
            limit=5,
            negation_filters=negation_filters
        )
        
        # Verify query_points was called (hybrid search backend)
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        # Check that filters were applied to prefetch queries
        assert 'prefetch' in call_args
        prefetch_queries = call_args['prefetch']
        for prefetch in prefetch_queries:
            assert hasattr(prefetch, 'filter')
            assert prefetch.filter is not None

@patch('qdrant_db.QdrantClient')
def test_search_with_both_filters_and_negation_filters(mock_qdrant_client_class):
    """Test search method with both positive and negative filters."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock query_points response for hybrid search
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        query_vector = [0.1, 0.2, 0.3]
        filters = {"tags": ["python"]}
        negation_filters = {"author": "John Wong"}
        
        db.search(
            query_vector=query_vector,
            limit=5,
            filters=filters,
            negation_filters=negation_filters
        )
        
        # Verify query_points was called (hybrid search backend)
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        # Check that filters were applied to prefetch queries
        assert 'prefetch' in call_args
        prefetch_queries = call_args['prefetch']
        for prefetch in prefetch_queries:
            assert hasattr(prefetch, 'filter')
            query_filter = prefetch.filter
            assert query_filter is not None
            assert len(query_filter.must) == 1  # positive filter
            assert len(query_filter.must_not) == 1  # negative filter

@patch('qdrant_db.QdrantClient')
def test_search_negation_filter_logging(mock_qdrant_client_class):
    """Test that negation filters are properly logged."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock query_points response for hybrid search
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        query_vector = [0.1, 0.2, 0.3]
        filters = {"author": "Jane Doe"}
        negation_filters = {"tags": ["exclude"]}
        
        # Capture log output
        with patch('qdrant_db.logger') as mock_logger:
            db.search(
                query_vector=query_vector,
                limit=5,
                filters=filters,
                negation_filters=negation_filters
            )
            
            # Check that the logging call includes both positive (+) and negative (-) prefixes
            log_calls = [call for call in mock_logger.info.call_args_list 
                        if "Applied hybrid search filters" in str(call)]
            assert len(log_calls) == 1
            
            log_message = str(log_calls[0])
            assert "+author" in log_message  # positive filter
            assert "-tags" in log_message    # negative filter

@patch('qdrant_db.QdrantClient')
def test_build_filter_conditions_comprehensive(mock_qdrant_client_class):
    """Test _build_filter_conditions method with all field types."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test with all supported filter types
    filters = {
        "author": "Test Author",
        "tags": ["python", "ai"],
        "title": "Test Title",
        "publication_date": {
            "gte": "2023-01-01",
            "lt": "2024-01-01"
        }
    }
    
    conditions = db._build_filter_conditions(filters)
    
    assert len(conditions) == 4
    
    # Verify all condition types are created correctly
    condition_keys = [condition.key for condition in conditions]
    assert "author" in condition_keys
    assert "tags" in condition_keys
    assert "title" in condition_keys
    assert "publication_date" in condition_keys

@patch('qdrant_db.QdrantClient')
def test_backward_compatibility_with_legacy_search(mock_qdrant_client_class):
    """Test that the enhanced search method maintains backward compatibility."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock query_points response for hybrid search
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        query_vector = [0.1, 0.2, 0.3]
        
        # Test legacy call (only filters parameter) - now uses hybrid search
        db.search(
            query_vector=query_vector,
            limit=5,
            filters={"author": "John Smith"}
        )
        
        # Should work without issues via hybrid search
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        assert 'prefetch' in call_args
        
        # Reset mock for second test
        mock_client.reset_mock()
        
        # Test completely legacy call (no filters at all) - now uses hybrid search
        db.search(
            query_vector=query_vector,
            limit=5
        )
        
        # Should work without filters via hybrid search
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        assert 'prefetch' in call_args
        # No filters should be applied to prefetch queries
        prefetch_queries = call_args['prefetch']
        for prefetch in prefetch_queries:
            assert not hasattr(prefetch, 'filter') or prefetch.filter is None


# ========================================
# HYBRID SEARCH TESTS
# ========================================

@patch('qdrant_db.QdrantClient')
def test_hybrid_search_basic_functionality(mock_qdrant_client_class):
    """Test basic hybrid search functionality with dense and sparse vectors."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock query_points response
    mock_response = Mock()
    mock_response.points = [Mock(id=1, score=0.9, payload={"title": "Test Doc"})]
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        sparse_vector = {"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]}
        
        results = db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5
        )
        
        # Verify query_points was called
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        # Check prefetch queries were created correctly
        assert 'prefetch' in call_args
        prefetch_queries = call_args['prefetch']
        assert len(prefetch_queries) == 3  # 2 dense + 1 sparse
        
        # Check RRF fusion query
        assert 'query' in call_args
        fusion_query = call_args['query']
        assert isinstance(fusion_query, FusionQuery)
        assert fusion_query.fusion == Fusion.RRF
        
        # Check results
        assert len(results) == 1
        assert results[0].score == 0.9


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_dense_only_fallback(mock_qdrant_client_class):
    """Test hybrid search with invalid sparse vector falls back to dense only."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = [[0.1, 0.2, 0.3]]
        sparse_vector = {"indices": [], "values": []}  # Invalid sparse vector
        
        db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5
        )
        
        # Should still call query_points but only with dense vectors
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        prefetch_queries = call_args['prefetch']
        assert len(prefetch_queries) == 1  # Only dense vector


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_with_filters(mock_qdrant_client_class):
    """Test hybrid search with metadata and negation filters."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = [[0.1, 0.2, 0.3]]
        sparse_vector = {"indices": [0, 1], "values": [0.1, 0.2]}
        filters = {"author": "John Smith"}
        negation_filters = {"tags": ["exclude"]}
        
        db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5,
            filters=filters,
            negation_filters=negation_filters
        )
        
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        # Check that filters were applied to prefetch queries
        prefetch_queries = call_args['prefetch']
        for prefetch in prefetch_queries:
            assert hasattr(prefetch, 'filter')
            assert prefetch.filter is not None


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_error_handling(mock_qdrant_client_class):
    """Test hybrid search error handling returns empty list."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock query_points to fail
    mock_client.query_points.side_effect = Exception("Hybrid search error")
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = [[0.1, 0.2, 0.3]]
        sparse_vector = {"indices": [0], "values": [0.1]}
        
        results = db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5
        )
        
        # Should have attempted query_points and returned empty list on error
        mock_client.query_points.assert_called_once()
        
        # Should return empty list on error
        assert results == []


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_no_dense_vectors(mock_qdrant_client_class):
    """Test hybrid search with no dense vectors returns empty list."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = []  # No dense vectors
        sparse_vector = {"indices": [0], "values": [0.1]}
        
        results = db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5
        )
        
        # Should return empty list without calling query_points
        assert results == []
        mock_client.query_points.assert_not_called()


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_collection_not_exists(mock_qdrant_client_class):
    """Test hybrid search when collection doesn't exist."""
    mock_client = create_mock_qdrant_client(collection_exists=False)
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=False):
        dense_vectors = [[0.1, 0.2, 0.3]]
        sparse_vector = {"indices": [0], "values": [0.1]}
        
        results = db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5
        )
        
        # Should return empty list
        assert results == []
        mock_client.query_points.assert_not_called()


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_score_threshold(mock_qdrant_client_class):
    """Test hybrid search with score threshold parameter."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = [[0.1, 0.2, 0.3]]
        sparse_vector = {"indices": [0], "values": [0.1]}
        score_threshold = 0.7
        
        db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=5,
            score_threshold=score_threshold
        )
        
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        # Check score threshold was passed
        assert 'score_threshold' in call_args
        assert call_args['score_threshold'] == 0.7


@patch('qdrant_db.QdrantClient')
def test_hybrid_search_prefetch_configuration(mock_qdrant_client_class):
    """Test that prefetch queries are configured correctly."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    mock_response = Mock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        dense_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        sparse_vector = {"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]}
        limit = 5
        
        db.hybrid_search(
            dense_vectors=dense_vectors,
            sparse_vector=sparse_vector,
            limit=limit
        )
        
        mock_client.query_points.assert_called_once()
        call_args = mock_client.query_points.call_args[1]
        
        prefetch_queries = call_args['prefetch']
        
        # Verify dense vector prefetches
        dense_prefetches = [p for p in prefetch_queries if hasattr(p, 'using') and p.using == 'dense']
        assert len(dense_prefetches) == 2
        
        for prefetch in dense_prefetches:
            assert prefetch.limit == limit * 2  # Should fetch 2x for better RRF
        
        # Verify sparse vector prefetch
        sparse_prefetches = [p for p in prefetch_queries if hasattr(p, 'using') and p.using == 'sparse']
        assert len(sparse_prefetches) == 1
        sparse_prefetch = sparse_prefetches[0]
        assert sparse_prefetch.limit == limit * 2