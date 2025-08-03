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
from qdrant_client.models import Filter

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
    mock_client.search.return_value = []
    
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
        
        # Verify search was called with query_filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        
        assert 'query_filter' in call_args
        assert call_args['query_filter'] is not None
        assert len(call_args['query_filter'].must_not) == 1

@patch('qdrant_db.QdrantClient')
def test_search_with_both_filters_and_negation_filters(mock_qdrant_client_class):
    """Test search method with both positive and negative filters."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    mock_client.search.return_value = []
    
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
        
        # Verify search was called with combined filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        
        assert 'query_filter' in call_args
        query_filter = call_args['query_filter']
        assert query_filter is not None
        assert len(query_filter.must) == 1  # positive filter
        assert len(query_filter.must_not) == 1  # negative filter

@patch('qdrant_db.QdrantClient')
def test_search_negation_filter_logging(mock_qdrant_client_class):
    """Test that negation filters are properly logged."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    mock_client.search.return_value = []
    
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
    mock_client.search.return_value = []
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    with patch.object(db, 'collection_exists', return_value=True):
        query_vector = [0.1, 0.2, 0.3]
        
        # Test legacy call (only filters parameter)
        db.search(
            query_vector=query_vector,
            limit=5,
            filters={"author": "John Smith"}
        )
        
        # Should work without issues
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        assert 'query_filter' in call_args
        
        # Reset mock for second test
        mock_client.reset_mock()
        
        # Test completely legacy call (no filters at all)
        db.search(
            query_vector=query_vector,
            limit=5
        )
        
        # Should work without query_filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        assert 'query_filter' not in call_args