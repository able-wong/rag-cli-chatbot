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
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

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

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_author_filter(mock_qdrant_client_class):
    """Test building Qdrant filter for author field."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test author filter
    filters = {"author": "John Smith"}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    assert isinstance(qdrant_filter, Filter)
    assert len(qdrant_filter.must) == 1
    
    condition = qdrant_filter.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "author"
    assert isinstance(condition.match, MatchValue)
    assert condition.match.value == "John Smith"

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_tags_filter(mock_qdrant_client_class):
    """Test building Qdrant filter for tags field (array)."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test tags filter with single tag
    filters = {"tags": ["python"]}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    assert isinstance(qdrant_filter, Filter)
    assert len(qdrant_filter.must) == 1
    
    condition = qdrant_filter.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "tags"
    assert isinstance(condition.match, MatchAny)
    assert condition.match.any == ["python"]

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_tags_multiple(mock_qdrant_client_class):
    """Test building Qdrant filter for multiple tags."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test multiple tags
    filters = {"tags": ["python", "machine learning"]}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert condition.match.any == ["python", "machine learning"]

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_tags_string_conversion(mock_qdrant_client_class):
    """Test that single tag string is converted to list."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test single tag as string (should be converted to list)
    filters = {"tags": "python"}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert condition.match.any == ["python"]

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_publication_date_year(mock_qdrant_client_class):
    """Test building Qdrant filter for publication date (year only)."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test year filter (now converted to DatetimeRange)
    filters = {"publication_date": "2023"}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "publication_date"
    from qdrant_client.models import DatetimeRange
    assert isinstance(condition.range, DatetimeRange)
    # Year 2023 should be converted to 2023-01-01 to 2024-01-01
    assert condition.range.gte.strftime('%Y-%m-%d') == "2023-01-01"
    assert condition.range.lt.strftime('%Y-%m-%d') == "2024-01-01"

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_publication_date_exact(mock_qdrant_client_class):
    """Test building Qdrant filter for exact publication date."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test exact date filter
    filters = {"publication_date": "2023-03-15"}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert condition.key == "publication_date"
    assert isinstance(condition.match, MatchValue)
    assert condition.match.value == "2023-03-15"

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_combined_filters(mock_qdrant_client_class):
    """Test building Qdrant filter with multiple field conditions."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test combined filters
    filters = {
        "author": "John Smith",
        "tags": ["python", "ai"],
        "publication_date": "2023"
    }
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    assert len(qdrant_filter.must) == 3
    
    # Check that all conditions are present
    keys = [condition.key for condition in qdrant_filter.must]
    assert "author" in keys
    assert "tags" in keys
    assert "publication_date" in keys

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_title_filter(mock_qdrant_client_class):
    """Test building Qdrant filter for title field."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test title filter
    filters = {"title": "Introduction to Machine Learning"}
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    condition = qdrant_filter.must[0]
    assert condition.key == "title"
    assert condition.match.value == "Introduction to Machine Learning"

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_empty_filters(mock_qdrant_client_class):
    """Test that empty filters return None."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test empty filters
    qdrant_filter = db._build_qdrant_filter({})
    assert qdrant_filter is None
    
    # Test None filters
    qdrant_filter = db._build_qdrant_filter(None)
    assert qdrant_filter is None

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_filters_with_empty_values(mock_qdrant_client_class):
    """Test that filters with empty values are ignored."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Test filters with empty values
    filters = {
        "author": "",  # Empty string
        "tags": [],    # Empty list
        "publication_date": None,  # None value
        "title": "Valid Title"  # Only this should be included
    }
    qdrant_filter = db._build_qdrant_filter(filters)
    
    assert qdrant_filter is not None
    assert len(qdrant_filter.must) == 1
    assert qdrant_filter.must[0].key == "title"

@patch('qdrant_db.QdrantClient')
def test_qdrant_filter_builder_error_handling(mock_qdrant_client_class):
    """Test error handling in filter building."""
    mock_client = create_mock_qdrant_client()
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Mock an exception in filter building
    with patch('qdrant_db.FieldCondition', side_effect=Exception("Filter error")):
        filters = {"author": "Test Author"}
        qdrant_filter = db._build_qdrant_filter(filters)
        
        # Should return None on error
        assert qdrant_filter is None

@patch('qdrant_db.QdrantClient')
def test_search_with_filters_integration(mock_qdrant_client_class):
    """Test search method with filters integration."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    mock_client.search.return_value = []  # Mock empty search results
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Mock collection_exists to return True
    with patch.object(db, 'collection_exists', return_value=True):
        # Test search with filters
        query_vector = [0.1, 0.2, 0.3]
        filters = {"author": "John Smith", "tags": ["python"]}
        
        db.search(
            query_vector=query_vector,
            limit=5,
            filters=filters
        )
        
        # Verify search was called with the correct parameters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]  # Get keyword arguments
        
        assert call_args['collection_name'] == 'test'
        assert call_args['query_vector'] == query_vector
        assert call_args['limit'] == 5
        assert 'query_filter' in call_args
        assert call_args['query_filter'] is not None

@patch('qdrant_db.QdrantClient')
def test_search_without_filters(mock_qdrant_client_class):
    """Test search method without filters (backward compatibility)."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    mock_client.search.return_value = []
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Mock collection_exists to return True
    with patch.object(db, 'collection_exists', return_value=True):
        # Test search without filters
        query_vector = [0.1, 0.2, 0.3]
        
        db.search(
            query_vector=query_vector,
            limit=5
        )
        
        # Verify search was called without query_filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        
        assert 'query_filter' not in call_args

@patch('qdrant_db.QdrantClient')
def test_search_with_empty_filters(mock_qdrant_client_class):
    """Test search method with empty filters."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    mock_client.search.return_value = []
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    db = QdrantDB(config)
    
    # Mock collection_exists to return True
    with patch.object(db, 'collection_exists', return_value=True):
        # Test search with empty filters
        query_vector = [0.1, 0.2, 0.3]
        
        db.search(
            query_vector=query_vector,
            limit=5,
            filters={}
        )
        
        # Should not include query_filter for empty filters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        assert 'query_filter' not in call_args

@patch('qdrant_db.QdrantClient')
def test_payload_index_validation_missing_collection(mock_qdrant_client_class):
    """Test payload index validation when collection doesn't exist."""
    mock_client = create_mock_qdrant_client()
    
    # Mock empty collections (no collection exists)
    mock_collections = Mock()
    mock_collections.collections = []
    mock_client.get_collections.return_value = mock_collections
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test_collection'}
    
    # Should not raise exception, just log info
    db = QdrantDB(config)
    assert db is not None

@patch('qdrant_db.QdrantClient')
def test_payload_index_validation_with_indexes(mock_qdrant_client_class):
    """Test payload index validation when indexes exist."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock collection info with payload schema
    mock_collection_info = Mock()
    mock_params = Mock()
    mock_params.payload_schema = {
        'tags': Mock(),
        'author': Mock(),
        'publication_date': Mock()
    }
    mock_collection_info.config = Mock()
    mock_collection_info.config.params = mock_params
    mock_client.get_collection.return_value = mock_collection_info
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    
    # Should initialize without issues when all indexes exist
    db = QdrantDB(config)
    assert db is not None

@patch('qdrant_db.QdrantClient')
def test_payload_index_validation_missing_indexes(mock_qdrant_client_class):
    """Test payload index validation when some indexes are missing."""
    mock_client = create_mock_qdrant_client(collection_exists=True)
    
    # Mock collection info with partial payload schema (missing some indexes)
    mock_collection_info = Mock()
    mock_params = Mock()
    mock_params.payload_schema = {
        'tags': Mock(),
        # Missing 'author' and 'publication_date'
    }
    mock_collection_info.config = Mock()
    mock_collection_info.config.params = mock_params
    mock_client.get_collection.return_value = mock_collection_info
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test'}
    
    # Should initialize with warning (but not fail)
    db = QdrantDB(config)
    assert db is not None

@patch('qdrant_db.QdrantClient') 
def test_payload_index_validation_exception_handling(mock_qdrant_client_class):
    """Test payload index validation exception handling."""
    mock_client = create_mock_qdrant_client()
    mock_client.get_collection.side_effect = Exception("Qdrant connection error")
    
    mock_qdrant_client_class.return_value = mock_client
    
    config = {'host': 'localhost', 'port': 6333, 'collection_name': 'test_collection'}
    
    # Should handle exception gracefully and still initialize
    db = QdrantDB(config)
    assert db is not None