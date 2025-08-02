"""
Integration tests for QdrantDB with real Qdrant instance.
Tests hybrid search filtering functionality with actual data and vector operations.
"""

import sys
import os
from typing import List, Dict, Any
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qdrant_db import QdrantDB
from config_manager import ConfigManager
from qdrant_client.models import PointStruct


class TestQdrantIntegration:
    """Integration tests for QdrantDB hybrid search filtering with real Qdrant instance."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with real Qdrant configuration and test collection."""
        # Load real Qdrant configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config_manager = ConfigManager(config_path)
        cls.qdrant_config = config_manager.get('vector_db', {})
        
        # Use test collection name to avoid interfering with production data
        cls.test_collection_name = "test_hybrid_search"
        cls.qdrant_config['collection_name'] = cls.test_collection_name
        
        # Initialize QdrantDB with test configuration
        cls.qdrant_db = QdrantDB(cls.qdrant_config)
        
        # Verify connection
        if not cls.qdrant_db.test_connection():
            pytest.skip("Cannot connect to Qdrant instance for integration tests")
        
        # Create test collection and populate with test data
        cls._create_test_collection()
        cls._populate_test_data()
        
        print(f"✅ Test collection '{cls.test_collection_name}' created and populated")
    
    @classmethod
    def teardown_class(cls):
        """Clean up test collection after all tests complete."""
        try:
            if hasattr(cls, 'qdrant_db') and cls.qdrant_db.client:
                # Delete test collection
                cls.qdrant_db.client.delete_collection(cls.test_collection_name)
                print(f"🧹 Test collection '{cls.test_collection_name}' cleaned up")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up test collection: {e}")
    
    @classmethod
    def _create_test_collection(cls):
        """Create test collection with appropriate vector dimensions and payload indexes."""
        # Use 384 dimensions to match sentence-transformers default
        vector_size = 384
        
        # Delete collection if it already exists (cleanup from previous failed runs)
        try:
            if cls.qdrant_db.collection_exists():
                cls.qdrant_db.client.delete_collection(cls.test_collection_name)
                print(f"🧹 Deleted existing test collection '{cls.test_collection_name}'")
        except Exception:
            pass  # Collection might not exist, which is fine
        
        # Create fresh test collection
        success = cls.qdrant_db.create_collection(vector_size)
        if not success:
            raise RuntimeError(f"Failed to create test collection '{cls.test_collection_name}'")
        
        # Create payload indexes for filtering
        cls._create_payload_indexes()
        
        print(f"📦 Created test collection '{cls.test_collection_name}' with {vector_size}D vectors and payload indexes")
    
    @classmethod
    def _create_payload_indexes(cls):
        """Create payload indexes for filterable fields."""
        from qdrant_client.models import CreateFieldIndex, PayloadSchemaType
        
        try:
            # Create index for author field (keyword type for exact matches)
            cls.qdrant_db.client.create_payload_index(
                collection_name=cls.test_collection_name,
                field_name="author",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Create index for tags field (keyword type for array matching)
            cls.qdrant_db.client.create_payload_index(
                collection_name=cls.test_collection_name,
                field_name="tags",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Create index for publication_date field
            cls.qdrant_db.client.create_payload_index(
                collection_name=cls.test_collection_name,
                field_name="publication_date",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Create index for title field (for completeness)
            cls.qdrant_db.client.create_payload_index(
                collection_name=cls.test_collection_name,
                field_name="title",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            print("🔍 Created payload indexes for filterable fields")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create payload indexes: {e}")
            # Don't fail the test setup - just log the warning
    
    @classmethod
    def _populate_test_data(cls):
        """Populate test collection with carefully crafted test documents."""
        # Create diverse test documents for comprehensive filter testing
        test_documents = cls._create_test_documents()
        
        # Convert to Qdrant points
        points = []
        for doc in test_documents:
            point = PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload=doc["payload"]
            )
            points.append(point)
        
        # Insert all points into collection
        cls.qdrant_db.client.upsert(
            collection_name=cls.test_collection_name,
            points=points
        )
        
        print(f"📄 Inserted {len(points)} test documents into collection")
    
    @classmethod
    def _create_test_documents(cls) -> List[Dict[str, Any]]:
        """Create test documents with known metadata for predictable testing."""
        # Generate dummy vectors (384 dimensions, normalized)
        import random
        random.seed(42)  # Reproducible test vectors
        
        def generate_dummy_vector() -> List[float]:
            """Generate normalized dummy vector for testing."""
            vector = [random.uniform(-1, 1) for _ in range(384)]
            # Simple normalization
            magnitude = sum(x * x for x in vector) ** 0.5
            return [x / magnitude for x in vector]
        
        return [
            {
                "id": 1,
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "Introduction to Python Programming",
                    "author": "John Smith",
                    "tags": ["python", "programming", "beginner"],
                    "publication_date": "2023",
                    "content": "A comprehensive guide to Python programming for beginners.",
                    "doc_name": "doc1"  # Keep doc name in payload for test assertions
                }
            },
            {
                "id": 2, 
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "Machine Learning with Python",
                    "author": "Jane Doe",
                    "tags": ["python", "machine learning", "ai"],
                    "publication_date": "2024",
                    "content": "Advanced machine learning techniques using Python.",
                    "doc_name": "doc2"
                }
            },
            {
                "id": 3,
                "vector": generate_dummy_vector(), 
                "payload": {
                    "title": "JavaScript Fundamentals",
                    "author": "John Smith",  # Same author as doc1, different topic
                    "tags": ["javascript", "web development", "frontend"],
                    "publication_date": "2023",  # Same year as doc1
                    "content": "Essential JavaScript concepts for web development.",
                    "doc_name": "doc3"
                }
            },
            {
                "id": 4,
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "Advanced Python Techniques", 
                    "author": "Alice Johnson",
                    "tags": ["python", "advanced", "optimization"],
                    "publication_date": "2024",  # Same year as doc2
                    "content": "Advanced Python programming patterns and optimization.",
                    "doc_name": "doc4"
                }
            },
            {
                "id": 5,
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "Deep Learning Fundamentals",
                    "author": "Bob Wilson",
                    "tags": ["machine learning", "deep learning", "neural networks"],
                    "publication_date": "2024",
                    "content": "Introduction to deep learning and neural networks.",
                    "doc_name": "doc5"
                }
            },
            {
                "id": 6,
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "Web Development with React",
                    "author": "Jane Doe",  # Same author as doc2, different topic
                    "tags": ["javascript", "react", "web development"],
                    "publication_date": "2025",
                    "content": "Building modern web applications with React.",
                    "doc_name": "doc6"
                }
            },
            {
                "id": 7,
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "Data Science with Python",
                    "author": "John Smith",  # Same author as doc1, doc3
                    "tags": ["python", "data science", "statistics"],
                    "publication_date": "2025",
                    "content": "Comprehensive data science techniques using Python.",
                    "doc_name": "doc7"
                }
            },
            {
                "id": 8,
                "vector": generate_dummy_vector(),
                "payload": {
                    "title": "AI Ethics and Society",
                    "author": "Dr. Sarah Chen",
                    "tags": ["ai", "ethics", "society"],
                    "publication_date": "2024",
                    "content": "Exploring the ethical implications of artificial intelligence.",
                    "doc_name": "doc8"
                }
            }
        ]
    
    def _generate_dummy_query_vector(self) -> List[float]:
        """Generate a dummy query vector for search testing."""
        import random
        random.seed(123)  # Different seed from document vectors
        vector = [random.uniform(-1, 1) for _ in range(384)]
        magnitude = sum(x * x for x in vector) ** 0.5
        return [x / magnitude for x in vector]
    
    @pytest.mark.integration
    def test_collection_setup_and_data_validation(self):
        """Verify test collection is properly set up with expected data."""
        # Check collection exists
        assert self.qdrant_db.collection_exists(), "Test collection should exist"
        
        # Get collection info
        info = self.qdrant_db.get_collection_info()
        assert info is not None, "Should be able to get collection info"
        assert info['name'] == self.test_collection_name
        assert info['points_count'] == 8, "Should have 8 test documents"
        
        # Verify we can scroll through collection
        points = self.qdrant_db.scroll_collection(limit=10)
        assert len(points) == 8, "Should retrieve all 8 test documents"
        
        # Verify payload structure
        sample_point = points[0]
        required_fields = ['title', 'author', 'tags', 'publication_date', 'content']
        for field in required_fields:
            assert field in sample_point.payload, f"Payload should contain {field} field"
        
        print("✅ Test collection setup and data validation passed")
    
    @pytest.mark.integration
    def test_author_filter_single_match(self):
        """Test filtering by author with single result expected."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by "Alice Johnson" (should match only doc4)
        filters = {"author": "Alice Johnson"}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 1, f"Should find exactly 1 document by Alice Johnson, got {len(results)}"
        assert results[0].payload["doc_name"] == "doc4", f"Should return doc4, got {results[0].payload.get('doc_name')}"
        assert results[0].payload["author"] == "Alice Johnson"
        assert results[0].payload["title"] == "Advanced Python Techniques"
        
        print(f"✅ Author filter test passed: Found {len(results)} document(s) by Alice Johnson")
    
    @pytest.mark.integration
    def test_author_filter_multiple_matches(self):
        """Test filtering by author with multiple results expected.""" 
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by "John Smith" (should match doc1, doc3, doc7)
        filters = {"author": "John Smith"}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 3, f"Should find exactly 3 documents by John Smith, got {len(results)}"
        
        # Verify all results are by John Smith
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc3", "doc7"}
        assert result_doc_names == expected_doc_names, f"Should return doc1, doc3, doc7, got {result_doc_names}"
        
        for result in results:
            assert result.payload["author"] == "John Smith"
        
        print(f"✅ Author filter test passed: Found {len(results)} documents by John Smith")
    
    @pytest.mark.integration
    def test_tags_filter_single_tag(self):
        """Test filtering by single tag."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by "react" tag (should match only doc6)
        filters = {"tags": ["react"]}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 1, f"Should find exactly 1 document with 'react' tag, got {len(results)}"
        assert results[0].payload["doc_name"] == "doc6", f"Should return doc6, got {results[0].payload.get('doc_name')}"
        assert "react" in results[0].payload["tags"]
        
        print(f"✅ Tags filter test passed: Found {len(results)} document(s) with 'react' tag")
    
    @pytest.mark.integration
    def test_tags_filter_common_tag(self):
        """Test filtering by common tag that appears in multiple documents."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by "python" tag (should match doc1, doc2, doc4, doc7)
        filters = {"tags": ["python"]}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 4, f"Should find exactly 4 documents with 'python' tag, got {len(results)}"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc2", "doc4", "doc7"}
        assert result_doc_names == expected_doc_names, f"Should return doc1, doc2, doc4, doc7, got {result_doc_names}"
        
        # Verify all results contain python tag
        for result in results:
            assert "python" in result.payload["tags"], f"Result {result.payload['doc_name']} should contain 'python' tag"
        
        print(f"✅ Tags filter test passed: Found {len(results)} documents with 'python' tag")
    
    @pytest.mark.integration
    def test_tags_filter_multiple_tags(self):
        """Test filtering with multiple tags (MatchAny behavior)."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by ["python", "javascript"] (should match docs with either tag)
        filters = {"tags": ["python", "javascript"]}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Should match: doc1,doc2,doc3,doc4,doc6,doc7 (all docs with python OR javascript)
        assert len(results) == 6, f"Should find 6 documents with python OR javascript tags, got {len(results)}"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc2", "doc3", "doc4", "doc6", "doc7"}
        assert result_doc_names == expected_doc_names, f"Should return docs with python or javascript, got {result_doc_names}"
        
        # Verify each result contains at least one of the target tags
        for result in results:
            has_target_tag = any(tag in result.payload["tags"] for tag in ["python", "javascript"])
            assert has_target_tag, f"Result {result.payload['doc_name']} should contain 'python' or 'javascript' tag"
        
        print(f"✅ Multiple tags filter test passed: Found {len(results)} documents")
    
    @pytest.mark.integration
    def test_publication_date_filter(self):
        """Test filtering by publication date."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by "2024" (should match doc2, doc4, doc5, doc8)
        filters = {"publication_date": "2024"}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 4, f"Should find exactly 4 documents from 2024, got {len(results)}"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc2", "doc4", "doc5", "doc8"}
        assert result_doc_names == expected_doc_names, f"Should return doc2, doc4, doc5, doc8, got {result_doc_names}"
        
        # Verify all results are from 2024
        for result in results:
            assert result.payload["publication_date"] == "2024"
        
        print(f"✅ Publication date filter test passed: Found {len(results)} documents from 2024")
    
    @pytest.mark.integration
    def test_combined_filter_author_and_tags(self):
        """Test combined filtering by author and tags."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by John Smith AND python tag (should match doc1, doc7)
        filters = {"author": "John Smith", "tags": ["python"]}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 2, f"Should find exactly 2 documents by John Smith with python tag, got {len(results)}"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc7"}
        assert result_doc_names == expected_doc_names, f"Should return doc1, doc7, got {result_doc_names}"
        
        # Verify all results match both criteria
        for result in results:
            assert result.payload["author"] == "John Smith"
            assert "python" in result.payload["tags"]
        
        print(f"✅ Combined author+tags filter test passed: Found {len(results)} documents")
    
    @pytest.mark.integration
    def test_combined_filter_author_and_date(self):
        """Test combined filtering by author and publication date."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by John Smith AND 2023 (should match doc1, doc3)
        filters = {"author": "John Smith", "publication_date": "2023"}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 2, f"Should find exactly 2 documents by John Smith from 2023, got {len(results)}"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc3"}
        assert result_doc_names == expected_doc_names, f"Should return doc1, doc3, got {result_doc_names}"
        
        # Verify all results match both criteria
        for result in results:
            assert result.payload["author"] == "John Smith"
            assert result.payload["publication_date"] == "2023"
        
        print(f"✅ Combined author+date filter test passed: Found {len(results)} documents")
    
    @pytest.mark.integration
    def test_combined_filter_tags_and_date(self):
        """Test combined filtering by tags and publication date."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by python tag AND 2024 (should match doc2, doc4)
        filters = {"tags": ["python"], "publication_date": "2024"}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 2, f"Should find exactly 2 documents with python tag from 2024, got {len(results)}"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc2", "doc4"}
        assert result_doc_names == expected_doc_names, f"Should return doc2, doc4, got {result_doc_names}"
        
        # Verify all results match both criteria
        for result in results:
            assert "python" in result.payload["tags"]
            assert result.payload["publication_date"] == "2024"
        
        print(f"✅ Combined tags+date filter test passed: Found {len(results)} documents")
    
    @pytest.mark.integration
    def test_combined_filter_all_criteria(self):
        """Test combined filtering with author, tags, and date (most restrictive)."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by John Smith AND python tag AND 2023 (should match only doc1)
        filters = {
            "author": "John Smith",
            "tags": ["python"], 
            "publication_date": "2023"
        }
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Validate results
        assert len(results) == 1, f"Should find exactly 1 document matching all criteria, got {len(results)}"
        assert results[0].payload["doc_name"] == "doc1", f"Should return doc1, got {results[0].payload.get('doc_name')}"
        
        # Verify result matches all criteria
        result = results[0]
        assert result.payload["author"] == "John Smith"
        assert "python" in result.payload["tags"]
        assert result.payload["publication_date"] == "2023"
        assert result.payload["title"] == "Introduction to Python Programming"
        
        print(f"✅ All criteria filter test passed: Found {len(results)} document matching all filters")
    
    @pytest.mark.integration
    def test_filter_no_matches(self):
        """Test filtering with criteria that should return no results."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test: Filter by non-existent author
        filters = {"author": "NonExistent Author"}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        assert len(results) == 0, f"Should find no documents by non-existent author, got {len(results)}"
        
        # Test: Filter by non-existent tag
        filters = {"tags": ["nonexistent-tag"]}
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        assert len(results) == 0, f"Should find no documents with non-existent tag, got {len(results)}"
        
        # Test: Filter by impossible combination (author exists but not with this tag)
        filters = {"author": "Alice Johnson", "tags": ["javascript"]}  # Alice doesn't have JS docs
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        assert len(results) == 0, f"Should find no documents matching impossible combination, got {len(results)}"
        
        print("✅ No matches filter test passed: All impossible filters returned 0 results")
    
    @pytest.mark.integration
    def test_search_without_filters_baseline(self):
        """Test search without filters to establish baseline (should return all documents)."""
        query_vector = self._generate_dummy_query_vector()
        
        # Search without any filters
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10
        )
        
        # Should return all 8 test documents
        assert len(results) == 8, f"Should find all 8 documents without filters, got {len(results)}"
        
        # Verify all documents are present
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8"}
        assert result_doc_names == expected_doc_names, f"Should return all document names, got {result_doc_names}"
        
        # Results should be ordered by vector similarity score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, "Results should be ordered by decreasing score"
        
        print(f"✅ Baseline search test passed: Found all {len(results)} documents without filters")
    
    @pytest.mark.integration
    def test_search_with_score_threshold(self):
        """Test search with score threshold combined with filters."""
        query_vector = self._generate_dummy_query_vector()
        
        # First, get baseline scores
        baseline_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters={"author": "John Smith"}
        )
        
        if len(baseline_results) > 1:
            # Use a score threshold that should filter out some results
            threshold = baseline_results[1].score  # Use second-highest score as threshold
            
            filtered_results = self.qdrant_db.search(
                query_vector=query_vector,
                limit=10,
                score_threshold=threshold,
                filters={"author": "John Smith"}
            )
            
            # Should have fewer results due to score threshold
            assert len(filtered_results) <= len(baseline_results), "Score threshold should reduce results"
            
            # All results should meet score threshold
            for result in filtered_results:
                assert result.score >= threshold, f"Result score {result.score} should be >= threshold {threshold}"
            
            print(f"✅ Score threshold test passed: {len(filtered_results)} results above threshold")
        else:
            print("⚠️ Score threshold test skipped: insufficient baseline results")
    
    @pytest.mark.integration
    def test_empty_filters_same_as_no_filters(self):
        """Test that empty filters dict behaves same as no filters."""
        query_vector = self._generate_dummy_query_vector()
        
        # Search with no filters
        no_filters_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10
        )
        
        # Search with empty filters dict
        empty_filters_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters={}
        )
        
        # Should return identical results
        assert len(no_filters_results) == len(empty_filters_results), "Empty filters should behave like no filters"
        
        no_filter_ids = [result.id for result in no_filters_results]
        empty_filter_ids = [result.id for result in empty_filters_results]
        assert no_filter_ids == empty_filter_ids, "Results order should be identical"
        
        print("✅ Empty filters test passed: Behaves identically to no filters")
    
    @pytest.mark.integration
    def test_filters_with_empty_values_ignored(self):
        """Test that filters with empty values are ignored."""
        query_vector = self._generate_dummy_query_vector()
        
        # Search with mixed empty and valid filters
        filters = {
            "author": "John Smith",  # Valid filter
            "tags": [],             # Empty list - should be ignored
            "publication_date": "", # Empty string - should be ignored
            "title": None           # None value - should be ignored
        }
        
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        # Should behave same as author-only filter
        author_only_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters={"author": "John Smith"}
        )
        
        assert len(results) == len(author_only_results), "Empty filter values should be ignored"
        assert len(results) == 3, "Should find 3 documents by John Smith"
        
        result_doc_names = {result.payload["doc_name"] for result in results}
        expected_doc_names = {"doc1", "doc3", "doc7"}
        assert result_doc_names == expected_doc_names, "Should only apply non-empty filters"
        
        print("✅ Empty filter values test passed: Empty values properly ignored")
    
    @pytest.mark.integration
    def test_case_sensitive_filtering(self):
        """Test that filtering is case-sensitive for exact matches."""
        query_vector = self._generate_dummy_query_vector()
        
        # Test case-sensitive author matching
        filters = {"author": "john smith"}  # lowercase, should not match "John Smith"
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        assert len(results) == 0, "Case-sensitive author filter should return no results"
        
        # Test case-sensitive tag matching
        filters = {"tags": ["Python"]}  # Capital P, should not match "python"
        results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,
            filters=filters
        )
        
        assert len(results) == 0, "Case-sensitive tag filter should return no results"
        
        print("✅ Case sensitivity test passed: Filters are properly case-sensitive")
    
    @pytest.mark.integration
    def test_vector_search_quality_with_filters(self):
        """Test that vector search quality is maintained when filters are applied."""
        query_vector = self._generate_dummy_query_vector()
        
        # Search with author filter
        filtered_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=5,
            filters={"author": "John Smith"}
        )
        
        # Verify results are still ordered by similarity score
        assert len(filtered_results) >= 2, "Should have multiple results to test ordering"
        
        for i in range(len(filtered_results) - 1):
            current_score = filtered_results[i].score
            next_score = filtered_results[i + 1].score
            assert current_score >= next_score, f"Results should be ordered by score: {current_score} >= {next_score}"
        
        # Verify all results match the filter
        for result in filtered_results:
            assert result.payload["author"] == "John Smith"
        
        # Verify we get meaningful similarity scores (not all identical)
        scores = [result.score for result in filtered_results]
        assert not all(s == scores[0] for s in scores), "Should have varied similarity scores"
        
        print(f"✅ Vector search quality test passed: {len(filtered_results)} results properly ordered by similarity")
    
    @pytest.mark.integration
    def test_limit_parameter_with_filters(self):
        """Test that limit parameter works correctly with filters."""
        query_vector = self._generate_dummy_query_vector()
        
        # Get all results for John Smith
        all_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=10,  # More than available
            filters={"author": "John Smith"}
        )
        
        # Get limited results
        limited_results = self.qdrant_db.search(
            query_vector=query_vector,
            limit=2,   # Less than available
            filters={"author": "John Smith"}
        )
        
        # Verify limit is respected
        assert len(all_results) == 3, "Should have 3 total documents by John Smith"
        assert len(limited_results) == 2, "Should limit results to 2"
        
        # Verify limited results are the top-scoring ones
        for i in range(len(limited_results)):
            assert limited_results[i].payload["doc_name"] == all_results[i].payload["doc_name"], "Limited results should be top-scoring"
            assert limited_results[i].score == all_results[i].score, "Scores should match"
        
        print("✅ Limit parameter test passed: Limit properly applied with filters")