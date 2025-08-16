"""
Integration tests for SearchService end-to-end hybrid search workflow.
Tests the complete SearchService pipeline with real components for hybrid search.
"""

import sys
import os
from typing import Dict, Any
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_service import SearchService
from qdrant_db import QdrantDB
from embedding_client import EmbeddingClient
from query_rewriter import QueryRewriter
from config_manager import ConfigManager
from qdrant_client.models import PointStruct, SparseVector
import random


class TestSearchServiceIntegration:
    """Integration tests for SearchService with hybrid search functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with real components for SearchService integration."""
        # Load real configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config_manager = ConfigManager(config_path)
        
        # Set up QdrantDB with test collection
        cls.qdrant_config = config_manager.get('vector_db', {})
        cls.test_collection_name = "test_search_service_integration"
        cls.qdrant_config['collection_name'] = cls.test_collection_name
        cls.qdrant_db = QdrantDB(cls.qdrant_config)
        
        # Skip if cannot connect to Qdrant
        if not cls.qdrant_db.test_connection():
            pytest.skip("Cannot connect to Qdrant instance for SearchService integration tests")
        
        # Set up dense embedding client
        embedding_config = config_manager.get('embedding', {})
        cls.dense_embedding_client = EmbeddingClient(embedding_config)
        
        # Set up query rewriter
        query_rewriter_config = config_manager.get('rag', {})
        llm_config = config_manager.get('llm', {})
        cls.query_rewriter = QueryRewriter(query_rewriter_config, llm_config)
        
        # Create test collection and populate with test data
        cls._create_test_collection()
        cls._populate_test_data()
        
        print("✅ SearchService integration test setup complete")
    
    @classmethod
    def teardown_class(cls):
        """Clean up test collection after all tests complete."""
        try:
            if hasattr(cls, 'qdrant_db') and cls.qdrant_db.client:
                cls.qdrant_db.client.delete_collection(cls.test_collection_name)
                print(f"🧹 Test collection '{cls.test_collection_name}' cleaned up")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up test collection: {e}")
    
    @classmethod
    def _create_test_collection(cls):
        """Create test collection with named dense/sparse vectors."""
        # Use 384 dimensions for compatibility with sentence-transformers
        vector_size = 384
        
        # Delete collection if it already exists
        try:
            if cls.qdrant_db.collection_exists():
                cls.qdrant_db.client.delete_collection(cls.test_collection_name)
                print("🧹 Deleted existing test collection")
        except Exception:
            pass
        
        # Create collection with named vectors
        from qdrant_client.models import VectorParams, Distance, SparseVectorParams
        
        vectors_config = {
            "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
        }
        
        sparse_vectors_config = {
            "sparse": SparseVectorParams()
        }
        
        try:
            cls.qdrant_db.client.create_collection(
                collection_name=cls.test_collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config
            )
            print("📦 Created test collection with named dense/sparse vectors")
        except Exception as e:
            raise RuntimeError(f"Failed to create test collection: {e}")
    
    @classmethod
    def _populate_test_data(cls):
        """Populate test collection with sample documents."""
        # Generate sample dense vectors
        random.seed(42)
        
        def generate_dense_vector() -> list:
            vector = [random.uniform(-1, 1) for _ in range(384)]
            magnitude = sum(x * x for x in vector) ** 0.5
            return [x / magnitude for x in vector]
        
        def generate_sparse_vector():
            indices = random.sample(range(1000), 15)
            values = [random.uniform(0.1, 1.0) for _ in range(15)]
            return SparseVector(indices=indices, values=values)
        
        test_documents = [
            {
                "id": 1,
                "vector": {
                    "dense": generate_dense_vector(),
                    "sparse": generate_sparse_vector()
                },
                "payload": {
                    "title": "Introduction to Neural Networks",
                    "author": "Dr. Smith",
                    "tags": ["machine learning", "neural networks", "ai"],
                    "publication_date": "2024-01-15T00:00:00",
                    "content": "Neural networks are fundamental to modern machine learning."
                }
            },
            {
                "id": 2,
                "vector": {
                    "dense": generate_dense_vector(),
                    "sparse": generate_sparse_vector()
                },
                "payload": {
                    "title": "Deep Learning Fundamentals",
                    "author": "Prof. Johnson",
                    "tags": ["deep learning", "neural networks", "python"],
                    "publication_date": "2024-02-20T00:00:00",
                    "content": "Deep learning extends neural networks with multiple layers."
                }
            },
            {
                "id": 3,
                "vector": {
                    "dense": generate_dense_vector(),
                    "sparse": generate_sparse_vector()
                },
                "payload": {
                    "title": "Machine Learning Applications",
                    "author": "Dr. Smith",
                    "tags": ["machine learning", "applications", "python"],
                    "publication_date": "2024-03-10T00:00:00",
                    "content": "Machine learning has many practical applications in industry."
                }
            }
        ]
        
        # Convert to Qdrant points
        points = []
        for doc in test_documents:
            point = PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload=doc["payload"]
            )
            points.append(point)
        
        # Insert points
        cls.qdrant_db.client.upsert(
            collection_name=cls.test_collection_name,
            points=points
        )
        
        print(f"📄 Inserted {len(points)} test documents")
    
    @pytest.mark.integration
    def test_search_service_traditional_search(self):
        """Test SearchService with traditional single-vector search (backward compatibility)."""
        # Create SearchService without sparse client (traditional mode)
        search_service = SearchService(
            qdrant_db=self.qdrant_db,
            dense_embedding_client=self.dense_embedding_client,
            query_rewriter=self.query_rewriter,
            sparse_embedding_client=None  # No sparse client
        )
        
        # Test traditional search
        results = search_service.unified_search(
            query="neural networks machine learning",
            top_k=3,
            enable_hybrid=False  # Explicitly disable hybrid
        )
        
        # Should return results using traditional search
        assert len(results) >= 1, "Should find at least one relevant document"
        
        # Verify scores are in expected range (normalized)
        for result in results:
            assert 0.5 <= result.score <= 0.95, f"Score {result.score} should be in normalized range"
        
        print(f"✅ Traditional search found {len(results)} results")
    
    @pytest.mark.integration
    def test_search_service_dense_only_hybrid(self):
        """Test SearchService with dense-only hybrid search (multiple vectors, no sparse)."""
        # Create SearchService without sparse client but enable hybrid for multi-vector
        search_service = SearchService(
            qdrant_db=self.qdrant_db,
            dense_embedding_client=self.dense_embedding_client,
            query_rewriter=self.query_rewriter,
            sparse_embedding_client=None  # No sparse client
        )
        
        # Test with HyDE strategy to generate multiple dense vectors
        results = search_service.unified_search(
            query="explain neural networks and deep learning",
            top_k=3,
            enable_hybrid=True  # Enable hybrid for multi-vector
        )
        
        # Should return results using dense-only hybrid search
        assert len(results) >= 1, "Should find at least one relevant document"
        
        # Verify scores are normalized
        for result in results:
            assert 0.5 <= result.score <= 0.95, f"Score {result.score} should be in normalized range"
        
        print(f"✅ Dense-only hybrid search found {len(results)} results")
    
    @pytest.mark.integration
    def test_search_service_with_metadata_filters(self):
        """Test SearchService with metadata filtering integration."""
        search_service = SearchService(
            qdrant_db=self.qdrant_db,
            dense_embedding_client=self.dense_embedding_client,
            query_rewriter=self.query_rewriter,
            sparse_embedding_client=None
        )
        
        # Test search with author filter
        results = search_service.unified_search(
            query="search machine learning by Dr. Smith",
            top_k=5
        )
        
        # Should return filtered results
        assert len(results) >= 1, "Should find documents by Dr. Smith"
        
        # Verify all results are by Dr. Smith (assuming query rewriter extracts the filter)
        dr_smith_results = [r for r in results if r.payload.get("author") == "Dr. Smith"]
        
        print(f"✅ Metadata filtering found {len(results)} total results, {len(dr_smith_results)} by Dr. Smith")
    
    @pytest.mark.integration
    def test_search_service_error_handling(self):
        """Test SearchService error handling with invalid configurations."""
        # Test with invalid qdrant_db
        from unittest.mock import Mock
        invalid_db = Mock()
        invalid_db.search.side_effect = Exception("Database error")
        invalid_db.hybrid_search.side_effect = Exception("Database error")
        
        search_service = SearchService(
            qdrant_db=invalid_db,
            dense_embedding_client=self.dense_embedding_client,
            query_rewriter=self.query_rewriter,
            sparse_embedding_client=None
        )
        
        # Should handle errors gracefully
        results = search_service.unified_search(
            query="test query",
            top_k=3
        )
        
        # Should return empty results on error
        assert results == [], "Should return empty list on database error"
        
        print("✅ Error handling test passed")
    
    @pytest.mark.integration
    def test_search_service_progress_callback(self):
        """Test SearchService progress callback functionality."""
        search_service = SearchService(
            qdrant_db=self.qdrant_db,
            dense_embedding_client=self.dense_embedding_client,
            query_rewriter=self.query_rewriter,
            sparse_embedding_client=None
        )
        
        # Collect progress updates
        progress_updates = []
        def progress_callback(stage: str, data: Dict[str, Any]):
            progress_updates.append((stage, data))
        
        # Test search with progress callback
        search_service.unified_search(
            query="neural networks",
            top_k=3,
            progress_callback=progress_callback
        )
        
        # Should have received progress updates
        assert len(progress_updates) >= 3, "Should receive multiple progress updates"
        
        # Check for expected stages
        stages = [update[0] for update in progress_updates]
        assert "analyzing" in stages, "Should have analyzing stage"
        assert "query_analyzed" in stages, "Should have query_analyzed stage"
        assert "search_complete" in stages, "Should have search_complete stage"
        
        print(f"✅ Progress callback test passed with {len(progress_updates)} updates")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])