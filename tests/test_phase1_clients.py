"""
Phase 1 Integration Tests for RAG CLI Chatbot clients.
Tests the EmbeddingClient, QdrantDB, and LLMClient with sentence-transformers (local).
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embedding_client import EmbeddingClient
from qdrant_db import QdrantDB
from llm_client import LLMClient
from logging_config import setup_logging

def test_embedding_client_sentence_transformers():
    """Test EmbeddingClient with sentence-transformers (works offline)."""
    
    config = {
        'provider': 'sentence_transformers',
        'sentence_transformers': {
            'model': 'all-MiniLM-L6-v2',
            'device': 'cpu'
        }
    }
    
    try:
        client = EmbeddingClient(config)
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        client.get_embedding(test_text)
        
        # Verify embedding generation
        
        # Test embedding dimensions
        expected_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        actual_dim = client.get_embedding_dimension()
        
        # Verify embedding dimensions
        
        assert actual_dim == expected_dim, f"Expected {expected_dim} dimensions, got {actual_dim}"
            
    except Exception as e:
        pytest.fail(f"EmbeddingClient test failed: {e}")

def test_qdrant_client_mock():
    """Test QdrantDB client initialization (mock connection test)."""
    
    config = {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'test_collection',
        'distance_metric': 'cosine'
    }
    
    try:
        client = QdrantDB(config)
        # Note: We can't test actual connection without running Qdrant server
        # But we can test that the client initializes without errors
        assert client is not None
        
    except Exception as e:
        pytest.fail(f"QdrantDB test failed: {e}")

def test_llm_client_mock():
    """Test LLMClient initialization (mock test since we don't have Ollama running)."""
    
    config = {
        'provider': 'ollama',
        'model': 'llama3.2',
        'base_url': 'http://localhost:11434',
        'temperature': 0.7,
        'max_tokens': 2048
    }
    
    try:
        client = LLMClient(config)
        # Note: We can't test actual LLM calls without running Ollama server
        # But we can test that the client initializes without errors  
        assert client is not None
        
    except Exception as e:
        pytest.fail(f"LLMClient test failed: {e}")

@pytest.fixture(scope="module", autouse=True)
def setup_logging_fixture():
    """Setup logging for all tests in this module."""
    setup_logging({'level': 'INFO'})