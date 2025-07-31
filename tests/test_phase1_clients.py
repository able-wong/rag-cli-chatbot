"""
Phase 1 Integration Tests for RAG CLI Chatbot clients.
Tests the EmbeddingClient, QdrantDB, and LLMClient with sentence-transformers (local).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_client import EmbeddingClient
from src.qdrant_db import QdrantDB
from src.llm_client import LLMClient
from src.config_manager import ConfigManager
from src.logging_config import setup_logging

def test_embedding_client_sentence_transformers():
    """Test EmbeddingClient with sentence-transformers (works offline)."""
    print("Testing EmbeddingClient with sentence-transformers...")
    
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
        embedding = client.get_embedding(test_text)
        
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        
        # Test embedding dimensions
        expected_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        actual_dim = client.get_embedding_dimension()
        
        print(f"✓ Embedding dimension: {actual_dim} (expected: {expected_dim})")
        
        if actual_dim == expected_dim:
            print("✓ EmbeddingClient test PASSED")
            return True
        else:
            print("✗ EmbeddingClient test FAILED - dimension mismatch")
            return False
            
    except Exception as e:
        print(f"✗ EmbeddingClient test FAILED: {e}")
        return False

def test_qdrant_client_mock():
    """Test QdrantDB client initialization (mock connection test)."""
    print("Testing QdrantDB client initialization...")
    
    config = {
        'host': 'localhost',
        'port': 6333,
        'collection_name': 'test_collection',
        'distance_metric': 'cosine'
    }
    
    try:
        client = QdrantDB(config)
        print("✓ QdrantDB client initialized successfully")
        
        # Note: We can't test actual connection without running Qdrant server
        # But we can test that the client initializes without errors
        print("✓ QdrantDB test PASSED (initialization only)")
        return True
        
    except Exception as e:
        print(f"✗ QdrantDB test FAILED: {e}")
        return False

def test_llm_client_mock():
    """Test LLMClient initialization (mock test since we don't have Ollama running)."""
    print("Testing LLMClient initialization...")
    
    config = {
        'provider': 'ollama',
        'model': 'llama3.2',
        'base_url': 'http://localhost:11434',
        'temperature': 0.7,
        'max_tokens': 2048
    }
    
    try:
        client = LLMClient(config)
        print("✓ LLMClient initialized successfully")
        
        # Note: We can't test actual LLM calls without running Ollama server
        # But we can test that the client initializes without errors  
        print("✓ LLMClient test PASSED (initialization only)")
        return True
        
    except Exception as e:
        print(f"✗ LLMClient test FAILED: {e}")
        return False

def main():
    """Run all Phase 1 client tests."""
    print("=== Phase 1 Client Tests ===")
    print()
    
    # Setup logging
    setup_logging({'level': 'INFO'})
    
    results = []
    
    # Test EmbeddingClient (this should work fully)
    results.append(test_embedding_client_sentence_transformers())
    print()
    
    # Test QdrantDB (initialization only)
    results.append(test_qdrant_client_mock())
    print()
    
    # Test LLMClient (initialization only)
    results.append(test_llm_client_mock())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All Phase 1 tests PASSED!")
        return True
    else:
        print("✗ Some Phase 1 tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)