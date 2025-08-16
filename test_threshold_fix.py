#!/usr/bin/env python3
"""
Test script to verify score threshold fix for RRF multi-vector search.
"""

import sys
import os
import random
import numpy as np
from typing import List

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qdrant_db import QdrantDB
from config_manager import ConfigManager

def generate_test_vector(seed: int = None, size: int = 384) -> List[float]:
    """Generate a normalized test vector."""
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    
    vector = np.random.uniform(-1, 1, size)
    magnitude = np.linalg.norm(vector)
    return (vector / magnitude).tolist()

def test_threshold_fix():
    """Test that score thresholds work correctly for both single and multi-vector RRF."""
    print("🔧 Testing Score Threshold Fix")
    
    # Initialize database
    try:
        config_manager = ConfigManager()
        qdrant_config = config_manager.get_vector_db_config()
        
        if not qdrant_config:
            print("❌ No Qdrant configuration found")
            return
        
        db = QdrantDB(qdrant_config)
        
        if not db.test_connection():
            print("❌ Cannot connect to Qdrant")
            return
            
        print(f"✅ Connected to Qdrant collection: {db.collection_name}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return
    
    # Generate test vectors
    vector1 = generate_test_vector(seed=500)
    vector2 = generate_test_vector(seed=501)
    
    print("\n" + "="*60)
    print(" SINGLE VECTOR THRESHOLD TEST")
    print("="*60)
    
    # Test single vector with different thresholds
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = db.hybrid_search(
            dense_vectors=[vector1],
            sparse_vector=None,
            limit=5,
            score_threshold=threshold
        )
        
        if results:
            top_score = results[0].score
            print(f"Threshold {threshold}: {len(results)} results, top score: {top_score:.3f}")
        else:
            print(f"Threshold {threshold}: 0 results")
    
    print("\n" + "="*60)
    print(" MULTI VECTOR THRESHOLD TEST")
    print("="*60)
    
    # Test multi vector with different thresholds
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        results = db.hybrid_search(
            dense_vectors=[vector1, vector2],
            sparse_vector=None,
            limit=5,
            score_threshold=threshold
        )
        
        if results:
            top_score = results[0].score
            print(f"Threshold {threshold}: {len(results)} results, top score: {top_score:.3f}")
        else:
            print(f"Threshold {threshold}: 0 results")
    
    print("\n💡 Expected behavior:")
    print("- Single vector: Threshold should filter based on raw RRF scores (~0.167-0.5)")
    print("- Multi vector: Threshold should filter based on normalized scores (0.5-0.95)")
    print("- Higher thresholds should return fewer results in both cases")

if __name__ == "__main__":
    test_threshold_fix()