#!/usr/bin/env python3
"""
Demo script to show QueryRewriter behavior with hybrid search ON vs OFF.
This script demonstrates the difference in parsed output.
"""

import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter
from llm_client import LLMClient
from config_manager import ConfigManager

def create_query_rewriter(use_hybrid_search: bool = False) -> QueryRewriter:
    """Create QueryRewriter with hybrid search setting."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config_manager = ConfigManager(config_path)
    
    # Initialize LLM client
    llm_config = config_manager.get_llm_config()
    llm_client = LLMClient(llm_config)
    
    # Create query rewriter config with hybrid search setting
    query_rewriter_config = config_manager.get('query_rewriter', {})
    query_rewriter_config['trigger_phrase'] = config_manager.get('rag.trigger_phrase', '@knowledgebase')
    query_rewriter_config['use_hybrid_search'] = use_hybrid_search  # Override the setting
    
    return QueryRewriter(llm_client, query_rewriter_config)

def demo_scenario(query_rewriter: QueryRewriter, user_query: str, scenario_name: str, hybrid_enabled: bool):
    """Demo a single scenario."""
    print(f"\n{'='*60}")
    print(f"🔍 {scenario_name}")
    print(f"Hybrid Search: {'ENABLED' if hybrid_enabled else 'DISABLED'}")
    print(f"{'='*60}")
    print(f"User Input: {user_query}")
    print("-" * 60)
    
    try:
        result = query_rewriter.transform_query(user_query)
        
        print("Parsed Output:")
        print(f"  search_rag: {result['search_rag']}")
        print(f"  embedding_source_text: \"{result['embedding_source_text']}\"")
        print(f"  llm_query: \"{result['llm_query']}\"")
        
        if 'filters' in result:
            if result['filters']:
                print(f"  filters: {json.dumps(result['filters'], indent=4)}")
            else:
                print("  filters: {}")
        else:
            print("  filters: (not present)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run the demonstration."""
    print("🤖 QueryRewriter Hybrid Search Demonstration")
    print("This demo shows how hybrid search affects query parsing.")
    
    # Test scenarios
    scenarios = [
        "@knowledgebase papers by John Smith about machine learning",
        "@knowledgebase articles from 2023",
        "@knowledgebase search on vibe coding from John Wong published in March 2025, then explain what is vibe coding, and pros/cons",
        "@knowledgebase What is Python programming?",
        "Tell me more about the benefits",  # Non-RAG query
    ]
    
    print("\n" + "="*80)
    print("PART 1: HYBRID SEARCH DISABLED (Current Default)")
    print("="*80)
    
    # Create QueryRewriter with hybrid search OFF
    qr_hybrid_off = create_query_rewriter(use_hybrid_search=False)
    
    for i, query in enumerate(scenarios, 1):
        demo_scenario(qr_hybrid_off, query, f"Scenario {i}", hybrid_enabled=False)
    
    print("\n" + "="*80)
    print("PART 2: HYBRID SEARCH ENABLED")
    print("="*80)
    
    # Create QueryRewriter with hybrid search ON
    qr_hybrid_on = create_query_rewriter(use_hybrid_search=True)
    
    for i, query in enumerate(scenarios, 1):
        demo_scenario(qr_hybrid_on, query, f"Scenario {i}", hybrid_enabled=True)
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES:")
    print("="*80)
    print("1. HYBRID SEARCH OFF:")
    print("   - No 'filters' field in output")
    print("   - Backward compatible with existing code")
    print("   - Simple RAG search without metadata filtering")
    print()
    print("2. HYBRID SEARCH ON:")
    print("   - Includes 'filters' field in output")
    print("   - Extracts metadata (author, date, tags) from natural language")
    print("   - Enables precise filtering combined with semantic search")
    print("   - Backward compatible (filters can be empty)")

if __name__ == "__main__":
    main()