#!/usr/bin/env python3
"""Debug script to test Ollama response directly."""

import sys
import os
import json
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import ConfigManager
from llm_client import LLMClient
from query_rewriter import QueryRewriter

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_ollama_direct():
    """Test Ollama response directly."""
    try:
        # Load configuration
        config_manager = ConfigManager('config/config.yaml')
        llm_config = config_manager.get_llm_config()
        
        print("=== LLM Configuration ===")
        print(f"Provider: {llm_config.get('provider')}")
        print(f"Model: {llm_config.get('model')}")
        print(f"Base URL: {llm_config.get('base_url')}")
        print()
        
        # Initialize LLM client
        llm_client = LLMClient(llm_config)
        
        # Test simple query first
        simple_messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
            {"role": "user", "content": "Generate a simple JSON object with 'test': 'hello world'"}
        ]
        
        print("=== Testing Simple JSON Response ===")
        try:
            simple_response = llm_client.get_llm_response(simple_messages)
            print(f"Simple response: {simple_response}")
            print()
        except Exception as e:
            print(f"Simple test failed: {e}")
            return
        
        # Test query rewriter
        print("=== Testing Query Rewriter ===")
        query_rewriter_config = config_manager.get('query_rewriter', {})
        query_rewriter_config['trigger_phrase'] = config_manager.get('rag.trigger_phrase', '@knowledgebase')
        
        query_rewriter = QueryRewriter(llm_client, query_rewriter_config)
        
        test_query = "@knowledgebase search articles related to vibe coding from John Wong only from 2025, then summarize the findings, list pros and cons, and mitigation to the cons"
        
        print(f"Test query: {test_query}")
        print()
        
        # Get the system prompt being used
        print("=== System Prompt (first 200 chars) ===")
        print(f"{query_rewriter.system_prompt[:200]}...")
        print()
        
        # Test with raw LLM response first
        print("=== Testing Raw LLM Response ===")
        messages = [
            {"role": "system", "content": query_rewriter.system_prompt},
            {"role": "user", "content": test_query}
        ]
        
        try:
            raw_response = llm_client.get_llm_response(messages, temperature=0.1, max_tokens=512)
            print(f"Raw LLM response: {raw_response}")
            print()
            
            # Try to parse it as JSON
            try:
                json_response = llm_client.get_json_response(messages, temperature=0.1, max_tokens=512)
                print("=== Parsed JSON Response ===")
                print(json.dumps(json_response, indent=2))
                print()
                
                # Check specific fields
                print("=== Field Validation ===")
                print(f"Has 'embedding_texts': {'embedding_texts' in json_response}")
                if 'embedding_texts' in json_response:
                    et = json_response['embedding_texts']
                    print(f"embedding_texts type: {type(et)}")
                    print(f"Has 'rewrite': {'rewrite' in et if isinstance(et, dict) else 'N/A'}")
                    if isinstance(et, dict) and 'rewrite' in et:
                        rewrite = et['rewrite']
                        print(f"rewrite type: {type(rewrite)}")
                        print(f"rewrite value: '{rewrite}'")
                        print(f"rewrite stripped: '{rewrite.strip()}' (empty: {not rewrite.strip()})")
                print()
                
            except Exception as e:
                print(f"❌ JSON parsing failed: {e}")
                
        except Exception as e:
            print(f"❌ Raw LLM call failed: {e}")
        
        # Test the query transformation
        print("=== Testing Query Transformation ===")
        try:
            result = query_rewriter.transform_query(test_query)
            print("=== Query Transformation Result ===")
            print(f"Source: {result.get('source', 'unknown')}")
            print(f"Search RAG: {result.get('search_rag')}")
            print(f"Rewrite: {result.get('embedding_texts', {}).get('rewrite', 'N/A')}")
            print(f"Filters: hard={len(result.get('hard_filters', {}))}, soft={len(result.get('soft_filters', {}))}")
            print()
            
            if result.get('source') == 'fallback':
                print("❌ Result came from fallback - LLM call failed")
            else:
                print("✅ Result came from LLM - success")
                
        except Exception as e:
            print(f"❌ Query transformation failed: {e}")
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_direct()