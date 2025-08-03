import sys
import json
sys.path.insert(0, 'src')
from config_manager import ConfigManager
from llm_client import LLMClient  
from query_rewriter import QueryRewriter

def main():
    if len(sys.argv) != 2:
        print("Query Rewriter Testing Tool")
        print("=" * 50)
        print("Test how the query rewriter transforms different types of queries.")
        print()
        print("Usage: python check_query_rewriter.py 'your query here'")
        print()
        print("Configuration via environment variables:")
        print("  RAG_STRATEGY=hyde|rewrite    Set retrieval strategy (default: rewrite)")
        print()
        print("Query Pattern Examples:")
        print("  # Pattern 1 - Pure Search (extracts filters + search summary)")
        print("  python check_query_rewriter.py 'search @knowledgebase on AI research'")
        print("  python check_query_rewriter.py '@knowledgebase find papers by Smith from 2024'")
        print()
        print("  # Pattern 2 - Search + Action (extracts filters + performs action)")
        print("  python check_query_rewriter.py 'search @knowledgebase on Python, explain the benefits'")
        print("  python check_query_rewriter.py '@knowledgebase get papers on ML and compare approaches'")
        print()
        print("  # Pattern 3 - Direct Questions (no filter extraction, semantic context)")
        print("  python check_query_rewriter.py '@knowledgebase what is machine learning'")
        print("  python check_query_rewriter.py '@knowledgebase explain the benefits of Python'")
        print()
        print("Strategy Examples:")
        print("  RAG_STRATEGY=hyde python check_query_rewriter.py '@knowledgebase what is AI'")
        print("  RAG_STRATEGY=rewrite python check_query_rewriter.py 'search @knowledgebase on neural networks'")
        sys.exit(1)
    
    query = sys.argv[1]
    
    config_manager = ConfigManager('config/config.yaml')
    llm_config = config_manager.get_llm_config()
    llm_client = LLMClient(llm_config)
    
    # Get QueryRewriter configuration (includes environment variable overrides)
    query_rewriter_config = config_manager.get('query_rewriter', {})
    query_rewriter_config['trigger_phrase'] = config_manager.get('rag.trigger_phrase', '@knowledgebase')
    
    # Get strategy from config (set by environment variable or default)
    strategy = query_rewriter_config.get('retrieval_strategy', 'rewrite')
    strategy_name = "HyDE" if strategy == 'hyde' else "Rewrite"
    
    query_rewriter = QueryRewriter(llm_client, query_rewriter_config)
    
    print(f"Testing query: {query}")
    print(f"Strategy: {strategy_name}")
    print("=" * 60)
    
    result = query_rewriter.transform_query(query)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
