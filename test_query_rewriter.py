import sys
import json
sys.path.insert(0, 'src')
from config_manager import ConfigManager
from llm_client import LLMClient  
from query_rewriter import QueryRewriter

if len(sys.argv) != 2:
    print("Usage: python test_query_rewriter.py 'your query here'")
    print("Example: python test_query_rewriter.py '@knowledgebase articles from 2023'")
    sys.exit(1)

query = sys.argv[1]

config_manager = ConfigManager('config/config.yaml')
llm_config = config_manager.get_llm_config()
llm_client = LLMClient(llm_config)

query_rewriter_config = config_manager.get('query_rewriter', {})
query_rewriter_config['trigger_phrase'] = config_manager.get('rag.trigger_phrase', '@knowledgebase')
query_rewriter = QueryRewriter(llm_client, query_rewriter_config)

print(f"Testing query: {query}")
print("=" * 50)

result = query_rewriter.transform_query(query)
print(json.dumps(result, indent=2))

# Phase 3 Analysis
print("\nPhase 3 Analysis:")
print(f"RAG triggered: {result.get('search_rag', False)}")

hard_filters = result.get('hard_filters', {})

if hard_filters:
    print("Hard Filters (must match exactly):")
    for key, value in hard_filters.items():
        if key == 'publication_date' and isinstance(value, dict):
            print(f"  {key}: DatetimeRange {value}")
        else:
            print(f"  {key}: {value}")
else:
    print("No hard filters extracted")
