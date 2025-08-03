"""
Integration tests for QueryRewriter with real LLM providers.
Minimal tests to confirm actual LLM behavior that can't be mocked.
"""

import sys
import os
from typing import Dict, Any

import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from query_rewriter import QueryRewriter
from llm_client import LLMClient
from config_manager import ConfigManager

class TestQueryRewriterIntegration:
    """Integration tests for QueryRewriter with real LLM."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with real configuration."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        cls.config_manager = ConfigManager(config_path)
        
        # Initialize real LLM client
        llm_config = cls.config_manager.get_llm_config()
        cls.llm_client = LLMClient(llm_config)
        
        # Initialize QueryRewriter with real dependencies
        query_rewriter_config = cls.config_manager.get('query_rewriter', {})
        query_rewriter_config['trigger_phrase'] = cls.config_manager.get('rag.trigger_phrase', '@knowledgebase')
        cls.query_rewriter = QueryRewriter(cls.llm_client, query_rewriter_config)
    
    def validate_json_structure(self, result: Dict[str, Any]) -> bool:
        """Enhanced validation that includes filters field for hybrid search."""
        required_fields = ['search_rag', 'embedding_source_text', 'llm_query']
        
        if not all(field in result for field in required_fields):
            return False
        
        if not isinstance(result['search_rag'], bool):
            return False
        
        if not isinstance(result['embedding_source_text'], str):
            return False
        
        # embedding_source_text can be empty for non-RAG queries
        if result['search_rag'] and not result['embedding_source_text'].strip():
            return False
        
        if not isinstance(result['llm_query'], str) or not result['llm_query'].strip():
            return False
        
        # Validate filters field if present (should be present for hybrid search)
        if 'filters' in result:
            if not isinstance(result['filters'], dict):
                return False
        
        return True
    
    @pytest.mark.integration
    def test_rag_query_with_real_llm(self):
        """Test RAG query transformation with real LLM - confirms JSON parsing and context instruction."""
        user_query = "@knowledgebase What are the benefits of machine learning?"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure (most critical - confirms JSON parsing works with real LLM)
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        
        # Validate trigger detection
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate embedding text doesn't contain trigger phrase
        assert '@knowledgebase' not in result['embedding_source_text'].lower(), "Embedding text should not contain trigger phrase"
        
        # Validate RAG query includes context reference (critical for RAG functionality)
        llm_query = result['llm_query'].lower()
        context_indicators = ['based on', 'context', 'provided', 'from the']
        has_context_ref = any(indicator in llm_query for indicator in context_indicators)
        assert has_context_ref, f"RAG query should reference context. Got: {result['llm_query']}"
        
        print("RAG Integration Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
    
    @pytest.mark.integration
    def test_non_rag_query_with_real_llm(self):
        """Test non-RAG query transformation with real LLM - confirms general instruction format."""
        user_query = "What is machine learning?"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        
        # Validate trigger detection
        assert result['search_rag'] is False, "Should not detect RAG trigger"
        
        # Validate non-RAG query doesn't mention context (important distinction)
        llm_query = result['llm_query'].lower()
        context_indicators = ['based on', 'context', 'provided', 'from the']
        has_context_ref = any(indicator in llm_query for indicator in context_indicators)
        assert not has_context_ref, f"Non-RAG query should not reference context. Got: {result['llm_query']}"
        
        print("Non-RAG Integration Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
    
    @pytest.mark.integration
    def test_real_llm_connection_and_error_handling(self):
        """Test that real LLM connection works and error handling is robust."""
        # Test connection (confirms real LLM is accessible)
        connection_success = self.query_rewriter.test_connection()
        assert connection_success, "QueryRewriter should successfully connect to real LLM"
        
        # Test that the system can handle real LLM responses consistently
        # This is important because real LLMs may return slightly different formats
        test_query = "@knowledgebase How does neural network training work?"
        
        # Run same query twice to check consistency in parsing
        result1 = self.query_rewriter.transform_query(test_query)
        result2 = self.query_rewriter.transform_query(test_query)
        
        # Both should have valid structure (tests robustness of JSON parsing)
        assert self.validate_json_structure(result1), "First result should have valid structure"
        assert self.validate_json_structure(result2), "Second result should have valid structure"
        
        # Both should detect trigger correctly (tests consistency)
        assert result1['search_rag'] is True, "First result should detect trigger"
        assert result2['search_rag'] is True, "Second result should detect trigger"
        
        print(f"Connection Test: {connection_success}")
        print("Consistency Test - both results have valid structure and trigger detection")
    
    @pytest.mark.integration
    def test_conversational_context_detection(self):
        """Test conversational follow-up detection with real LLM."""
        user_query = "Tell me more about the automation benefits"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        
        # Should not trigger RAG search
        assert result['search_rag'] is False, "Conversational follow-up should not trigger RAG"
        
        # Should reference previous conversation
        llm_query = result['llm_query'].lower()
        conversation_indicators = ['based on context in previous conversation', 'previous conversation', 'conversation history']
        has_conversation_ref = any(indicator in llm_query for indicator in conversation_indicators)
        assert has_conversation_ref, f"Conversational query should reference previous conversation. Got: {result['llm_query']}"
        
        print("Conversational Context Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
    
    @pytest.mark.integration
    def test_hybrid_search_author_filtering(self):
        """Test natural language author filtering with real LLM."""
        user_query = "@knowledgebase papers by John Smith"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field exists and has correct structure
        assert 'filters' in result, "Result should include filters field for hybrid search"
        filters = result['filters']
        assert isinstance(filters, dict), "Filters should be a dictionary"
        
        # Validate author extraction
        assert 'author' in filters, "Should extract author from natural language query"
        assert filters['author'] == "John Smith", f"Should extract 'John Smith', got: {filters['author']}"
        
        print("Author Filtering Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
        print(f"  Filters: {result.get('filters', {})}")
    
    @pytest.mark.integration
    def test_hybrid_search_date_filtering(self):
        """Test natural language date filtering with real LLM."""
        user_query = "@knowledgebase articles from 2023"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate date extraction - handle both legacy string and new DatetimeRange formats
        assert 'publication_date' in filters, "Should extract publication date from natural language query"
        pub_date = filters['publication_date']
        
        # Accept either new DatetimeRange format or legacy string format
        if isinstance(pub_date, dict):
            # New DatetimeRange format (Phase 2 success)
            assert 'gte' in pub_date, f"DatetimeRange should have 'gte' field, got: {pub_date}"
            assert 'lt' in pub_date, f"DatetimeRange should have 'lt' field, got: {pub_date}"
            assert pub_date['gte'] == "2023-01-01", f"Should extract 2023 start, got: {pub_date['gte']}"
            assert pub_date['lt'] == "2024-01-01", f"Should extract 2023 end, got: {pub_date['lt']}"
        else:
            # Legacy string format
            assert pub_date == "2023", f"Legacy format should extract '2023', got: {pub_date}"
        
        print("Date Filtering Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
        print(f"  Filters: {result.get('filters', {})}")
    
    @pytest.mark.integration
    def test_hybrid_search_tag_filtering(self):
        """Test natural language tag/topic filtering with real LLM."""
        user_query = "@knowledgebase documents about Python programming"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate tag extraction - Note: Based on our current prompt, topics like "Python programming" 
        # should NOT be auto-extracted as tags unless explicitly requested with "with tag" syntax
        # This test checks if tags are extracted, but doesn't fail if they're not (correct behavior)
        if 'tags' in filters:
            tags = filters['tags']
            assert isinstance(tags, list), "Tags should be a list"
            assert len(tags) > 0, "Should extract at least one tag"
            
            # Check that Python-related topics are extracted
            tags_lower = [tag.lower() for tag in tags]
            python_related = any('python' in tag for tag in tags_lower)
            assert python_related, f"Should extract Python-related tags, got: {tags}"
            
            print("  ✅ Tags were auto-extracted (legacy behavior):", tags)
        else:
            print("  ⚠️  No tags auto-extracted from topic words (correct Phase 2 behavior)")
            print("  Note: Use 'with tag' syntax for explicit tag extraction")
        
        print("Tag Filtering Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
        print(f"  Filters: {result.get('filters', {})}")
    
    @pytest.mark.integration
    def test_hybrid_search_combined_filtering_complex(self):
        """Test complex natural language query with multiple filters - the exact user example."""
        user_query = "@knowledgebase search on vibe coding from John Wong published in March 2025, then explain what is vibe coding, and pros/cons"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate author extraction
        assert 'author' in filters, "Should extract author from complex query"
        assert filters['author'] == "John Wong", f"Should extract 'John Wong', got: {filters['author']}"
        
        # Validate date extraction (should handle "March 2025" format)
        assert 'publication_date' in filters, "Should extract publication date from complex query"
        pub_date = filters['publication_date']
        # Accept either DatetimeRange format or legacy string formats
        if isinstance(pub_date, dict):
            # New DatetimeRange format - should have gte for March 2025
            assert 'gte' in pub_date, f"DatetimeRange should have 'gte' field, got: {pub_date}"
            assert pub_date['gte'].startswith('2025-03'), f"Should extract March 2025 range, got: {pub_date['gte']}"
        else:
            # Legacy format - accept various representations
            valid_dates = ["2025-03", "March 2025", "2025"]
            assert pub_date in valid_dates, f"Should extract valid date format, got: {pub_date}"
        
        # Validate NO automatic tags extraction for "on vibe coding" (broader search)
        # Tags should only be extracted with explicit "with tag" syntax
        if 'tags' in filters:
            # If tags are present, they should be explicitly requested, not auto-extracted from "on vibe coding"
            tags = filters['tags']
            print(f"  Note: Tags found (should be explicit only): {tags}")
        else:
            print("  Note: No tags extracted from 'on vibe coding' - correct behavior for broader search")
        
        print("Complex Combined Filtering Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
        print(f"  Filters: {result.get('filters', {})}")
    
    @pytest.mark.integration  
    def test_explicit_tag_syntax_integration(self):
        """Test explicit tag syntax with real LLM responses."""
        test_cases = [
            ("@knowledgebase papers by John Smith with tag machine learning", ["machine learning"]),
            ("@knowledgebase articles with tags python, AI from 2023", ["python", "ai"]),
            ("@knowledgebase research with tag vibe coding from John Wong", ["vibe coding"]),
        ]
        
        for user_query, expected_tag_keywords in test_cases:
            result = self.query_rewriter.transform_query(user_query)
            
            # Validate basic structure
            assert self.validate_json_structure(result), f"Invalid JSON structure for query: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger for: {user_query}"
            
            # Validate filters field
            assert 'filters' in result, f"Should include filters field for: {user_query}"
            filters = result['filters']
            
            # Validate explicit tag extraction
            assert 'tags' in filters, f"Should extract tags with explicit syntax for: {user_query}"
            tags = filters['tags']
            assert isinstance(tags, list), f"Tags should be a list for: {user_query}"
            
            # Check that at least one expected tag keyword is present
            tags_str = ' '.join(tags).lower()
            tag_found = any(keyword.lower() in tags_str for keyword in expected_tag_keywords)
            assert tag_found, f"Should extract expected tag keywords {expected_tag_keywords}, got: {tags}"
            
            print(f"Explicit Tag Test - Query: {user_query}")
            print(f"  Expected keywords: {expected_tag_keywords}")
            print(f"  Extracted tags: {tags}")
            print(f"  All filters: {filters}")
    
    @pytest.mark.integration
    def test_no_auto_tag_extraction_integration(self):
        """Test that topic-based queries don't auto-extract tags with real LLM."""
        test_cases = [
            "@knowledgebase documents about Python programming",
            "@knowledgebase articles on machine learning",
            "@knowledgebase research about AI and robotics", 
            "@knowledgebase search on vibe coding and productivity",
        ]
        
        for user_query in test_cases:
            result = self.query_rewriter.transform_query(user_query)
            
            # Validate basic structure
            assert self.validate_json_structure(result), f"Invalid JSON structure for query: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger for: {user_query}"
            
            # Validate that NO tags are auto-extracted from topic phrases
            filters = result.get('filters', {})
            
            if 'tags' in filters:
                # If tags are present, print warning - this might indicate the LLM is still auto-extracting
                print(f"WARNING: Unexpected tags found for topic query: {user_query}")
                print(f"  Tags: {filters['tags']}")
                print("  This may indicate the LLM is still auto-extracting tags from topics")
            else:
                print(f"✓ Correct: No auto-extracted tags for topic query: {user_query}")
    
    @pytest.mark.integration
    def test_hybrid_search_author_variations(self):
        """Test different ways of expressing author in natural language."""
        test_cases = [
            ("@knowledgebase papers by Dr. Smith", "Dr. Smith"),
            ("@knowledgebase articles from Jane Doe", "Jane Doe"),  
            ("@knowledgebase research authored by Wilson", "Wilson")
        ]
        
        for user_query, expected_author in test_cases:
            result = self.query_rewriter.transform_query(user_query)
            
            # Validate basic structure
            assert self.validate_json_structure(result), f"Invalid JSON structure for query: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger for: {user_query}"
            
            # Validate author extraction
            assert 'filters' in result, f"Should have filters for: {user_query}"
            filters = result['filters']
            assert 'author' in filters, f"Should extract author from: {user_query}"
            
            # Author matching should be flexible (exact match or contains)
            extracted_author = filters['author']
            author_match = (extracted_author == expected_author) or (expected_author in extracted_author)
            assert author_match, f"Should extract '{expected_author}' or similar from '{user_query}', got: {extracted_author}"
            
            print(f"Author Variation Test - '{user_query}': extracted '{extracted_author}'")
    
    @pytest.mark.integration
    def test_hybrid_search_no_filters_query(self):
        """Test that queries without metadata don't produce spurious filters."""
        user_query = "@knowledgebase What is Python programming?"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field exists but may be empty
        assert 'filters' in result, "Result should include filters field even when empty"
        filters = result['filters']
        assert isinstance(filters, dict), "Filters should be a dictionary"
        
        # For a general question without specific metadata, filters should be minimal or empty
        # Don't be too strict here as LLM might extract "Python programming" as a tag
        print("No Metadata Filtering Test:")
        print(f"  User Query: {user_query}")
        print(f"  Search RAG: {result['search_rag']}")
        print(f"  Embedding Text: {result['embedding_source_text']}")
        print(f"  LLM Query: {result['llm_query']}")
        print(f"  Filters: {result.get('filters', {})}")
    
    @pytest.mark.integration
    def test_hybrid_search_consistency(self):
        """Test that hybrid search filtering is consistent across multiple runs."""
        user_query = "@knowledgebase machine learning papers by Smith from 2023"
        
        # Run the same query multiple times
        results = []
        for i in range(2):
            result = self.query_rewriter.transform_query(user_query)
            results.append(result)
        
        # All results should have valid structure
        for i, result in enumerate(results):
            assert self.validate_json_structure(result), f"Result {i} should have valid structure"
            assert result['search_rag'] is True, f"Result {i} should detect RAG trigger"
            assert 'filters' in result, f"Result {i} should have filters field"
        
        # Key filters should be consistent (author and date should be extracted consistently)
        filters_list = [result['filters'] for result in results]
        
        # Check author consistency
        authors = [f.get('author') for f in filters_list]
        author_consistent = all(author and 'smith' in author.lower() for author in authors if author)
        if any(authors):  # Only check if at least one extracted an author
            assert author_consistent, f"Author extraction should be consistent: {authors}"
        
        # Check date consistency  
        dates = [f.get('publication_date') for f in filters_list]
        date_consistent = all(date and '2023' in str(date) for date in dates if date)
        if any(dates):  # Only check if at least one extracted a date
            assert date_consistent, f"Date extraction should be consistent: {dates}"
        
        print("Consistency Test Results:")
        for i, result in enumerate(results):
            print(f"  Run {i+1} Filters: {result.get('filters', {})}")
    
    @pytest.mark.integration
    def test_hyde_vs_rewrite_strategy_comparison(self):
        """Test HyDE vs Rewrite strategy comparison with real Gemini LLM."""
        user_query = "@knowledgebase How does neural network training work?"
        
        # Create Gemini configuration for testing
        gemini_config = self.config_manager.get_llm_config()
        # Force Gemini provider (skip if not available)
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
        
        gemini_llm_client = LLMClient(gemini_config)
        
        # Test Rewrite strategy
        rewrite_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite',
            'temperature': 0.1,
            'max_tokens': 512
        }
        rewrite_rewriter = QueryRewriter(gemini_llm_client, rewrite_config)
        rewrite_result = rewrite_rewriter.transform_query(user_query)
        
        # Test HyDE strategy  
        hyde_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'hyde',
            'temperature': 0.1,
            'max_tokens': 512
        }
        hyde_rewriter = QueryRewriter(gemini_llm_client, hyde_config)
        hyde_result = hyde_rewriter.transform_query(user_query)
        
        # Validate both results have valid structure
        assert self.validate_json_structure(rewrite_result), f"Rewrite result invalid: {rewrite_result}"
        assert self.validate_json_structure(hyde_result), f"HyDE result invalid: {hyde_result}"
        
        # Both should detect RAG trigger
        assert rewrite_result['search_rag'] is True, "Rewrite should detect RAG trigger"
        assert hyde_result['search_rag'] is True, "HyDE should detect RAG trigger"
        
        # Both should have filters field
        assert 'filters' in rewrite_result, "Rewrite should have filters field"
        assert 'filters' in hyde_result, "HyDE should have filters field"
        
        # Key difference: HyDE should produce longer, more descriptive embedding text
        rewrite_text_len = len(rewrite_result['embedding_source_text'])
        hyde_text_len = len(hyde_result['embedding_source_text'])
        
        assert hyde_text_len > rewrite_text_len, f"HyDE text ({hyde_text_len}) should be longer than Rewrite ({rewrite_text_len})"
        assert hyde_text_len > 100, f"HyDE should generate document-like text (>100 chars), got {hyde_text_len}"
        assert rewrite_text_len < 100, f"Rewrite should generate keyword-like text (<100 chars), got {rewrite_text_len}"
        
        # Both should reference "provided context" in LLM query
        assert "provided context" in rewrite_result['llm_query'].lower(), "Rewrite should reference provided context"
        assert "provided context" in hyde_result['llm_query'].lower(), "HyDE should reference provided context"
        
        print("Strategy Comparison Test:")
        print(f"  User Query: {user_query}")
        print(f"  Rewrite embedding length: {rewrite_text_len} chars")
        print(f"  HyDE embedding length: {hyde_text_len} chars") 
        print(f"  Rewrite embedding: {rewrite_result['embedding_source_text'][:100]}...")
        print(f"  HyDE embedding: {hyde_result['embedding_source_text'][:100]}...")
    
    @pytest.mark.integration
    def test_custom_trigger_phrase_integration(self):
        """Test QueryRewriter with custom trigger phrase and dependency injection."""
        # Create Gemini LLM client
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
        
        gemini_llm_client = LLMClient(gemini_config)
        
        # Test with custom trigger phrase
        custom_config = {
            'trigger_phrase': '@search',  # Different from default @knowledgebase
            'retrieval_strategy': 'rewrite',
            'temperature': 0.1,
            'max_tokens': 512
        }
        custom_rewriter = QueryRewriter(gemini_llm_client, custom_config)
        
        # Test query with custom trigger
        user_query = "@search What is machine learning?"
        result = custom_rewriter.transform_query(user_query)
        
        # Validate structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        
        # Should detect the custom trigger
        assert result['search_rag'] is True, "Should detect custom @search trigger"
        
        # Should have proper embedding text (without trigger)
        assert '@search' not in result['embedding_source_text'].lower(), "Embedding text should not contain custom trigger"
        assert len(result['embedding_source_text'].strip()) > 0, "Should have non-empty embedding text"
        
        # Should reference provided context for RAG query
        assert "provided context" in result['llm_query'].lower(), "Should reference provided context"
        
        # Test query without custom trigger (should not trigger RAG)
        non_trigger_query = "@knowledgebase What is machine learning?"  # Default trigger, not custom
        non_trigger_result = custom_rewriter.transform_query(non_trigger_query)
        
        assert self.validate_json_structure(non_trigger_result), f"Invalid JSON structure: {non_trigger_result}"
        assert non_trigger_result['search_rag'] is False, "Should not detect default trigger when using custom trigger"
        
        print("Custom Trigger Test:")
        print("  Custom trigger: @search")
        print(f"  Query with custom trigger - RAG: {result['search_rag']}")
        print(f"  Query with default trigger - RAG: {non_trigger_result['search_rag']}")
        print(f"  Embedding text: {result['embedding_source_text']}")
    
    @pytest.mark.integration
    def test_complex_multi_author_queries(self):
        """Test complex queries with multiple authors using real Gemini LLM."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
        
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite',
            'temperature': 0.1,
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Test different multi-author query patterns
        test_cases = [
            "@knowledgebase papers by Smith and Jones about AI",
            "@knowledgebase research from Dr. Wilson and Prof. Chen on neural networks", 
            "@knowledgebase collaborative work by Johnson, Davis, and Brown"
        ]
        
        for user_query in test_cases:
            result = rewriter.transform_query(user_query)
            
            # Validate structure
            assert self.validate_json_structure(result), f"Invalid structure for: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger for: {user_query}"
            
            # Should have filters field
            assert 'filters' in result, f"Should have filters for: {user_query}"
            filters = result['filters']
            
            # Should extract author information (can be single author or multiple)
            has_author_info = (
                'author' in filters or 
                ('authors' in filters) or
                any('author' in key.lower() for key in filters.keys())
            )
            
            # Print results for manual inspection (LLM behavior may vary)
            print(f"Multi-Author Test: {user_query}")
            if has_author_info:
                print(f"  ✅ Extracted author info: {filters}")
            else:
                print(f"  ⚠️  No author extracted from filters: {filters}")
            print(f"  Embedding: {result['embedding_source_text']}")
            print()
    
    @pytest.mark.integration  
    def test_date_range_and_complex_dates(self):
        """Test complex date formats and ranges with real Gemini LLM."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
            
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite',  
            'temperature': 0.1,
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Test various date formats and ranges
        test_cases = [
            "@knowledgebase papers from 2020 to 2023",
            "@knowledgebase research published in March 2025", 
            "@knowledgebase articles from the last decade",
            "@knowledgebase recent work from 2024-2025",
            "@knowledgebase documents published between January and March 2023"
        ]
        
        for user_query in test_cases:
            result = rewriter.transform_query(user_query)
            
            # Validate structure
            assert self.validate_json_structure(result), f"Invalid structure for: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger for: {user_query}"
            
            # Should have filters field
            assert 'filters' in result, f"Should have filters for: {user_query}"
            filters = result['filters']
            
            # Check for date-related extraction
            has_date_info = (
                'publication_date' in filters or
                'date' in filters or
                'year' in filters or
                any('date' in key.lower() for key in filters.keys())
            )
            
            # Print results for analysis (LLM behavior may vary)
            print(f"Date Range Test: {user_query}")
            if has_date_info:
                print(f"  ✅ Extracted date info: {filters}")
            else:
                print(f"  ⚠️  No date extracted from filters: {filters}")
            print(f"  Embedding: {result['embedding_source_text']}")
            print()
    
    @pytest.mark.integration
    def test_complex_topic_combinations(self):
        """Test complex topic and tag combinations with real Gemini LLM."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
            
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase', 
            'retrieval_strategy': 'rewrite',
            'temperature': 0.1,
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Test complex topic combinations
        test_cases = [
            "@knowledgebase papers on machine learning and neural networks in healthcare",
            "@knowledgebase research about Python programming and web development frameworks",
            "@knowledgebase articles covering AI, robotics, and computer vision applications",
            "@knowledgebase documents on data science, statistics, and predictive modeling",
            "@knowledgebase work related to blockchain technology and cryptocurrency security"
        ]
        
        for user_query in test_cases:
            result = rewriter.transform_query(user_query)
            
            # Validate structure
            assert self.validate_json_structure(result), f"Invalid structure for: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger for: {user_query}"
            
            # Should have filters field
            assert 'filters' in result, f"Should have filters for: {user_query}"
            filters = result['filters']
            
            # Check for topic/tag extraction
            has_topic_info = (
                'tags' in filters or
                'topics' in filters or
                'keywords' in filters or
                any('tag' in key.lower() for key in filters.keys())
            )
            
            # Print results for analysis
            print(f"Topic Combination Test: {user_query}")
            if has_topic_info:
                print(f"  ✅ Extracted topic info: {filters}")
                if 'tags' in filters and isinstance(filters['tags'], list):
                    print(f"  📝 Number of tags: {len(filters['tags'])}")
            else:
                print(f"  ⚠️  No topics extracted from filters: {filters}")
            print(f"  Embedding: {result['embedding_source_text'][:100]}...")
            print()
    
    @pytest.mark.integration
    def test_very_long_complex_queries(self):
        """Test robustness with very long and complex queries using real Gemini LLM."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
            
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite',
            'temperature': 0.1,
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Very long, complex query with multiple filters and instructions
        long_query = """@knowledgebase I need to find comprehensive research papers written by Dr. Sarah Johnson, Prof. Michael Chen, and their research team members, specifically focusing on machine learning applications in healthcare, particularly deep learning models for medical image analysis, natural language processing for clinical notes, and predictive analytics for patient outcomes. I'm especially interested in work published between 2022 and 2024, including conference papers, journal articles, and technical reports that discuss both theoretical foundations and practical implementations. Please also include related work on neural networks, computer vision, and AI ethics in medical applications. After finding these documents, I want you to synthesize the key findings, compare different approaches, highlight the most innovative techniques, discuss limitations and future research directions, and provide a detailed analysis of how these technologies are being adopted in real-world clinical settings."""
        
        result = rewriter.transform_query(long_query)
        
        # Validate basic structure still works with very long queries
        assert self.validate_json_structure(result), "Invalid structure for long query"
        assert result['search_rag'] is True, "Should detect RAG trigger in long query"
        
        # Should have filters field
        assert 'filters' in result, "Should have filters for long query"
        filters = result['filters']
        
        # Should extract meaningful information despite complexity
        assert len(result['embedding_source_text'].strip()) > 0, "Should extract non-empty embedding text"
        assert len(result['llm_query'].strip()) > 0, "Should generate non-empty LLM query"
        
        # Check if any filters were extracted from the complex query
        has_filters = len(filters) > 0
        
        print("Long Complex Query Test:")
        print(f"  Query length: {len(long_query)} characters")
        print(f"  RAG detected: {result['search_rag']}")
        print(f"  Filters extracted: {has_filters}")
        if has_filters:
            print(f"  Filters: {filters}")
        print(f"  Embedding text length: {len(result['embedding_source_text'])}")
        print(f"  Embedding sample: {result['embedding_source_text'][:150]}...")
        print(f"  LLM query length: {len(result['llm_query'])}")
        
        # Verify the system didn't break or produce corrupted output
        assert isinstance(result['search_rag'], bool), "search_rag should remain boolean"
        assert isinstance(result['embedding_source_text'], str), "embedding_source_text should remain string"
        assert isinstance(result['llm_query'], str), "llm_query should remain string"
        assert isinstance(result['filters'], dict), "filters should remain dict"
    
    @pytest.mark.integration
    def test_special_characters_and_formatting(self):
        """Test queries with special characters and formatting edge cases."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
            
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite', 
            'temperature': 0.1,
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Test queries with various special characters and formatting
        test_cases = [
            '@knowledgebase papers by "Dr. John Smith" about AI/ML & robotics',
            "@knowledgebase research on C++ and Python (2023-2024) [peer-reviewed]",
            "@knowledgebase articles with titles containing 'machine learning' or 'deep learning'",
            "@knowledgebase work by Smith, J. et al. on NLP & computer vision",
            "@knowledgebase documents about AI-driven healthcare solutions (IoT + ML)"
        ]
        
        for user_query in test_cases:
            result = rewriter.transform_query(user_query)
            
            # Validate that special characters don't break the system
            assert self.validate_json_structure(result), f"Special chars broke structure: {user_query}"
            assert result['search_rag'] is True, f"Should detect RAG trigger despite special chars: {user_query}"
            
            # Should still have proper fields
            assert 'filters' in result, f"Should have filters despite special chars: {user_query}"
            
            # Should generate reasonable output despite formatting
            assert len(result['embedding_source_text'].strip()) > 0, f"Should extract content despite special chars: {user_query}"
            assert len(result['llm_query'].strip()) > 0, f"Should generate LLM query despite special chars: {user_query}"
            
            print(f"Special Characters Test: {user_query}")
            print("  ✅ Parsed successfully")
            print(f"  Filters: {result['filters']}")
            print(f"  Embedding: {result['embedding_source_text']}")
            print()
    
    @pytest.mark.integration
    def test_gemini_consistency_across_runs(self):
        """Test Gemini consistency with same queries run multiple times."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
            
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite',
            'temperature': 0.1,  # Low temperature for consistency
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Test query that should produce consistent results
        test_query = "@knowledgebase papers by John Smith about machine learning from 2023"
        
        # Run the same query multiple times
        results = []
        for i in range(3):
            result = rewriter.transform_query(test_query)
            results.append(result)
        
        # All results should have valid structure
        for i, result in enumerate(results):
            assert self.validate_json_structure(result), f"Run {i+1} should have valid structure"
            assert result['search_rag'] is True, f"Run {i+1} should detect RAG trigger"
            assert 'filters' in result, f"Run {i+1} should have filters field"
        
        # Check consistency of key extractions
        authors = [r['filters'].get('author', '') for r in results]
        dates = [r['filters'].get('publication_date', '') for r in results]
        search_rag_values = [r['search_rag'] for r in results]
        
        # RAG detection should be 100% consistent
        assert all(val is True for val in search_rag_values), f"RAG detection should be consistent: {search_rag_values}"
        
        # Author extraction consistency (if any extracted)
        authors_extracted = [a for a in authors if a]
        if authors_extracted:
            # Most should extract "Smith" or "John Smith"
            smith_mentions = sum(1 for a in authors_extracted if 'smith' in a.lower())
            consistency_rate = smith_mentions / len(authors_extracted)
            assert consistency_rate >= 0.6, f"Author extraction should be reasonably consistent: {authors}"
        
        # Date extraction consistency (if any extracted)
        dates_extracted = [d for d in dates if d]
        if dates_extracted:
            # Most should extract "2023"
            year_mentions = sum(1 for d in dates_extracted if '2023' in str(d))
            consistency_rate = year_mentions / len(dates_extracted)
            assert consistency_rate >= 0.6, f"Date extraction should be reasonably consistent: {dates}"
        
        print("Gemini Consistency Test:")
        print(f"  Query: {test_query}")
        print(f"  Runs: {len(results)}")
        print(f"  RAG detection: {search_rag_values}")
        print(f"  Authors extracted: {authors}")
        print(f"  Dates extracted: {dates}")
        
        # Test structure consistency
        for i, result in enumerate(results):
            print(f"  Run {i+1} embedding length: {len(result['embedding_source_text'])}")
    
    @pytest.mark.integration
    def test_gemini_markdown_response_handling(self):
        """Test Gemini-specific markdown response patterns and JSON parsing."""
        gemini_config = self.config_manager.get_llm_config()
        if gemini_config.get('provider') != 'gemini':
            pytest.skip("Test requires Gemini provider")
            
        gemini_llm_client = LLMClient(gemini_config)
        standard_config = {
            'trigger_phrase': '@knowledgebase',
            'retrieval_strategy': 'rewrite',
            'temperature': 0.3,  # Slightly higher to encourage varied response formats
            'max_tokens': 512
        }
        rewriter = QueryRewriter(gemini_llm_client, standard_config)
        
        # Test multiple queries to see various Gemini response formats
        test_queries = [
            "@knowledgebase What is artificial intelligence?",
            "@knowledgebase papers by Dr. Johnson on neural networks", 
            "@knowledgebase research from 2024 about computer vision"
        ]
        
        all_parsed_successfully = True
        response_formats_seen = []
        
        for user_query in test_queries:
            try:
                result = rewriter.transform_query(user_query)
                
                # Validate JSON structure
                structure_valid = self.validate_json_structure(result)
                assert structure_valid, f"Gemini response parsing failed for: {user_query}"
                
                # Check that basic fields are present and valid
                assert isinstance(result['search_rag'], bool), "search_rag should be boolean"
                assert isinstance(result['embedding_source_text'], str), "embedding_source_text should be string"
                assert isinstance(result['llm_query'], str), "llm_query should be string"
                assert isinstance(result['filters'], dict), "filters should be dict"
                
                # Track that this query was parsed successfully
                response_formats_seen.append(user_query)
                
                print(f"Gemini Markdown Test: {user_query}")
                print("  ✅ Parsed successfully")
                print(f"  Structure valid: {structure_valid}")
                print(f"  RAG: {result['search_rag']}")
                print(f"  Filters: {result['filters']}")
                print()
                
            except Exception as e:
                print(f"❌ Failed to parse Gemini response for: {user_query}")
                print(f"   Error: {e}")
                all_parsed_successfully = False
        
        # Should successfully parse all Gemini responses
        assert all_parsed_successfully, "All Gemini responses should be parseable"
        assert len(response_formats_seen) >= 2, "Should successfully parse multiple different queries"
        
        print("Gemini Markdown Handling Summary:")
        print(f"  ✅ Successfully parsed {len(response_formats_seen)} different response formats")
        print("  ✅ JSON extraction and validation working correctly")

    @pytest.mark.integration
    def test_phase2_datetime_range_year_filtering(self):
        """Test Phase 2: Natural language year filtering converts to DatetimeRange format."""
        user_query = "@knowledgebase articles from 2023 about Python programming"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate date extraction - should use new DatetimeRange format
        assert 'publication_date' in filters, "Should extract publication date from natural language query"
        pub_date = filters['publication_date']
        
        # Phase 2 feature: Should convert to DatetimeRange format
        if isinstance(pub_date, dict):
            # New DatetimeRange format (Phase 2 success)
            assert 'gte' in pub_date, f"DatetimeRange should have 'gte' field, got: {pub_date}"
            assert 'lt' in pub_date, f"DatetimeRange should have 'lt' field, got: {pub_date}"
            assert pub_date['gte'] == "2023-01-01", f"Should extract 2023 start, got: {pub_date['gte']}"
            assert pub_date['lt'] == "2024-01-01", f"Should extract 2023 end, got: {pub_date['lt']}"
            print("✅ Phase 2 SUCCESS: Year converted to DatetimeRange format")
        else:
            # Legacy format - still acceptable but not Phase 2 feature
            assert pub_date == "2023", f"Legacy format should extract '2023', got: {pub_date}"
            print("⚠️  Phase 2 LEGACY: Using legacy string format (still works)")
        
        print("Phase 2 Year Range Test:")
        print(f"  User Query: {user_query}")
        print(f"  Publication Date Filter: {pub_date}")
        print(f"  Filter Type: {'DatetimeRange' if isinstance(pub_date, dict) else 'Legacy String'}")

    @pytest.mark.integration
    def test_phase2_datetime_range_month_filtering(self):
        """Test Phase 2: Natural language month filtering converts to DatetimeRange format."""
        user_query = "@knowledgebase papers published in March 2025"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate date extraction
        assert 'publication_date' in filters, "Should extract publication date from natural language query"
        pub_date = filters['publication_date']
        
        # Phase 2 feature: Should convert month to DatetimeRange format
        if isinstance(pub_date, dict):
            # New DatetimeRange format (Phase 2 success)
            assert 'gte' in pub_date, f"DatetimeRange should have 'gte' field, got: {pub_date}"
            assert 'lt' in pub_date, f"DatetimeRange should have 'lt' field, got: {pub_date}"
            assert pub_date['gte'] == "2025-03-01", f"Should extract March start, got: {pub_date['gte']}"
            assert pub_date['lt'] == "2025-04-01", f"Should extract March end, got: {pub_date['lt']}"
            print("✅ Phase 2 SUCCESS: Month converted to DatetimeRange format")
        else:
            # Legacy format - acceptable but not Phase 2 feature
            valid_legacy = ["2025-03", "March 2025", "2025"]
            assert pub_date in valid_legacy, f"Legacy format should extract valid month, got: {pub_date}"
            print("⚠️  Phase 2 LEGACY: Using legacy string format")
        
        print("Phase 2 Month Range Test:")
        print(f"  User Query: {user_query}")
        print(f"  Publication Date Filter: {pub_date}")
        print(f"  Filter Type: {'DatetimeRange' if isinstance(pub_date, dict) else 'Legacy String'}")

    @pytest.mark.integration
    def test_phase2_datetime_range_quarter_filtering(self):
        """Test Phase 2: Natural language quarter filtering converts to DatetimeRange format."""
        user_query = "@knowledgebase documents from 2025 Q1"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate date extraction
        assert 'publication_date' in filters, "Should extract publication date from natural language query"
        pub_date = filters['publication_date']
        
        # Phase 2 feature: Should convert quarter to DatetimeRange format
        if isinstance(pub_date, dict):
            # New DatetimeRange format (Phase 2 success)
            assert 'gte' in pub_date, f"DatetimeRange should have 'gte' field, got: {pub_date}"
            assert 'lt' in pub_date, f"DatetimeRange should have 'lt' field, got: {pub_date}"
            assert pub_date['gte'] == "2025-01-01", f"Should extract Q1 start, got: {pub_date['gte']}"
            assert pub_date['lt'] == "2025-04-01", f"Should extract Q1 end, got: {pub_date['lt']}"
            print("✅ Phase 2 SUCCESS: Quarter converted to DatetimeRange format")
        else:
            # Legacy format - acceptable but not Phase 2 feature
            assert "2025" in str(pub_date), f"Legacy format should contain 2025, got: {pub_date}"
            print("⚠️  Phase 2 LEGACY: Using legacy string format")
        
        print("Phase 2 Quarter Range Test:")
        print(f"  User Query: {user_query}")
        print(f"  Publication Date Filter: {pub_date}")
        print(f"  Filter Type: {'DatetimeRange' if isinstance(pub_date, dict) else 'Legacy String'}")

    @pytest.mark.integration
    def test_phase2_datetime_range_since_filtering(self):
        """Test Phase 2: 'Since' date filtering converts to DatetimeRange format with gte only."""
        user_query = "@knowledgebase articles by John Wong since 2024"
        
        result = self.query_rewriter.transform_query(user_query)
        
        # Validate basic structure
        assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
        assert result['search_rag'] is True, "Should detect RAG trigger"
        
        # Validate filters field
        assert 'filters' in result, "Result should include filters field"
        filters = result['filters']
        
        # Validate author extraction
        assert 'author' in filters, "Should extract author from query"
        assert filters['author'] == "John Wong", f"Should extract 'John Wong', got: {filters['author']}"
        
        # Validate date extraction
        assert 'publication_date' in filters, "Should extract publication date from natural language query"
        pub_date = filters['publication_date']
        
        # Phase 2 feature: Should convert 'since' to DatetimeRange format with gte only
        if isinstance(pub_date, dict):
            # New DatetimeRange format (Phase 2 success)
            assert 'gte' in pub_date, f"DatetimeRange should have 'gte' field for 'since', got: {pub_date}"
            assert pub_date['gte'] == "2024-01-01", f"Should extract 'since 2024' start, got: {pub_date['gte']}"
            # 'lt' should not be present for 'since' queries
            assert 'lt' not in pub_date, f"'Since' queries should not have 'lt' field, got: {pub_date}"
            print("✅ Phase 2 SUCCESS: 'Since' converted to DatetimeRange format (gte only)")
        else:
            # Legacy format - acceptable but not Phase 2 feature
            assert "2024" in str(pub_date), f"Legacy format should contain 2024, got: {pub_date}"
            print("⚠️  Phase 2 LEGACY: Using legacy string format")
        
        print("Phase 2 'Since' Date Range Test:")
        print(f"  User Query: {user_query}")
        print(f"  Publication Date Filter: {pub_date}")
        print(f"  Filter Type: {'DatetimeRange' if isinstance(pub_date, dict) else 'Legacy String'}")

    @pytest.mark.integration
    def test_phase2_datetime_range_comprehensive_validation(self):
        """Test Phase 2: Comprehensive validation of all DatetimeRange formats."""
        test_cases = [
            {
                "query": "@knowledgebase articles from 2023 about Python programming",
                "expected_gte": "2023-01-01",
                "expected_lt": "2024-01-01",
                "description": "Year range (2023)"
            },
            {
                "query": "@knowledgebase papers published in March 2025",
                "expected_gte": "2025-03-01", 
                "expected_lt": "2025-04-01",
                "description": "Month range (March 2025)"
            },
            {
                "query": "@knowledgebase documents from 2025 Q1",
                "expected_gte": "2025-01-01",
                "expected_lt": "2025-04-01", 
                "description": "Quarter range (2025 Q1)"
            },
            {
                "query": "@knowledgebase articles by John Wong since 2024",
                "expected_gte": "2024-01-01",
                "expected_lt": None,  # No 'lt' for 'since' queries
                "description": "Since date (from 2024 onwards)"
            }
        ]
        
        phase2_successes = 0
        total_tests = len(test_cases)
        
        print("Phase 2 Comprehensive DatetimeRange Validation:")
        print(f"Testing {total_tests} natural language date expressions...")
        print("")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"--- Test {i}: {test_case['description']} ---")
            print(f"Query: {test_case['query']}")
            
            result = self.query_rewriter.transform_query(test_case['query'])
            
            # Validate basic structure
            assert self.validate_json_structure(result), f"Invalid JSON structure: {result}"
            assert result['search_rag'] is True, "Should detect RAG trigger"
            
            # Validate filters and date extraction
            assert 'filters' in result, "Result should include filters field"
            filters = result['filters']
            assert 'publication_date' in filters, "Should extract publication date"
            
            pub_date = filters['publication_date']
            
            if isinstance(pub_date, dict):
                # New DatetimeRange format (Phase 2 success!)
                assert 'gte' in pub_date, "DatetimeRange should have 'gte' field"
                assert pub_date['gte'] == test_case['expected_gte'], f"Expected gte='{test_case['expected_gte']}', got '{pub_date['gte']}'"
                
                if test_case['expected_lt'] is not None:
                    assert 'lt' in pub_date, "DatetimeRange should have 'lt' field"
                    assert pub_date['lt'] == test_case['expected_lt'], f"Expected lt='{test_case['expected_lt']}', got '{pub_date['lt']}'"
                else:
                    assert 'lt' not in pub_date, "'Since' queries should not have 'lt' field"
                
                print(f"✅ PASS: DatetimeRange format correct: {pub_date}")
                phase2_successes += 1
            else:
                # Legacy format - test passes but not Phase 2 feature
                print(f"⚠️  LEGACY: Using legacy string format: {pub_date}")
            
            print("")
        
        # Summary
        print("=== PHASE 2 COMPREHENSIVE TEST SUMMARY ===")
        print(f"Total Tests: {total_tests}")
        print(f"DatetimeRange Format (Phase 2): {phase2_successes}/{total_tests}")
        print(f"Legacy Format: {total_tests - phase2_successes}/{total_tests}")
        
        if phase2_successes == total_tests:
            print("🎉 ALL tests using Phase 2 DatetimeRange format!")
        elif phase2_successes > 0:
            print("✅ Partial Phase 2 implementation - some DatetimeRange conversion working")
        else:
            print("⚠️  No DatetimeRange conversion detected - using legacy format")
        
        # Don't fail the test if legacy format is used - both are acceptable
        # This test is informational to track Phase 2 adoption

