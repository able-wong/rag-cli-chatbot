"""
Unit tests for CLI logic with mocked dependencies.
Tests the core RAG logic without requiring external services.
"""

import sys
import os
from unittest.mock import Mock, patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli import RAGCLI

class MockScoredPoint:
    """Mock for Qdrant ScoredPoint."""
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload

def test_rag_trigger_detection():
    """Test RAG trigger phrase detection."""
    print("Testing RAG trigger detection...")
    
    # Mock all dependencies
    with patch('src.cli.ConfigManager'), \
         patch('src.cli.EmbeddingClient'), \
         patch('src.cli.QdrantDB'), \
         patch('src.cli.LLMClient'), \
         patch('src.cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)  # Create without calling __init__
        cli.trigger_phrase = "@knowledgebase"
        
        # Test positive cases
        assert cli._detect_rag_trigger("@knowledgebase what is Python?") == True
        assert cli._detect_rag_trigger("Can you @knowledgebase tell me about AI?") == True
        assert cli._detect_rag_trigger("@KNOWLEDGEBASE search") == True  # Case insensitive
        
        # Test negative cases
        assert cli._detect_rag_trigger("What is Python?") == False
        assert cli._detect_rag_trigger("Tell me about AI") == False
        
        print("✓ RAG trigger detection test PASSED")
        return True

def test_rag_context_decision():
    """Test decision logic for using RAG context."""
    print("Testing RAG context decision logic...")
    
    with patch('src.cli.ConfigManager'), \
         patch('src.cli.EmbeddingClient'), \
         patch('src.cli.QdrantDB'), \
         patch('src.cli.LLMClient'), \
         patch('src.cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.min_score = 0.7
        
        # Test with high score results
        high_score_results = [
            MockScoredPoint(0.85, {"content": "Test content"}),
            MockScoredPoint(0.75, {"content": "More content"})
        ]
        assert cli._should_use_rag_context(high_score_results) == True
        
        # Test with low score results
        low_score_results = [
            MockScoredPoint(0.65, {"content": "Test content"}),
            MockScoredPoint(0.55, {"content": "More content"})
        ]
        assert cli._should_use_rag_context(low_score_results) == False
        
        # Test with empty results
        assert cli._should_use_rag_context([]) == False
        
        print("✓ RAG context decision test PASSED")
        return True

def test_conversation_history_management():
    """Test conversation history length management."""
    print("Testing conversation history management...")
    
    with patch('src.cli.ConfigManager'), \
         patch('src.cli.EmbeddingClient'), \
         patch('src.cli.QdrantDB'), \
         patch('src.cli.LLMClient'), \
         patch('src.cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.max_history_length = 2  # Keep only 2 user-assistant pairs
        
        # Create a conversation history that exceeds the limit
        cli.conversation_history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Response 3"},
            {"role": "user", "content": "Message 4"},
            {"role": "assistant", "content": "Response 4"},
        ]
        
        cli._manage_conversation_history()
        
        # Should keep system prompt + last 2 pairs (4 messages) = 5 total
        assert len(cli.conversation_history) == 5
        assert cli.conversation_history[0]["role"] == "system"
        assert cli.conversation_history[-1]["content"] == "Response 4"
        
        print("✓ Conversation history management test PASSED")
        return True

def test_rag_context_building():
    """Test RAG context string building."""
    print("Testing RAG context building...")
    
    with patch('src.cli.ConfigManager'), \
         patch('src.cli.EmbeddingClient'), \
         patch('src.cli.QdrantDB'), \
         patch('src.cli.LLMClient'), \
         patch('src.cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.max_context_length = 200  # Small limit for testing
        
        results = [
            MockScoredPoint(0.9, {
                "content": "Short content", 
                "source": "doc1.txt"
            }),
            MockScoredPoint(0.8, {
                "content": "This is a longer piece of content that might exceed the context length limit when combined with other results",
                "source": "doc2.txt"
            })
        ]
        
        context = cli._build_rag_context(results)
        
        # Should include at least the first result
        assert "Short content" in context
        assert "Source 1" in context
        assert len(context) <= cli.max_context_length + 100  # Some tolerance
        
        print("✓ RAG context building test PASSED")
        return True

def test_prompt_building():
    """Test prompt building with and without RAG."""
    print("Testing prompt building...")
    
    with patch('src.cli.ConfigManager'), \
         patch('src.cli.EmbeddingClient'), \
         patch('src.cli.QdrantDB'), \
         patch('src.cli.LLMClient'), \
         patch('src.cli.setup_logging'):
        
        cli = RAGCLI.__new__(RAGCLI)
        cli.trigger_phrase = "@knowledgebase"
        
        # Test RAG prompt building
        query = "@knowledgebase What is Python?"
        context = "Python is a programming language."
        
        rag_prompt = cli._build_prompt_with_rag(query, context)
        assert "Context:" in rag_prompt
        assert "Python is a programming language" in rag_prompt
        assert "What is Python?" in rag_prompt
        
        # Test no-answer prompt building
        no_answer_prompt = cli._build_no_answer_prompt(query)
        assert "couldn't find relevant information" in no_answer_prompt
        assert "What is Python?" in no_answer_prompt
        
        print("✓ Prompt building test PASSED")
        return True

def main():
    """Run all CLI unit tests with mocked dependencies."""
    print("=== CLI Unit Tests (Mocked) ===")
    print()
    
    results = []
    
    results.append(test_rag_trigger_detection())
    print()
    
    results.append(test_rag_context_decision())
    print()
    
    results.append(test_conversation_history_management())
    print()
    
    results.append(test_rag_context_building())
    print()
    
    results.append(test_prompt_building())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All CLI unit tests PASSED!")
        return True
    else:
        print("✗ Some CLI unit tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)