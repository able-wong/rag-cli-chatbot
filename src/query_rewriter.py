import logging
import json
import re
from typing import Dict, Any
from llm_client import LLMClient

logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    Service for transforming user queries using LLM to optimize RAG retrieval.
    Analyzes queries to detect trigger phrases and transforms them into structured
    search queries and generation prompts.
    """
    
    def __init__(self, llm_client: LLMClient, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.trigger_phrase = config.get('trigger_phrase', '@knowledgebase')
        
        # Query rewriter specific settings
        self.temperature = config.get('temperature', 0.1)  # Low temp for consistent JSON
        self.max_tokens = config.get('max_tokens', 512)    # Shorter responses for structured output
        
        self.system_prompt = self._build_system_prompt()
        logger.info("QueryRewriter initialized")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for query transformation."""
        return f"""You are a query transformation assistant. Your job is to analyze user queries and optimize them for both vector search and LLM generation.

You must respond with a JSON object containing exactly these fields:
- "search_rag": boolean - True if the user query contains "{self.trigger_phrase}", False otherwise
- "embedding_source_text": string - Concise keywords optimized for vector similarity search
- "llm_query": string - Clear instruction for the LLM to generate the final response

Key principles:
1. search_rag: Simply check if "{self.trigger_phrase}" appears in the user query
2. embedding_source_text: Extract core concepts and keywords that would match relevant documents. Focus on nouns, topics, and key terms rather than questions.
3. llm_query: For RAG queries (search_rag=true), always include "based on the provided context" to ensure context-grounded responses. For non-RAG queries, create general instructions.

Examples:

User: "What is machine learning?"
Response: {{"search_rag": false, "embedding_source_text": "machine learning definition algorithms", "llm_query": "Explain what machine learning is, including key concepts and applications."}}

User: "{self.trigger_phrase} How does neural network training work?"
Response: {{"search_rag": true, "embedding_source_text": "neural network training backpropagation gradient descent", "llm_query": "Explain how neural network training works based on the provided context, including the key processes and algorithms involved."}}

User: "{self.trigger_phrase} according to the docs on vibe coding, what are the benefits?"
Response: {{"search_rag": true, "embedding_source_text": "vibe coding benefits advantages", "llm_query": "List and explain the benefits of vibe coding based on the provided context."}}

User: "{self.trigger_phrase} tell me about Python's memory management"
Response: {{"search_rag": true, "embedding_source_text": "Python memory management garbage collection", "llm_query": "Describe Python's memory management system based on the provided context, including how it handles memory allocation and garbage collection."}}

User: "{self.trigger_phrase} what are the differences between REST and GraphQL APIs?"  
Response: {{"search_rag": true, "embedding_source_text": "REST GraphQL API differences comparison", "llm_query": "Compare and contrast REST and GraphQL APIs based on the provided context, highlighting their key differences, advantages, and use cases."}}

Always respond with valid JSON only. Do not include any other text or formatting."""

    def transform_query(self, user_query: str) -> Dict[str, Any]:
        """
        Transform a user query into structured format for RAG processing.
        
        Args:
            user_query: The original user query
            
        Returns:
            Dict containing:
            - search_rag: boolean indicating if RAG search should be performed
            - embedding_source_text: optimized text for vector search
            - llm_query: refined prompt for LLM generation
        """
        try:
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            # Get LLM response
            logger.debug(f"Transforming query: {user_query}")
            response = self.llm_client.get_llm_response(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse JSON response
            result = self._parse_json_response(response)
            
            # Validate and return result
            validated_result = self._validate_result(result, user_query)
            logger.info(f"Query transformed successfully. RAG search: {validated_result['search_rag']}")
            return validated_result
            
        except Exception as e:
            logger.error(f"Query transformation failed: {e}")
            return self._create_fallback_result(user_query)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM, handling various formats.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try to parse as direct JSON first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON-like content in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from response: {response}")
    
    def _validate_result(self, result: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Validate and sanitize the transformation result.
        
        Args:
            result: Parsed JSON result from LLM
            original_query: Original user query for fallback
            
        Returns:
            Validated result dictionary
        """
        # Check required fields
        required_fields = ['search_rag', 'embedding_source_text', 'llm_query']
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field '{field}' in transformation result")
                return self._create_fallback_result(original_query)
        
        # Validate search_rag is boolean
        if not isinstance(result['search_rag'], bool):
            logger.warning("search_rag field is not boolean, converting")
            # Try to convert string boolean values
            if isinstance(result['search_rag'], str):
                result['search_rag'] = result['search_rag'].lower() in ('true', '1', 'yes')
            else:
                result['search_rag'] = bool(result['search_rag'])
        
        # Validate string fields are not empty
        for field in ['embedding_source_text', 'llm_query']:
            if not isinstance(result[field], str) or not result[field].strip():
                logger.warning(f"Field '{field}' is empty or invalid")
                return self._create_fallback_result(original_query)
        
        # Double-check search_rag logic against actual trigger detection
        actual_has_trigger = self.trigger_phrase.lower() in original_query.lower()
        if result['search_rag'] != actual_has_trigger:
            logger.warning(f"LLM trigger detection mismatch. Expected: {actual_has_trigger}, Got: {result['search_rag']}")
            result['search_rag'] = actual_has_trigger
        
        return result
    
    def _create_fallback_result(self, user_query: str) -> Dict[str, Any]:
        """
        Create fallback result when transformation fails.
        
        Args:
            user_query: Original user query
            
        Returns:
            Fallback result dictionary
        """
        # Simple trigger detection
        has_trigger = self.trigger_phrase.lower() in user_query.lower()
        
        # Clean query for embedding (remove trigger phrase)
        clean_query = user_query.replace(self.trigger_phrase, '').strip()
        embedding_text = clean_query if clean_query else user_query
        
        fallback_result = {
            'search_rag': has_trigger,
            'embedding_source_text': embedding_text,
            'llm_query': user_query
        }
        
        logger.info("Using fallback query transformation")
        return fallback_result
    
    def test_connection(self) -> bool:
        """
        Test the query rewriter by transforming a simple test query.
        
        Returns:
            True if transformation works, False otherwise
        """
        try:
            test_query = f"{self.trigger_phrase} What is artificial intelligence?"
            
            # Try to call LLM directly to test actual connection, not fallback
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": test_query}
            ]
            
            response = self.llm_client.get_llm_response(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Try to parse the response
            parsed_result = self._parse_json_response(response)
            
            # Validate the test result has required fields
            expected_fields = ['search_rag', 'embedding_source_text', 'llm_query']
            if all(field in parsed_result for field in expected_fields):
                logger.info("QueryRewriter connection test successful")
                return True
            else:
                logger.error("QueryRewriter test failed: missing required fields")
                return False
            
        except Exception as e:
            logger.error(f"QueryRewriter connection test failed: {e}")
            return False