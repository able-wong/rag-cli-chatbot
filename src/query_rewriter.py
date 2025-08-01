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
        return f"""You are a query transformation assistant. Your job is to analyze user queries and determine the appropriate context source for responses.

You must respond with a JSON object containing exactly these fields:
- "search_rag": boolean - True if the user query contains "{self.trigger_phrase}", False otherwise
- "embedding_source_text": string - Concise keywords optimized for vector similarity search (only needed if search_rag=true)
- "llm_query": string - Clear instruction for the LLM with appropriate context reference

Context source logic:
1. **If "{self.trigger_phrase}" present**: Always search knowledge base
   - Use "based on the provided context" in llm_query
   
2. **If no "{self.trigger_phrase}"**: Analyze if query references previous conversation
   - **Conversational references** (uses "that", "it", "the X part", "more about", "tell me more", "what you mentioned", etc.):
     Use "based on context in previous conversation" in llm_query
   - **General standalone questions**: 
     No context reference needed in llm_query

Examples:

User: "What is machine learning?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Explain what machine learning is, including key concepts and applications."}}

User: "{self.trigger_phrase} How does neural network training work?"
Response: {{"search_rag": true, "embedding_source_text": "neural network training backpropagation gradient descent", "llm_query": "Explain how neural network training works based on the provided context, including the key processes and algorithms involved."}}

User: "Tell me more about the automation benefits"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Provide more details about the automation benefits based on context in previous conversation."}}

User: "Can you elaborate on that approach?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Elaborate on that approach based on context in previous conversation."}}

User: "What's the weather today?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "What's the weather today?"}}

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
        
        # Validate embedding_source_text (only required for RAG queries)
        if result['search_rag']:
            if not isinstance(result['embedding_source_text'], str) or not result['embedding_source_text'].strip():
                logger.warning("embedding_source_text is required for RAG queries")
                return self._create_fallback_result(original_query)
        
        # Validate llm_query is not empty
        if not isinstance(result['llm_query'], str) or not result['llm_query'].strip():
            logger.warning("llm_query field is empty or invalid")
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
        
        # Only trigger RAG if there's content after the trigger phrase
        search_rag = has_trigger and bool(clean_query)
        
        fallback_result = {
            'search_rag': search_rag,
            'embedding_source_text': clean_query,
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