import logging
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
        self.retrieval_strategy = config.get('retrieval_strategy', 'rewrite')
        
        # Query rewriter specific settings
        self.temperature = config.get('temperature', 0.1)  # Low temp for consistent JSON
        self.max_tokens = config.get('max_tokens', 512)    # Shorter responses for structured output
        
        self.system_prompt = self._build_system_prompt()
        logger.info(f"QueryRewriter initialized with strategy: {self.retrieval_strategy}")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for query transformation based on retrieval strategy."""
        return self._get_strategy_prompt_builder()
    
    def _get_strategy_prompt_builder(self) -> str:
        """
        Strategy selection method that encapsulates the logic for choosing the appropriate
        system prompt based on the configured retrieval strategy.
        
        Returns:
            The appropriate system prompt string for the configured strategy
        """
        strategy_map = {
            'hyde': self._build_hyde_system_prompt,
            'rewrite': self._build_rewrite_system_prompt
        }
        
        prompt_builder = strategy_map.get(self.retrieval_strategy, self._build_rewrite_system_prompt)
        return prompt_builder()
    
    def _build_shared_instructions(self) -> str:
        """Build the shared instructions used by both strategies."""
        return f"""Context source logic:
1. **If "{self.trigger_phrase}" present**: Always search knowledge base
   - Extract metadata filters from natural language with STRICT classification:

   **HARD FILTERS** - Only for explicit restrictive keywords (excludes if not matching):
   - Keywords: "only", "exclusively", "strictly", "limited to", "solely", "must be"
   - Examples: "papers ONLY from 2025", "articles EXCLUSIVELY by Smith", "documents STRICTLY tagged AI"
   
   **NEGATION FILTERS** - For clear negation keywords (excludes if matching):
   - Keywords: "not from", "not by", "excluding", "except", "without", "other than"
   - Examples: "not from John Wong", "excluding Smith", "except papers by Johnson"
   
   **SOFT FILTERS** - DEFAULT for everything else (boost score but don't exclude):
   - All other author/date/tag mentions: "papers by Smith", "from 2025", "with tags AI", "about Python"
   - Examples: "papers by Smith" → soft_filters, "from 2025" → soft_filters, "tagged as ML" → soft_filters

   **Date format conversion** (same for all filter types):
   - Years: "2023" → {{"publication_date": {{"gte": "2023-01-01", "lt": "2024-01-01"}}}}
   - Quarters: "2025 Q1" → {{"publication_date": {{"gte": "2025-01-01", "lt": "2025-04-01"}}}}
   - Months: "March 2025" → {{"publication_date": {{"gte": "2025-03-01", "lt": "2025-04-01"}}}}
   - Since: "since 2024" → {{"publication_date": {{"gte": "2024-01-01"}}}}
   - Before: "before 2024" → {{"publication_date": {{"lt": "2024-01-01"}}}}

   **Key principle**: Everything goes to soft_filters unless explicitly restrictive or negated.
   
   **Search-Only Query Detection**: Detect queries that only specify search criteria without asking questions:
   - **Search-only patterns**: "search", "find", "show me", "get", "retrieve" + filters but NO question words
   - **Question indicators**: "what", "how", "why", "when", "where", "explain", "compare", "summarize", "pros and cons"
   - **If search-only detected**: Use "SEARCH_SUMMARY_MODE" as llm_query (exactly this string)
   - **If has questions**: Use "based on the provided context" + question in llm_query
   
   **LLM Query Guidelines**: 
   - For search-only: llm_query = "SEARCH_SUMMARY_MODE"
   - For questions: Focus on the core task/question only. Do NOT include metadata filtering details.
   - GOOD: "Based on the provided context, explain what vibe coding is."
   - BAD: "Based on the provided context, explain what vibe coding is from John Wong's 2025 papers, excluding gemini-tagged work."
   
2. **If no "{self.trigger_phrase}"**: Analyze if query references previous conversation
   - **Conversational references** (uses "that", "it", "the X part", "more about", "tell me more", "what you mentioned", etc.):
     Use "based on context in previous conversation" in llm_query
   - **General standalone questions**: 
     No context reference needed in llm_query"""

    def _build_shared_examples(self) -> str:
        """Build the shared examples used by both strategies.""" 
        return f"""User: "What is machine learning?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Explain what machine learning is, including key concepts and applications.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "Tell me more about the automation benefits"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Provide more details about the automation benefits based on context in previous conversation.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "Can you elaborate on that approach?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Elaborate on that approach based on context in previous conversation.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} search papers by John Wong about vibe coding"
Response: {{"search_rag": true, "embedding_source_text": "vibe coding", "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "John Wong", "tags": ["vibe coding"]}}}}

User: "{self.trigger_phrase} find documents only from 2025, without tag gemini"
Response: {{"search_rag": true, "embedding_source_text": "documents 2025", "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{"publication_date": {{"gte": "2025-01-01", "lt": "2026-01-01"}}}}, "negation_filters": {{"tags": ["gemini"]}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} show me research by Dr. Smith on AI"
Response: {{"search_rag": true, "embedding_source_text": "AI research", "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Dr. Smith", "tags": ["ai"]}}}}

Always respond with valid JSON only. Do not include any other text or formatting."""

    def _build_rewrite_system_prompt(self) -> str:
        """Build the system prompt for the default rewrite strategy."""
        return f"""You are a query transformation assistant. Your job is to analyze user queries and determine the appropriate context source for responses.

You must respond with a JSON object containing exactly these fields:
- "search_rag": boolean - True if the user query contains "{self.trigger_phrase}", False otherwise
- "embedding_source_text": string - Core topic keywords only, ignoring instruction words like "explain", "pros and cons" (only needed if search_rag=true)
- "llm_query": string - Clear instruction for the LLM with appropriate context reference
- "hard_filters": object - Must-match metadata filters (excludes documents if not matching)
- "negation_filters": object - Must-NOT-match metadata filters (excludes documents if matching)
- "soft_filters": object - Boost-if-match metadata filters (boost score but don't exclude)

{self._build_shared_instructions()}

Examples:

User: "{self.trigger_phrase} How does neural network training work?"
Response: {{"search_rag": true, "embedding_source_text": "neural network training", "llm_query": "Explain how neural network training works based on the provided context, including the key processes and algorithms involved.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} papers by John Smith about machine learning"
Response: {{"search_rag": true, "embedding_source_text": "machine learning", "llm_query": "Based on the provided context, provide information about machine learning.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "John Smith", "tags": ["machine learning"]}}}}

User: "{self.trigger_phrase} articles ONLY from 2023 about Python programming"
Response: {{"search_rag": true, "embedding_source_text": "Python programming", "llm_query": "Based on the provided context, provide information about Python programming.", "hard_filters": {{"publication_date": {{"gte": "2023-01-01", "lt": "2024-01-01"}}}}, "negation_filters": {{}}, "soft_filters": {{"tags": ["python", "programming"]}}}}

User: "{self.trigger_phrase} papers published only in 2025 on vibe coding, not from John Wong, with tags gemini"
Response: {{"search_rag": true, "embedding_source_text": "vibe coding programming approach", "llm_query": "Based on the provided context, explain what vibe coding is.", "hard_filters": {{"publication_date": {{"gte": "2025-01-01", "lt": "2026-01-01"}}}}, "negation_filters": {{"author": "John Wong"}}, "soft_filters": {{"tags": ["gemini"]}}}}

User: "{self.trigger_phrase} research EXCLUSIVELY by Dr. Johnson, excluding robotics work"
Response: {{"search_rag": true, "embedding_source_text": "Dr Johnson research", "llm_query": "Based on the provided context, provide information about Dr. Johnson's research.", "hard_filters": {{"author": "Dr. Johnson"}}, "negation_filters": {{"tags": ["robotics"]}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} articles by Smith from 2024 about AI"
Response: {{"search_rag": true, "embedding_source_text": "AI artificial intelligence", "llm_query": "Based on the provided context, provide information about AI.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Smith", "publication_date": {{"gte": "2024-01-01", "lt": "2025-01-01"}}, "tags": ["ai", "artificial intelligence"]}}}}

User: "{self.trigger_phrase} papers with tags python, AI, not by Johnson"
Response: {{"search_rag": true, "embedding_source_text": "python AI artificial intelligence", "llm_query": "Based on the provided context, provide information about Python and AI.", "hard_filters": {{}}, "negation_filters": {{"author": "Johnson"}}, "soft_filters": {{"tags": ["python", "ai"]}}}}

User: "{self.trigger_phrase} documents STRICTLY tagged machine learning from March 2025"
Response: {{"search_rag": true, "embedding_source_text": "machine learning", "llm_query": "Based on the provided context, provide information about machine learning.", "hard_filters": {{"tags": ["machine learning"], "publication_date": {{"gte": "2025-03-01", "lt": "2025-04-01"}}}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} what are the differences between REST and GraphQL APIs?"  
Response: {{"search_rag": true, "embedding_source_text": "REST GraphQL APIs", "llm_query": "Compare and contrast REST and GraphQL APIs based on the provided context, highlighting their key differences, advantages, and use cases.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} summarize pros and cons of neural networks"
Response: {{"search_rag": true, "embedding_source_text": "neural networks", "llm_query": "Based on the provided context, summarize the pros and cons of neural networks.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} papers only from John Wong, without tag gemini, then summarize pros and cons of vibe coding"
Response: {{"search_rag": true, "embedding_source_text": "vibe coding", "llm_query": "Based on the provided context, summarize the pros and cons of vibe coding.", "hard_filters": {{"author": "John Wong"}}, "negation_filters": {{"tags": ["gemini"]}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} get papers by Smith from 2024"
Response: {{"search_rag": true, "embedding_source_text": "papers Smith 2024", "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Smith", "publication_date": {{"gte": "2024-01-01", "lt": "2025-01-01"}}}}}}

User: "{self.trigger_phrase} retrieve documents tagged python, not by Johnson"
Response: {{"search_rag": true, "embedding_source_text": "python documents", "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{"author": "Johnson"}}, "soft_filters": {{"tags": ["python"]}}}}

{self._build_shared_examples()}"""
    
    def _build_hyde_system_prompt(self) -> str:
        """Build the system prompt for HyDE (Hypothetical Document Embeddings) strategy."""
        return f"""You are a query transformation assistant using the HyDE (Hypothetical Document Embeddings) strategy. Your job is to analyze user queries and generate hypothetical documents that would answer their questions.

You must respond with a JSON object containing exactly these fields:
- "search_rag": boolean - True if the user query contains "{self.trigger_phrase}", False otherwise
- "embedding_source_text": string - A focused 2-4 sentence hypothetical document that answers the core topic (only needed if search_rag=true)
- "llm_query": string - Clear instruction for the LLM with appropriate context reference
- "hard_filters": object - Must-match metadata filters (excludes documents if not matching)
- "negation_filters": object - Must-NOT-match metadata filters (excludes documents if matching)
- "soft_filters": object - Boost-if-match metadata filters (boost score but don't exclude)

Context source logic:
1. **If "{self.trigger_phrase}" present**: Always search knowledge base
   - Generate a hypothetical document/passage that would answer the user's question
   - Use "based on the provided context" in llm_query
   - Extract metadata filters from natural language with STRICT classification:

   **HARD FILTERS** - Only for explicit restrictive keywords (excludes if not matching):
   - Keywords: "only", "exclusively", "strictly", "limited to", "solely", "must be"
   - Examples: "papers ONLY from 2025", "articles EXCLUSIVELY by Smith", "documents STRICTLY tagged AI"
   
   **NEGATION FILTERS** - For clear negation keywords (excludes if matching):
   - Keywords: "not from", "not by", "excluding", "except", "without", "other than"
   - Examples: "not from John Wong", "excluding Smith", "except papers by Johnson"
   
   **SOFT FILTERS** - DEFAULT for everything else (boost score but don't exclude):
   - All other author/date/tag mentions: "papers by Smith", "from 2025", "with tags AI", "about Python"
   - Examples: "papers by Smith" → soft_filters, "from 2025" → soft_filters, "tagged as ML" → soft_filters

   **Date format conversion** (same for all filter types):
   - Years: "2023" → {{"publication_date": {{"gte": "2023-01-01", "lt": "2024-01-01"}}}}
   - Quarters: "2025 Q1" → {{"publication_date": {{"gte": "2025-01-01", "lt": "2025-04-01"}}}}
   - Months: "March 2025" → {{"publication_date": {{"gte": "2025-03-01", "lt": "2025-04-01"}}}}
   - Since: "since 2024" → {{"publication_date": {{"gte": "2024-01-01"}}}}
   - Before: "before 2024" → {{"publication_date": {{"lt": "2024-01-01"}}}}

   **Key principle**: Everything goes to soft_filters unless explicitly restrictive or negated.
   
   **LLM Query Guidelines**: Focus on the core task/question only. Do NOT include metadata filtering details in the llm_query. The filtering is handled separately.
   - GOOD: "Based on the provided context, explain what vibe coding is."
   - BAD: "Based on the provided context, explain what vibe coding is from John Wong's 2025 papers, excluding gemini-tagged work."
   
2. **If no "{self.trigger_phrase}"**: Analyze if query references previous conversation
   - **Conversational references** (uses "that", "it", "the X part", "more about", "tell me more", "what you mentioned", etc.):
     Use "based on context in previous conversation" in llm_query
   - **General standalone questions**: 
     No context reference needed in llm_query

Examples:

User: "What is machine learning?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Explain what machine learning is, including key concepts and applications.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} How does neural network training work?"
Response: {{"search_rag": true, "embedding_source_text": "Neural network training is a process where the network learns from data through backpropagation and gradient descent. During training, the network adjusts its weights and biases by calculating the error between predicted and actual outputs, then propagating this error backward through the layers. The gradient descent algorithm optimizes these parameters iteratively to minimize the loss function, enabling the network to improve its predictions over time.", "llm_query": "Explain how neural network training works based on the provided context, including the key processes and algorithms involved.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} papers by John Smith about machine learning"
Response: {{"search_rag": true, "embedding_source_text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns, make predictions, and improve performance through experience. Common applications include image recognition, natural language processing, and predictive analytics across various industries.", "llm_query": "Based on the provided context, provide information about machine learning.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "John Smith", "tags": ["machine learning"]}}}}

User: "{self.trigger_phrase} papers published only in 2025 on vibe coding, not from John Wong, with tags gemini"
Response: {{"search_rag": true, "embedding_source_text": "Vibe coding is a programming approach that emphasizes writing code based on intuition, flow state, and personal rhythm rather than strict methodologies. This coding style prioritizes developer comfort, creativity, and maintaining a natural coding rhythm. Practitioners focus on writing code that feels right and maintains consistent energy levels during development sessions.", "llm_query": "Based on the provided context, explain what vibe coding is.", "hard_filters": {{"publication_date": {{"gte": "2025-01-01", "lt": "2026-01-01"}}}}, "negation_filters": {{"author": "John Wong"}}, "soft_filters": {{"tags": ["gemini"]}}}}

User: "{self.trigger_phrase} research EXCLUSIVELY by Dr. Johnson, excluding robotics work"
Response: {{"search_rag": true, "embedding_source_text": "Dr. Johnson's research focuses on advanced computational methods and algorithm development across multiple domains. His work spans machine learning optimization, data structures, and software engineering methodologies. The research emphasizes practical applications and theoretical foundations, contributing significantly to the field through innovative approaches and comprehensive analysis of complex problems.", "llm_query": "Based on the provided context, provide information about Dr. Johnson's research.", "hard_filters": {{"author": "Dr. Johnson"}}, "negation_filters": {{"tags": ["robotics"]}}, "soft_filters": {{}}}}

User: "Tell me more about the automation benefits"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Provide more details about the automation benefits based on context in previous conversation.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "Can you elaborate on that approach?"
Response: {{"search_rag": false, "embedding_source_text": "", "llm_query": "Elaborate on that approach based on context in previous conversation.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} what are the differences between REST and GraphQL APIs?"  
Response: {{"search_rag": true, "embedding_source_text": "REST and GraphQL are both API design approaches with distinct characteristics. REST uses multiple endpoints with fixed data structures and HTTP methods, making it simple and cacheable but potentially leading to over-fetching or under-fetching of data. GraphQL uses a single endpoint with flexible queries that allow clients to request exactly the data they need, providing better performance and developer experience but with increased complexity and learning curve.", "llm_query": "Compare and contrast REST and GraphQL APIs based on the provided context, highlighting their key differences, advantages, and use cases.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

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
            
            # Get JSON response directly (handles parsing and markdown wrappers)
            logger.debug(f"Transforming query: {user_query}")
            result = self.llm_client.get_json_response(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Validate and return result
            validated_result = self._validate_result(result, user_query)
            
            # Convert structured llm_query to final, ready-to-use prompt
            validated_result['llm_query'] = self._build_final_llm_query(validated_result['llm_query'])
            
            logger.info(f"Query transformed successfully. RAG search: {validated_result['search_rag']}")
            return validated_result
            
        except Exception as e:
            logger.error(f"Query transformation failed: {e}")
            return self._create_fallback_result(user_query)
    
    def _build_final_llm_query(self, llm_prompt: str) -> str:
        """Convert structured llm_prompt to final, ready-to-use prompt."""
        if llm_prompt == "SEARCH_SUMMARY_MODE":
            return """If no context documents are provided, inform the user that no documents were found matching their criteria and suggest they refine their search terms, check spelling, or try broader keywords.

If context documents are provided, please:
1. **Document Summary**: Provide a concise 2-3 sentence summary of what these documents contain
2. **Key Topics**: List 3-4 main topics/themes found across the documents  
3. **Question Suggestions**: Suggest 3-4 specific questions I could ask about this content

Format your response clearly with the sections above."""
        else:
            return llm_prompt  # Pass through regular prompts unchanged
    
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
        
        # Ensure all filter fields exist
        if 'hard_filters' not in result:
            result['hard_filters'] = {}
        if 'negation_filters' not in result:
            result['negation_filters'] = {}
        if 'soft_filters' not in result:
            result['soft_filters'] = {}
        
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
        
        # Validate all filter fields are dictionaries
        filter_fields = ['hard_filters', 'negation_filters', 'soft_filters']
        for field in filter_fields:
            if not isinstance(result[field], dict):
                logger.warning(f"{field} field is not a dictionary, resetting to empty dict")
                result[field] = {}
        
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
            'llm_query': user_query,
            'hard_filters': {},
            'negation_filters': {},
            'soft_filters': {}
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
            
            # Try to get JSON response directly  
            parsed_result = self.llm_client.get_json_response(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Validate the test result has required fields
            expected_fields = ['search_rag', 'embedding_source_text', 'llm_query', 'hard_filters', 'negation_filters', 'soft_filters']
            if all(field in parsed_result for field in expected_fields):
                logger.info("QueryRewriter connection test successful")
                return True
            else:
                logger.error("QueryRewriter test failed: missing required fields")
                return False
            
        except Exception as e:
            logger.error(f"QueryRewriter connection test failed: {e}")
            return False