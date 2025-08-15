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
        """Build the unified system prompt for query transformation."""
        return self._build_unified_system_prompt()
    

    def _build_unified_system_prompt(self) -> str:
        """Build the unified system prompt for query transformation that returns both rewrite and hyde formats."""
        return f"""Transform user queries into structured format for retrieval. Return both keyword and hypothetical document formats.

**RESPONSE FORMAT** (JSON only):
- "search_rag": boolean - True if query contains "{self.trigger_phrase}"
- "embedding_texts": {{"rewrite": "keywords", "hyde": ["doc1", "doc2", "doc3"]}}
- "llm_query": string - LLM instruction
- "hard_filters": object - Must-match filters
- "negation_filters": object - Must-NOT-match filters  
- "soft_filters": object - Boost-if-match filters

**EMBEDDING GENERATION**:
- "rewrite": Extract core topic keywords only
- "hyde": Generate 3 short hypothetical documents (1-2 sentences each) from different perspectives:
  * Science topics: Professor/Teacher/Student views
  * Business topics: Director/Manager/Assistant views
  * Other topics: Expert/Educator/Learner views

**FILTER RULES** (only when "{self.trigger_phrase}" present):
- **Soft filters**: DEFAULT for ALL mentions (authors, dates, tags) - "papers by Smith", "from 2024"
- **Hard filters**: ONLY explicit restrictions ("papers ONLY from 2024", "exclusively by Smith")
- **Negation filters**: ONLY clear negations ("not from Smith", "excluding 2024")
- **Empty filters**: For pure questions without search intent

KEY RULE: Everything goes to soft_filters unless explicitly restrictive or negated.

**PATTERNS**:
1. Pure search: "find papers by Smith" → SEARCH_SUMMARY_MODE + extract filters
2. Search + action: "find papers by Smith and explain" → action + extract filters
3. Pure action: "explain machine learning" → action + empty filters

**EXAMPLES**:

User: "What is Python?"
Response: {{"search_rag": false, "embedding_texts": {{"rewrite": "", "hyde": ["", "", ""]}}, "llm_query": "Explain what Python is.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} find papers by Smith"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "papers", "hyde": ["Research papers contain peer-reviewed findings.", "Academic papers present systematic research.", "Papers document experimental results."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Smith"}}}}

User: "{self.trigger_phrase} papers by Smith from 2024"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "papers Smith 2024", "hyde": ["Research papers by Smith from 2024.", "Smith's 2024 publications show recent work.", "2024 papers by Smith present new findings."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Smith", "publication_date": {{"gte": "2024-01-01", "lt": "2025-01-01"}}}}}}

User: "{self.trigger_phrase} papers ONLY from 2024, not by Johnson"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "papers 2024", "hyde": ["2024 papers show recent research.", "Current papers reflect latest findings.", "Recent papers demonstrate new methods."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{"publication_date": {{"gte": "2024-01-01", "lt": "2025-01-01"}}}}, "negation_filters": {{"author": "Johnson"}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} explain machine learning"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "machine learning", "hyde": ["Machine learning uses algorithms to find patterns in data.", "ML teaches computers to learn from examples.", "Studying ML helps understand automated learning systems."]}}, "llm_query": "Based on the provided context, explain machine learning.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

Always respond with valid JSON only."""

    def transform_query(self, user_query: str) -> Dict[str, Any]:
        """
        Transform a user query into structured format for RAG processing.
        
        Args:
            user_query: The original user query
            
        Returns:
            Dict containing:
            - search_rag: boolean indicating if RAG search should be performed
            - embedding_texts: dict with 'rewrite' and 'hyde' texts for vector search
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
            
            
            # Add source indicator (only if not already set by fallback)
            if 'source' not in validated_result:
                validated_result['source'] = 'llm'
            
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
        # Check required fields - embedding_texts is now required
        required_fields = ['search_rag', 'llm_query']
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field '{field}' in transformation result")
                return self._create_fallback_result(original_query)
        
        # Check for new embedding_texts structure
        if 'embedding_texts' not in result:
            logger.warning("Missing 'embedding_texts' field in transformation result")
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
        
        # Validate embedding_texts structure (only required for RAG queries)
        if result['search_rag']:
            embedding_texts = result.get('embedding_texts', {})
            if not isinstance(embedding_texts, dict):
                logger.warning("embedding_texts must be a dictionary")
                return self._create_fallback_result(original_query)
            
            # Validate rewrite text
            if 'rewrite' not in embedding_texts or not isinstance(embedding_texts['rewrite'], str) or not embedding_texts['rewrite'].strip():
                logger.warning("embedding_texts.rewrite is required for RAG queries")
                return self._create_fallback_result(original_query)
            
            # Validate and clean hyde texts array
            if 'hyde' not in embedding_texts or not isinstance(embedding_texts['hyde'], list):
                logger.warning("embedding_texts.hyde must be an array")
                return self._create_fallback_result(original_query)
            
            # Filter out empty/invalid hyde texts and keep only valid ones
            valid_hyde_texts = []
            for i, hyde_text in enumerate(embedding_texts['hyde']):
                if isinstance(hyde_text, str) and hyde_text.strip():
                    valid_hyde_texts.append(hyde_text.strip())
                else:
                    logger.debug(f"Filtering out empty/invalid hyde text at index {i}")
            
            # Require at least one valid hyde text for RAG queries
            if len(valid_hyde_texts) == 0:
                logger.warning("No valid hyde texts generated, falling back")
                return self._create_fallback_result(original_query)
            
            # Update with cleaned hyde texts
            embedding_texts['hyde'] = valid_hyde_texts
            logger.debug(f"Using {len(valid_hyde_texts)} valid hyde texts out of {len(embedding_texts.get('hyde', []))}")
        
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
        
        # Ensure strategy field is always present with configured value
        if 'strategy' not in result:
            logger.debug(f"Adding missing strategy field: {self.retrieval_strategy}")
            result['strategy'] = self.retrieval_strategy
        
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
            'embedding_texts': {
                'rewrite': clean_query,
                'hyde': [clean_query, clean_query, clean_query]
            },
            'llm_query': user_query,
            'hard_filters': {},
            'negation_filters': {},
            'soft_filters': {},
            'strategy': self.retrieval_strategy,  # Use configured strategy even in fallback
            'source': 'fallback'
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
            
            # Test the complete transform_query process
            result = self.transform_query(test_query)
            
            # Validate the final result has required fields including source
            expected_fields = ['search_rag', 'embedding_texts', 'llm_query', 'hard_filters', 'negation_filters', 'soft_filters', 'source']
            if all(field in result for field in expected_fields):
                # Also check embedding_texts structure
                embedding_texts = result.get('embedding_texts', {})
                if ('rewrite' in embedding_texts and 'hyde' in embedding_texts and 
                    isinstance(embedding_texts['hyde'], list) and len(embedding_texts['hyde']) >= 1 and
                    result['source'] == 'llm'):
                    logger.info("QueryRewriter connection test successful")
                    return True
                else:
                    logger.error("QueryRewriter test failed: invalid structure or not from LLM")
                    return False
            else:
                logger.error("QueryRewriter test failed: missing required fields")
                return False
            
        except Exception as e:
            logger.error(f"QueryRewriter connection test failed: {e}")
            return False