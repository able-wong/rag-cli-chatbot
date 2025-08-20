import logging
from typing import Dict, Any
from llm_client import LLMClient

logger = logging.getLogger(__name__)

class QueryRewriterDebugError(Exception):
    """Exception raised in debug mode with detailed LLM response information."""
    def __init__(self, failure_type: str, message: str, raw_response: str = None, parsed_data: Dict = None):
        self.failure_type = failure_type
        self.raw_response = raw_response
        self.parsed_data = parsed_data
        super().__init__(message)

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
        self.max_tokens = config.get('max_tokens', 4096)   # Large buffer for thinking models and complex JSON responses
        
        self.system_prompt = self._build_system_prompt()
        logger.info(f"QueryRewriter initialized with strategy: {self.retrieval_strategy}")
    
    
    def _build_system_prompt(self) -> str:
        """Build the unified system prompt for query transformation."""
        return self._build_unified_system_prompt()
    

    def _build_unified_system_prompt(self) -> str:
        """Build the unified system prompt for query transformation that returns both rewrite and hyde formats."""
        return f"""You are an advanced query rewriter designed to transform user queries into a JSON response for retrieval-augmented generation (RAG) systems. Your task is to analyze user prompts, extract relevant information, and generate structured outputs that can be used for both RAG search and LLM generation.
        
## 1. Overview

IF user prompt contains "{self.trigger_phrase}":
- STEP 1: Split user prompt into RAG search criteria and LLM instructions
- STEP 2: Extract filters from RAG search criteria (author, tags, publication_date)
- STEP 3: Generate embedding texts (rewrite keywords and hyde answers) from remaining content

IF user prompt does NOT contain "{self.trigger_phrase}":
- Set "search_rag": false
- Set all filters to empty objects
- Set embedding_texts to empty values 
- Transform user query into clean LLM instruction format

**PROCESSING ORDER**: Always complete filter extraction BEFORE generating embedding texts to avoid conflicts.

## 2. Response Format

**JSON Structure (JSON only):**
- "search_rag": boolean - True if query contains "{self.trigger_phrase}" (see Section 3 for trigger detection)
- "embedding_texts": object - Contains following fields:
  - "rewrite": string - see section 5.1 for generation details
  - "hyde": array - see section 5.2 for generation details
- "llm_query": string - LLM instruction (see Section 5.3 for extraction rules)
- "hard_filters": object - Must-match filters (see Section 4 for filter extraction)
- "negation_filters": object - Must-NOT-match filters (see Section 4 for filter extraction)
- "soft_filters": object - Boost-if-match filters (see Section 4 for filter extraction)

(Detailed instructions for each field are provided in the sections referenced above)

## 3. Prompt Splitting Rules

Use natural language understanding to identify user intent and split accordingly into one of 3 patterns:

1. **Document Discovery Intent** - User wants to find/browse documents
   - User is seeking to discover or locate documents/information  
   - Split: Document-seeking portion → RAG criteria | Use "SEARCH_SUMMARY_MODE" → LLM instruction

2. **Search + Analysis Intent** - User wants to find documents AND perform analysis
   - User wants to retrieve information and then do something with it
   - Split: Retrieval portion → RAG criteria | Analytical action → LLM instruction

3. **Direct Question Intent** - User asks direct questions about topics
   - User is asking for explanations or information about concepts
   - Split: Question topic → RAG criteria | Question format → LLM instruction

**Splitting Examples:**
- "find papers by Smith" → RAG: "papers by Smith" | LLM: "SEARCH_SUMMARY_MODE"
- "locate research by Smith and explain findings" → RAG: "research by Smith" | LLM: "explain findings"  
- "{self.trigger_phrase} what is machine learning" → RAG: "machine learning" | LLM: "what is machine learning"

## 4. Filter Extraction

Extract filters from RAG search criteria only when "{self.trigger_phrase}" present in user prompt.

**Supported filterable fields:** author, tags, publication_date

**Filter Types:**
- **Soft filters**: DEFAULT for ALL mentions - boost relevance but don't exclude ("papers by Smith", "from 2024", "tagged Python", "about python")
- **Hard filters**: Use when user expresses restrictive/exclusive intent (examples: "only", "just", "exclusively" and similar limiting language). KEY: Analyze what the restrictive word directly modifies - if "only from Smith, published in 2025", the word "only" modifies "from Smith" but not "published in 2025".
- **Negation filters**: Use when user expresses exclusion/avoidance intent (examples: "not", "without", "except" and similar excluding language)

**Tag Detection:**
- Extract topic words as tags: "papers about python" → tags=["python"]
- Extract explicit tags: "papers with tag python" → tags=["python"] 
- Multiple topics: "articles about python and machine learning" → tags=["python", "machine learning"]

**Date Format Conversion:**
For publication_date filters, convert date mentions to start/end date range format:
- "2024" → {{"gte": "2024-01-01", "lt": "2025-01-01"}}
- "from 2024" → {{"gte": "2024-01-01"}}
- "before 2024" → {{"lt": "2024-01-01"}}
- "in March 2024" → {{"gte": "2024-03-01", "lt": "2024-04-01"}}
- Use best guess for ambiguous dates and always provide gte/lt format when possible

**Filter Scope Examples:**
- "articles by Smith from 2024" → author=soft, date=soft (both descriptive)
- "papers about python by Smith" → tags=soft, author=soft (both descriptive) 
- "papers about python and ML ONLY by Smith" → tags=soft, author=hard (restrictive applies to author only)
- "papers ONLY by Smith, from 2024" → author=hard, date=soft (restrictive applies to author, date is descriptive)
- "research exclusively from 2024 by Smith" → date=hard, author=soft (restrictive applies to date, author is descriptive)

KEY RULE: Use natural language understanding to detect intent. Everything goes to soft_filters unless clearly expressing restrictive or exclusion intent.

## 5. Content Generation

### 5.1 embedding_texts.rewrite text Generation
- "rewrite": Extract clean topic keywords from RAG search criteria. REQUIRED - NEVER leave empty. Remove:
  * Action words: "find", "search", "get", "show", "locate", "retrieve"
  * Stop words: "the", "and", "of", "in", "from", "by", "about"  
  * Author names and dates (but KEEP topic words even if they were extracted as tags)
  * If removal results in empty string, use the original RAG search criteria as fallback
  * Example: "papers about python by Smith from 2025" → "papers python" (keep "python" even though it's also a tag)

### 5.2 embedding_texts.hyde array Generation
- "hyde": Generate 3 DISTINCT answers about the topics from the "rewrite" string from section 5.1 (3-4 sentences each), each from a DIFFERENT expert perspective/persona. Choose appropriate personas based on the topic (examples: Professor/Teacher/Student for science, Director/Manager/Assistant for business, Expert/Educator/Learner for other topics). ALWAYS provide your best answer even if unsure - use your knowledge to generate helpful content.

### 5.3 LLM Instruction Extraction
- For Document Discovery Intent: Use placeholder "SEARCH_SUMMARY_MODE" (caller will convert to document summary instructions)
- For Search + Analysis Intent: Extract the analytical action and format as "Based on the provided context, [action]"
- For Direct Question Intent: Format as "Based on the provided context, [question]"
- Remove trigger phrase ("{self.trigger_phrase}") from all instructions
- Remove filter clauses from instructions

## 6. Examples

IMPORTANT: The following examples show proper JSON structure with real knowledge-based content. Generate similar high-quality content for your responses. Your hyde entries should be complete, informative text without any placeholder markers or brackets.

User: "What is Python?"
Response: {{"search_rag": false, "embedding_texts": {{"rewrite": "", "hyde": ["", "", ""]}}, "llm_query": "Explain what Python is.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} find papers by Smith"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "papers", "hyde": ["Research papers are scholarly documents that present original findings and contribute to academic knowledge. They undergo peer review to ensure quality and provide evidence-based insights for the academic community.", "Academic publications serve as the foundation for scientific discourse and knowledge sharing. They document experimental procedures, theoretical frameworks, and help advance understanding in their respective fields.", "Scientific documents provide evidence-based insights through rigorous research methodologies. They enable knowledge transfer between researchers and facilitate reproducibility of research results."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Smith"}}}}

User: "{self.trigger_phrase} papers about python by Smith from 2024"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "papers python", "hyde": ["Research papers about Python programming cover diverse topics from basic syntax to advanced frameworks. They provide comprehensive insights into Python's versatility across web development, data science, and automation, making it accessible for both beginners and experienced developers.", "Academic publications on Python demonstrate its effectiveness in solving complex computational problems. These documents showcase Python's extensive library ecosystem and its role in machine learning, artificial intelligence, and scientific computing applications.", "Scientific literature about Python programming illustrates its practical applications across various industries. These papers document best practices, performance optimizations, and innovative solutions that leverage Python's readable syntax and powerful capabilities."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"tags": ["python"], "author": "Smith", "publication_date": {{"gte": "2024-01-01", "lt": "2025-01-01"}}}}}}

User: "{self.trigger_phrase} papers ONLY from 2024, not by Johnson"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "papers", "hyde": ["Contemporary research papers reflect current trends and innovative approaches to complex scientific problems. They incorporate advanced technologies and interdisciplinary perspectives in their methodologies.", "Recent academic publications demonstrate the evolving landscape of scientific inquiry, providing insights into modern research practices and emerging theoretical frameworks that shape future research directions.", "Current research documents present timely findings that address contemporary challenges and opportunities, helping to shape the trajectory of academic research and scientific advancement."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{"publication_date": {{"gte": "2024-01-01", "lt": "2025-01-01"}}}}, "negation_filters": {{"author": "Johnson"}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} explain machine learning"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "machine learning", "hyde": ["As a computer science professor, I can tell you that machine learning represents a fundamental shift from traditional programming paradigms. Instead of explicitly coding every rule, we design algorithms that can learn patterns from data and improve their performance over time through statistical optimization.", "From a business perspective, machine learning is a powerful tool for gaining competitive advantage and driving innovation. Companies use ML to personalize customer experiences, optimize operations, and make data-driven decisions that directly impact revenue and efficiency.", "As a student learning about AI, I find machine learning fascinating because it mimics how humans learn from experience. The algorithms start with basic parameters and gradually become more accurate as they process more examples, much like how we improve at tasks through practice."]}}, "llm_query": "Based on the provided context, explain machine learning.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

**CRITICAL INSTRUCTIONS:**
- For "hyde" array: Generate ONLY your own knowledge-based content. Do NOT include placeholder text, brackets, or reference markers.
- Response format: Always respond with ONLY valid JSON. Do NOT add explanatory text, notes, or content after the JSON."""

    def transform_query(self, user_query: str, rethrow_on_error: bool = False) -> Dict[str, Any]:
        """
        Transform a user query into structured format for RAG processing.
        
        Args:
            user_query: The original user query
            rethrow_on_error: If True, raise exceptions instead of returning fallback result
            
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
            validated_result = self._validate_result(result, user_query, rethrow_on_error)
            
            # Convert structured llm_query to final, ready-to-use prompt
            validated_result['llm_query'] = self._build_final_llm_query(validated_result['llm_query'])
            
            
            # Add source indicator (only if not already set by fallback)
            if 'source' not in validated_result:
                validated_result['source'] = 'llm'
            
            logger.info(f"Query transformed successfully. RAG search: {validated_result['search_rag']}")
            return validated_result
            
        except QueryRewriterDebugError:
            # Re-raise QueryRewriterDebugError unchanged (validation and other debug errors)
            raise
        except Exception as e:
            logger.error(f"Query transformation failed: {e}")
            
            # If rethrow_on_error is True, raise the exception instead of fallback
            if rethrow_on_error:
                # Extract raw response from exception if available
                raw_response = getattr(e, 'raw_response', str(e))
                cleaned_response = getattr(e, 'cleaned_response', None)
                
                # Create enhanced error with debugging information
                enhanced_error = QueryRewriterDebugError(
                    failure_type="transform_error",
                    message=f"Query transformation failed: {e}",
                    raw_response=raw_response,
                    parsed_data={"cleaned_response": cleaned_response} if cleaned_response else None
                )
                raise enhanced_error from e
            
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
    
    def _validate_result(self, result: Dict[str, Any], original_query: str, rethrow_on_error: bool = False) -> Dict[str, Any]:
        """
        Validate and sanitize the transformation result.
        
        Args:
            result: Parsed JSON result from LLM
            original_query: Original user query for fallback
            rethrow_on_error: If True, raise exceptions instead of returning fallback
            
        Returns:
            Validated result dictionary
        """
        # Check required fields - embedding_texts is now required
        required_fields = ['search_rag', 'llm_query']
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing field '{field}' in transformation result")
                if rethrow_on_error:
                    raise QueryRewriterDebugError(
                        failure_type="validation_error",
                        message=f"Missing required field '{field}' in LLM response",
                        raw_response=str(result),
                        parsed_data=result
                    )
                return self._create_fallback_result(original_query)
        
        # Check for new embedding_texts structure
        if 'embedding_texts' not in result:
            logger.warning("Missing 'embedding_texts' field in transformation result")
            if rethrow_on_error:
                raise QueryRewriterDebugError(
                    failure_type="validation_error",
                    message="Missing 'embedding_texts' field in LLM response",
                    raw_response=str(result),
                    parsed_data=result
                )
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
                if rethrow_on_error:
                    raise QueryRewriterDebugError(
                        failure_type="validation_error",
                        message="embedding_texts field must be a dictionary",
                        raw_response=str(result),
                        parsed_data=result
                    )
                return self._create_fallback_result(original_query)
            
            # Validate rewrite text
            if 'rewrite' not in embedding_texts or not isinstance(embedding_texts['rewrite'], str) or not embedding_texts['rewrite'].strip():
                logger.warning("embedding_texts.rewrite is required for RAG queries")
                if rethrow_on_error:
                    raise QueryRewriterDebugError(
                        failure_type="validation_error",
                        message="embedding_texts.rewrite is required for RAG queries and must be non-empty string",
                        raw_response=str(result),
                        parsed_data=result
                    )
                return self._create_fallback_result(original_query)
            
            # Validate and clean hyde texts array
            if 'hyde' not in embedding_texts or not isinstance(embedding_texts['hyde'], list):
                logger.warning("embedding_texts.hyde must be an array")
                if rethrow_on_error:
                    raise QueryRewriterDebugError(
                        failure_type="validation_error",
                        message="embedding_texts.hyde must be an array",
                        raw_response=str(result),
                        parsed_data=result
                    )
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
                if rethrow_on_error:
                    raise QueryRewriterDebugError(
                        failure_type="validation_error",
                        message="No valid hyde texts generated - all entries were empty or invalid",
                        raw_response=str(result),
                        parsed_data=result
                    )
                return self._create_fallback_result(original_query)
            
            # Update with cleaned hyde texts
            embedding_texts['hyde'] = valid_hyde_texts
            logger.debug(f"Using {len(valid_hyde_texts)} valid hyde texts out of {len(embedding_texts.get('hyde', []))}")
        
        # Validate llm_query is not empty
        if not isinstance(result['llm_query'], str) or not result['llm_query'].strip():
            logger.warning("llm_query field is empty or invalid")
            if rethrow_on_error:
                raise QueryRewriterDebugError(
                    failure_type="validation_error",
                    message="llm_query field is empty or invalid",
                    raw_response=str(result),
                    parsed_data=result
                )
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