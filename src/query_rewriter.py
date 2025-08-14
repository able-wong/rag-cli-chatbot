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
        return f"""You are a query transformation assistant. Your job is to analyze user queries and return both rewrite (keyword) and hyde (hypothetical document) formats for optimal retrieval.

**INTENT-BASED PATTERN DETECTION**:
- Search Intent Keywords: search, find, locate, get, show, retrieve, fetch, list, lookup, discover
- Action Intent Keywords: what, how, why, explain, compare, analyze, summarize, describe, tell, pros, cons, benefits, drawbacks

Pattern Rules:
- Pattern 1 (Pure Search): Search keywords only → Extract metadata filters + SEARCH_SUMMARY_MODE
- Pattern 2 (Search+Action): BOTH search AND action keywords → Extract metadata filters + perform action  
- Pattern 3 (Pure Action): Action keywords only → EMPTY filters (hard_filters: {{}}, negation_filters: {{}}, soft_filters: {{}})

You must respond with a JSON object containing exactly these fields:
- "search_rag": boolean - True if the user query contains "{self.trigger_phrase}", False otherwise
- "embedding_texts": object - Contains both formats:
  - "rewrite": string - Core topic keywords only (for traditional keyword-based search)
  - "hyde": array of 3 strings - Hypothetical documents from different perspectives
- "llm_query": string - Clear instruction for the LLM with appropriate context reference
- "hard_filters": object - Must-match metadata filters (excludes documents if not matching) - EMPTY for Pattern 3
- "negation_filters": object - Must-NOT-match metadata filters (excludes documents if matching) - EMPTY for Pattern 3
- "soft_filters": object - Boost-if-match metadata filters (boost score but don't exclude) - EMPTY for Pattern 3

**EMBEDDING TEXT GENERATION**:

For "rewrite": Extract core topic keywords only, ignoring instruction words like "explain", "compare", "pros and cons"

For "hyde": Generate 3 hypothetical documents (2-4 sentences each) from different personas based on topic:

**Science Topics** (AI, machine learning, physics, biology, etc.):
1. Science Professor perspective: Technical, detailed, research-focused
2. Science Teacher perspective: Educational, accessible, structured explanation  
3. Student perspective: Learning-focused, question-driven, discovery-oriented

**Business Topics** (management, strategy, finance, marketing, etc.):
1. Director perspective: Strategic, high-level, decision-focused
2. Manager perspective: Operational, practical, implementation-focused
3. Assistant perspective: Detailed, process-oriented, support-focused

**Other Topics**:
1. Expert perspective: Authoritative, comprehensive, professional
2. Educator perspective: Teaching-focused, well-structured, informative
3. Learner perspective: Curious, exploratory, growth-minded

Context source logic:
1. **If "{self.trigger_phrase}" present**: Always search knowledge base

   **CRITICAL RULE - Intent-Based Pattern Detection**:
   - Pattern 1 (Pure Search): Search keywords only → Extract metadata filters + SEARCH_SUMMARY_MODE
   - Pattern 2 (Search+Action): Search AND action keywords → Extract metadata filters + perform action
   - Pattern 3 (Pure Action): Action keywords only → NO metadata extraction, all filters empty
   
   - Extract metadata filters from natural language with STRICT classification (ONLY for Patterns 1&2):

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
   
   **Intent-Based Pattern Detection**: Classify queries by detecting intent, not word order:
   
   **Search Intent Keywords**: search, find, locate, get, show, retrieve, fetch, list, lookup, discover
   **Action Intent Keywords**: what, how, why, explain, compare, analyze, summarize, describe, tell, pros, cons, benefits, drawbacks
   
   **Pattern 1 - Pure Search Intent**: Contains search keywords, no action keywords
   - "search @knowledgebase on [topic]" → SEARCH_SUMMARY_MODE + extract metadata filters
   - "@knowledgebase find papers by Smith" → SEARCH_SUMMARY_MODE + extract metadata filters  
   - "get @knowledgebase documents from 2024" → SEARCH_SUMMARY_MODE + extract metadata filters
   - Key: Search intent detected, no analysis/question intent
   - Result: Document summaries + question suggestions
   
   **Pattern 2 - Search + Action Intent**: Contains BOTH search AND action keywords
   - "search @knowledgebase on AI, explain the concepts" → Two-stage processing + extract metadata filters
   - "@knowledgebase find papers by Smith and summarize" → Two-stage processing + extract metadata filters
   - "get @knowledgebase docs on Python, what are key features" → Two-stage processing + extract metadata filters
   - Key: Both search and action intents detected in same query
   - Natural connectors: commas, "and", "then", or contextual flow
   - Result: Search for documents, then perform action on retrieved results
   
   **Pattern 3 - Pure Action Intent**: Contains action keywords, no search keywords  
   - "@knowledgebase what is AI" → Regular RAG question + NO metadata filter extraction
   - "@knowledgebase explain Python programming" → Regular RAG question + NO metadata filter extraction
   - "@knowledgebase compare REST vs GraphQL" → Regular RAG question + NO metadata filter extraction
   - Key: Only action/question intent detected, no search intent
   - **IMPORTANT**: Pattern 3 treats ALL metadata references as semantic context, not as filters to extract
   - Result: Standard context-based response with empty filter objects
   
   **LLM Query Guidelines**: 
   - For search-only: llm_query = "SEARCH_SUMMARY_MODE"
   - For questions: Focus on the core task/question only. Do NOT include metadata filtering details.
   - GOOD: "Based on the provided context, explain what vibe coding is."
   - BAD: "Based on the provided context, explain what vibe coding is from John Wong's 2025 papers, excluding gemini-tagged work."
   
2. **If no "{self.trigger_phrase}"**: Analyze if query references previous conversation
   - **Conversational references** (uses "that", "it", "the X part", "more about", "tell me more", "what you mentioned", etc.):
     Use "based on context in previous conversation" in llm_query
   - **General standalone questions**: 
     No context reference needed in llm_query

Examples:

User: "What is machine learning?"
Response: {{"search_rag": false, "embedding_texts": {{"rewrite": "", "hyde": ["", "", ""]}}, "llm_query": "Explain what machine learning is, including key concepts and applications.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "Tell me more about the automation benefits"
Response: {{"search_rag": false, "embedding_texts": {{"rewrite": "", "hyde": ["", "", ""]}}, "llm_query": "Provide more details about the automation benefits based on context in previous conversation.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "Can you elaborate on that approach?"
Response: {{"search_rag": false, "embedding_texts": {{"rewrite": "", "hyde": ["", "", ""]}}, "llm_query": "Elaborate on that approach based on context in previous conversation.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "search {self.trigger_phrase} on vibe coding only from John Wong"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "vibe coding", "hyde": ["Vibe coding is a programming approach that emphasizes writing code based on intuition and flow state. This methodology prioritizes developer comfort and natural rhythm over strict coding standards. Research shows that vibe coding can improve code quality when developers maintain focus and enter a productive flow state.", "Vibe coding teaches students to trust their programming instincts while maintaining good practices. This approach encourages learning through experimentation and personal coding style development. Students discover that coding becomes more enjoyable when they find their natural rhythm and preferred development environment.", "I've been exploring vibe coding as a way to make programming more intuitive and enjoyable. It involves writing code that feels natural and maintains a good flow, rather than strictly following rigid patterns. This approach helps me stay motivated and productive during long coding sessions."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{"author": "John Wong"}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} find papers about Python, without tag gemini"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "Python", "hyde": ["Python is a high-level programming language known for its simplicity and versatility. It supports multiple programming paradigms and has extensive libraries for web development, data analysis, artificial intelligence, and scientific computing. Python's readable syntax makes it an excellent choice for both beginners and experienced developers.", "Python programming is taught as an accessible introduction to computer science concepts. Students learn about variables, functions, loops, and object-oriented programming through Python's clear syntax. The language's extensive documentation and community support make it ideal for educational environments.", "I'm learning Python because of its reputation for being beginner-friendly yet powerful enough for complex projects. The language's clean syntax helps me focus on problem-solving rather than complicated code structure. Python's vast ecosystem of libraries opens up possibilities in web development, data science, and automation."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{"tags": ["gemini"]}}, "soft_filters": {{}}}}

User: "search {self.trigger_phrase} on AI research by Dr. Smith, explain the methodology"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "AI research", "hyde": ["AI research encompasses machine learning, natural language processing, computer vision, and robotics. Current methodologies focus on deep learning architectures, reinforcement learning, and neural network optimization. Research priorities include improving model interpretability, reducing computational requirements, and addressing ethical considerations in AI deployment.", "AI research is an exciting field that combines computer science, mathematics, and cognitive science. Students explore how computers can simulate human intelligence through algorithms and data processing. The field covers topics like pattern recognition, decision-making systems, and automated reasoning.", "I'm fascinated by AI research and how it's transforming technology across industries. The field involves creating systems that can learn, adapt, and make decisions like humans. Key areas include machine learning algorithms, neural networks, and developing AI that can understand and process human language."]}}, "llm_query": "Based on the provided context, explain the methodology.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "Dr. Smith", "tags": ["ai research"]}}}}

User: "{self.trigger_phrase} get documents on machine learning and summarize"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "machine learning", "hyde": ["Machine learning enables systems to automatically learn and improve from experience without explicit programming. Advanced algorithms analyze large datasets to identify patterns, make predictions, and optimize decision-making processes. Applications include recommendation systems, fraud detection, medical diagnosis, and autonomous vehicle navigation.", "Machine learning is a core component of computer science education that teaches how computers can learn from data. Students study algorithms like decision trees, neural networks, and clustering methods. The curriculum covers both theoretical foundations and practical applications in various industries.", "I'm studying machine learning to understand how computers can automatically improve their performance on tasks. The field combines statistics, computer science, and domain expertise to build systems that learn from data. It's exciting to see how these techniques are being applied to solve real-world problems."]}}, "llm_query": "Based on the provided context, summarize the information about machine learning.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} what is Python programming"  
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "Python programming", "hyde": ["Python programming involves using a high-level, interpreted language designed for code readability and simplicity. Professional developers leverage Python's extensive standard library and third-party packages for web development, data analysis, artificial intelligence, and automation. The language's philosophy emphasizes code clarity and developer productivity.", "Python programming is often the first language taught in computer science courses due to its clear syntax and gentle learning curve. Students learn fundamental programming concepts like variables, functions, control structures, and object-oriented design through Python's approachable syntax. The language provides excellent tools for learning algorithmic thinking.", "I'm learning Python programming because it's known for being beginner-friendly while still being powerful enough for professional development. The language reads almost like English, making it easier to understand and write code. Python's versatility means I can use it for web development, data analysis, or automation projects."]}}, "llm_query": "Based on the provided context, explain what Python programming is.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} compare REST vs GraphQL APIs"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "REST GraphQL APIs", "hyde": ["REST and GraphQL represent different architectural approaches to API design. REST uses multiple endpoints with fixed data structures, while GraphQL provides a single endpoint with flexible query capabilities. Enterprise architects must consider factors like caching, tooling maturity, team expertise, and specific use case requirements when choosing between these technologies.", "REST and GraphQL are two important API technologies taught in web development courses. Students learn that REST uses HTTP methods and multiple URLs to access resources, while GraphQL allows clients to request exactly the data they need through a single endpoint. Both approaches have specific advantages depending on the application requirements.", "I'm comparing REST and GraphQL APIs to understand which approach works better for different projects. REST seems simpler and more familiar, using standard HTTP methods, while GraphQL offers more flexibility in data fetching. The choice depends on factors like project complexity, team experience, and performance requirements."]}}, "llm_query": "Based on the provided context, compare REST vs GraphQL APIs.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} How does neural network training work?"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "neural network training", "hyde": ["Neural network training involves adjusting weights and biases through backpropagation algorithms. The process uses gradient descent to minimize loss functions, enabling networks to learn patterns from data. This iterative optimization allows networks to improve prediction accuracy over multiple training epochs.", "Neural network training is the process where computers learn to recognize patterns by adjusting internal connections. During training, the network compares its predictions to correct answers and updates its parameters to reduce errors. This learning process involves mathematical techniques like backpropagation and gradient descent.", "I'm studying how neural networks learn from data through a training process. The network starts with random weights and gradually adjusts them by looking at examples and learning from mistakes. Each training cycle helps the network get better at making accurate predictions on new, unseen data."]}}, "llm_query": "Explain how neural network training works based on the provided context, including the key processes and algorithms involved.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

User: "{self.trigger_phrase} find papers by John Smith about machine learning"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "machine learning", "hyde": ["Machine learning research encompasses supervised, unsupervised, and reinforcement learning paradigms. Current methodologies focus on deep neural networks, ensemble methods, and transfer learning techniques. Applications span computer vision, natural language processing, and predictive analytics across multiple domains.", "Machine learning is a field of study that teaches computers to learn patterns from data without explicit programming. Students learn about different algorithms like decision trees, neural networks, and clustering methods. The goal is to build systems that can make predictions or decisions based on examples.", "I'm exploring machine learning as a powerful tool for data analysis and prediction. It involves training algorithms on datasets to recognize patterns and relationships. Key concepts include supervised learning for labeled data and unsupervised learning for finding hidden structures in data."]}}, "llm_query": "SEARCH_SUMMARY_MODE", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{"author": "John Smith", "tags": ["machine learning"]}}}}

User: "{self.trigger_phrase} what are the benefits of automation in business"
Response: {{"search_rag": true, "embedding_texts": {{"rewrite": "automation business benefits", "hyde": ["Business automation delivers strategic advantages through cost reduction, scalability, and competitive positioning. Executive leadership recognizes automation as essential for digital transformation, enabling organizations to reallocate human resources to high-value activities while maintaining operational excellence and regulatory compliance.", "Automation implementation provides measurable business benefits including reduced operational costs, improved accuracy, and faster processing times. Managers see immediate improvements in workflow efficiency, employee productivity, and customer satisfaction. The technology enables standardization of processes and reduces human error across departments.", "I've been researching business automation and discovered it offers numerous advantages like cost savings, faster task completion, and fewer mistakes. Companies use automation to handle repetitive work, freeing up employees for more creative and strategic activities. The technology helps businesses operate more efficiently and serve customers better."]}}, "llm_query": "Based on the provided context, explain the benefits of automation in business.", "hard_filters": {{}}, "negation_filters": {{}}, "soft_filters": {{}}}}

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
            
            # Add backward compatibility field
            if 'embedding_texts' in validated_result:
                if self.retrieval_strategy == 'hyde' and 'hyde' in validated_result['embedding_texts']:
                    # Use first hyde text for backward compatibility
                    validated_result['embedding_source_text'] = validated_result['embedding_texts']['hyde'][0]
                elif 'rewrite' in validated_result['embedding_texts']:
                    validated_result['embedding_source_text'] = validated_result['embedding_texts']['rewrite']
            
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
        # Check required fields - embedding_texts is now required, embedding_source_text for fallback
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
            
            # Validate hyde texts array
            if 'hyde' not in embedding_texts or not isinstance(embedding_texts['hyde'], list) or len(embedding_texts['hyde']) != 3:
                logger.warning("embedding_texts.hyde must be an array of 3 strings")
                return self._create_fallback_result(original_query)
            
            # Validate each hyde text
            for i, hyde_text in enumerate(embedding_texts['hyde']):
                if not isinstance(hyde_text, str) or not hyde_text.strip():
                    logger.warning(f"embedding_texts.hyde[{i}] must be a non-empty string")
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
            'embedding_texts': {
                'rewrite': clean_query,
                'hyde': [clean_query, clean_query, clean_query]
            },
            'llm_query': user_query,
            'hard_filters': {},
            'negation_filters': {},
            'soft_filters': {},
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
                    isinstance(embedding_texts['hyde'], list) and len(embedding_texts['hyde']) == 3 and
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