"""
Search service with soft filtering and score boosting.

This module implements a unified search service that combines hard filtering,
negation filtering, and soft filtering with weighted diminishing returns boosting.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from qdrant_client.models import ScoredPoint

logger = logging.getLogger(__name__)

# Type definition for progress callback
ProgressCallback = Callable[[str, Dict[str, Any]], None]

# Soft filtering configuration at top of file for easy tweaking
SOFT_FILTER_CONFIG = {
    # Base boost configuration
    'base_boost_per_match': 0.12,     # 12% boost for first match
    'diminishing_factor': 0.75,       # Each subsequent match worth 75% of previous
    'max_total_boost': 0.6,           # Cap total boost at 60%
    
    # Field importance weights (higher = more important)
    'field_weights': {
        'tags': 1.2,                  # 120% weight - explicit tags very important
        'title': 1.0,                 # 100% weight - title matches important  
        'author': 0.8,                # 80% weight - author somewhat important
        'publication_date': 0.6,      # 60% weight - date less important
        'file_extension': 0.4,        # 40% weight - file type least important
        'default': 0.7                # 70% weight - other fields
    },
    
    # Search configuration
    'fetch_multiplier': 4,            # Fetch 4x more docs for re-ranking
}


class SearchService:
    """
    Search Service responsible for search Vector DB for LLM context retrieval.
    
    This service combines semantic vector search with metadata filtering and score boosting:
    - Hard filters: Must-match conditions (excludes documents if not matching)
    - Negation filters: Must-NOT-match conditions (excludes documents if matching)  
    - Soft filters: Boost-if-match conditions (increases score but doesn't exclude)

    Flow:
    1. Analyze query with QueryRewriter to extract filters and rewrite query if needed
    2. Generate dense (and optionally sparse) embeddings for the query
    3. Perform vector search with hard and negation filters applied, natively supported by Qdrant
    4. If soft filters present, fetch extra results and apply diminishing returns boosting
    5. Re-rank by boosted scores and return top-k results
    """
    
    def __init__(
        self, 
        qdrant_db,
        dense_embedding_client,  # Required
        query_rewriter,  # Required
        sparse_embedding_client=None,  # Optional (can be None if hybrid disabled)
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SearchService with all required providers for self-contained operation.
        
        Args:
            qdrant_db: QdrantDB instance for vector search operations
            dense_embedding_client: EmbeddingClient for dense vector generation
            query_rewriter: QueryRewriter for query analysis and transformation
            sparse_embedding_client: Optional EmbeddingClient for sparse vector generation
            config: Optional configuration overrides for soft filtering
        """
        self.qdrant_db = qdrant_db
        self.dense_embedding_client = dense_embedding_client
        self.query_rewriter = query_rewriter
        self.sparse_embedding_client = sparse_embedding_client
        self.config = {**SOFT_FILTER_CONFIG, **(config or {})}
        logger.info("SearchService initialized with self-contained embedding and query processing")
        
    def unified_search(
        self, 
        query: str,  # Raw query string to process and search
        top_k: int, 
        score_threshold: Optional[float] = None,
        enable_hybrid: Optional[bool] = None,  # Optional override for global config
        progress_callback: Optional[ProgressCallback] = None  # Optional progress updates
    ) -> List[ScoredPoint]:
        """
        Unified search with query processing, embedding generation, and filtering.
        
        Analyzes the query using QueryRewriter to extract filters and rewrite the query,
        then performs vector search with automatic filter application and score boosting.
        
        Args:
            query: Raw query string to process and search
            top_k: Number of final results to return
            score_threshold: Minimum similarity score threshold
            enable_hybrid: Optional override to enable/disable hybrid search
            progress_callback: Optional callback for progress updates during search
        
        Returns:
            List of top_k ScoredPoint objects, re-ranked by boosted scores
        """
        try:
            # Step 1: Analyze query using QueryRewriter
            if progress_callback:
                progress_callback("analyzing", {})
            
            logger.debug(f"Analyzing query: {query}")
            query_analysis = self.query_rewriter.transform_query(query)
            embedding_texts = query_analysis.get('embedding_texts', {})
            strategy = query_analysis.get('strategy', 'rewrite')
            
            # Select embedding text based on strategy
            if strategy == 'hyde' and 'hyde' in embedding_texts and embedding_texts['hyde']:
                embedding_text = embedding_texts['hyde'][0] if isinstance(embedding_texts['hyde'], list) else embedding_texts['hyde']
            else:
                # Default to rewrite text (fallback for both rewrite strategy and hyde failures)
                embedding_text = embedding_texts.get('rewrite', '')
            
            # Step 2: Extract filters from query analysis
            hard_filters = query_analysis.get('hard_filters', {})
            negation_filters = query_analysis.get('negation_filters', {})
            soft_filters = query_analysis.get('soft_filters', {})
            
            if progress_callback:
                progress_callback("query_analyzed", {
                    "embedding_texts": embedding_texts,
                    "embedding_text": embedding_text,
                    "strategy": strategy,
                    "original_query": query,
                    "hard_filters": hard_filters,
                    "negation_filters": negation_filters,
                    "soft_filters": soft_filters,
                    "source": query_analysis.get('source', 'unknown')
                })
            
            # Convert empty dicts to None for cleaner processing
            final_hard_filters = hard_filters if hard_filters else None
            final_negation_filters = negation_filters if negation_filters else None
            final_soft_filters = soft_filters if soft_filters else None
            
            # Step 3: Generate dense embedding (always required)
            logger.debug(f"Generating dense embedding for: {embedding_text}")
            dense_vector = self.dense_embedding_client.get_embedding(embedding_text)
            
            # Step 4: Generate sparse embedding if hybrid enabled
            sparse_vector = None
            hybrid_enabled = enable_hybrid if enable_hybrid is not None else self.config.get('enable_hybrid', False)
            if hybrid_enabled and self.sparse_embedding_client and self.sparse_embedding_client.has_sparse_embedding():
                logger.debug("Generating sparse embedding for hybrid search")
                sparse_vector = self.sparse_embedding_client.get_sparse_embedding(embedding_text)
            
            # Step 5: Determine fetch limit for re-ranking
            has_soft_filters = final_soft_filters and any(final_soft_filters.values())
            fetch_limit = top_k * self.config['fetch_multiplier'] if has_soft_filters else top_k
            
            # Step 6: Perform search with dense (and optionally sparse) vectors
            if progress_callback:
                progress_callback("search_ready", {})
            
            logger.debug(f"Performing vector search with limit={fetch_limit}")
            logger.debug(f"Hard filters being passed to Qdrant: {final_hard_filters}")
            logger.debug(f"Negation filters being passed to Qdrant: {final_negation_filters}")
            initial_results = self._search_with_vectors(
                dense_vector, sparse_vector, fetch_limit, 
                final_hard_filters, final_negation_filters, score_threshold
            )
            
            if not initial_results:
                logger.info("No results found from initial search")
                if progress_callback:
                    progress_callback("search_complete", {
                        "result_count": 0
                    })
                return []
            
            logger.info(f"Initial search returned {len(initial_results)} results")
            if progress_callback:
                progress_callback("search_complete", {
                    "result_count": len(initial_results)
                })
            
            # Step 7: Apply soft filter boosting if soft filters provided
            if has_soft_filters:
                logger.debug("Applying soft filter boosting")
                boosted_results = self._apply_soft_filter_boosting(initial_results, final_soft_filters)
                
                # Step 7: Re-rank by boosted scores and return top_k
                boosted_results.sort(key=lambda x: x.score, reverse=True)
                final_results = boosted_results[:top_k]
                
                logger.info(f"Soft filtering boosted and re-ranked to {len(final_results)} results")
                return final_results
            else:
                logger.debug("No soft filters provided, returning initial results")
                return initial_results
                
        except Exception as e:
            logger.error(f"Unified search failed: {e}")
            return []
    
    def _search_with_vectors(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[str, Any]],
        limit: int,
        hard_filters: Optional[Dict[str, Any]] = None,
        negation_filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[ScoredPoint]:
        """
        Perform vector search with dense and optionally sparse vectors.
        
        Args:
            dense_vector: Dense embedding vector
            sparse_vector: Optional sparse embedding vector for hybrid search
            limit: Number of results to return
            hard_filters: Must-match metadata filters
            negation_filters: Must-NOT-match metadata filters  
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of ScoredPoint objects from vector search
        """
        # For now, use only dense vector search
        # TODO: Implement hybrid search combining dense + sparse vectors when sparse_vector is provided
        return self.qdrant_db.search(
            query_vector=dense_vector,
            limit=limit,
            score_threshold=score_threshold,
            filters=hard_filters,
            negation_filters=negation_filters
        )
    
    def _apply_soft_filter_boosting(
        self, 
        results: List[ScoredPoint], 
        soft_filters: Dict[str, Any]
    ) -> List[ScoredPoint]:
        """
        Apply diminishing returns boosting based on soft filter matches.
        
        Args:
            results: List of ScoredPoint objects from initial search
            soft_filters: Dictionary of soft filter conditions
            
        Returns:
            List of ScoredPoint objects with boosted scores
        """
        boosted_results = []
        boost_stats = {'total_boosted': 0, 'max_boost': 0.0, 'avg_boost': 0.0}
        total_boost = 0.0
        
        for result in results:
            boost_multiplier = self._calculate_boost_multiplier(result.payload, soft_filters)
            
            if boost_multiplier > 0:
                boost_stats['total_boosted'] += 1
                boost_stats['max_boost'] = max(boost_stats['max_boost'], boost_multiplier)
                total_boost += boost_multiplier
            
            # Create new ScoredPoint with boosted score
            boosted_score = result.score * (1 + boost_multiplier)
            boosted_result = ScoredPoint(
                id=result.id,
                version=result.version,
                score=boosted_score,
                payload=result.payload,
                vector=result.vector
            )
            boosted_results.append(boosted_result)
        
        # Calculate statistics
        if boost_stats['total_boosted'] > 0:
            boost_stats['avg_boost'] = total_boost / boost_stats['total_boosted']
        
        logger.debug(
            f"Boost statistics: {boost_stats['total_boosted']}/{len(results)} documents boosted, "
            f"max boost: {boost_stats['max_boost']:.3f}, avg boost: {boost_stats['avg_boost']:.3f}"
        )
        
        return boosted_results
    
    def _calculate_boost_multiplier(
        self, 
        document_payload: Dict[str, Any], 
        soft_filters: Dict[str, Any]
    ) -> float:
        """
        Calculate boost multiplier using diminishing returns with field weights.
        
        Args:
            document_payload: Document metadata from Qdrant
            soft_filters: Dictionary of soft filter conditions
            
        Returns:
            Boost multiplier (0.0 to max_total_boost)
        """
        matches = []
        
        # Collect all soft filter matches with their weights
        for field, value in soft_filters.items():
            if self._field_matches(document_payload, field, value):
                field_weight = self.config['field_weights'].get(
                    field, 
                    self.config['field_weights']['default']
                )
                matches.append(field_weight)
                logger.debug(f"Soft filter match: {field}={value} (weight: {field_weight})")
        
        if not matches:
            return 0.0
        
        # Calculate diminishing returns boost
        total_boost = 0.0
        base_boost = self.config['base_boost_per_match']
        diminishing = self.config['diminishing_factor']
        
        # Sort matches by weight (highest weight gets full boost)
        matches.sort(reverse=True)
        for i, weight in enumerate(matches):
            match_boost = base_boost * weight * (diminishing ** i)
            total_boost += match_boost
            logger.debug(f"Match {i+1}: weight={weight:.2f}, boost={match_boost:.4f}")
        
        # Apply maximum boost cap
        capped_boost = min(total_boost, self.config['max_total_boost'])
        
        if capped_boost != total_boost:
            logger.debug(f"Boost capped: {total_boost:.4f} -> {capped_boost:.4f}")
        
        return capped_boost
    
    def _field_matches(
        self, 
        document_payload: Dict[str, Any], 
        field: str, 
        value: Any
    ) -> bool:
        """
        Check if document field matches soft filter value.
        
        Args:
            document_payload: Document metadata from Qdrant
            field: Field name to check
            value: Filter value to match against
            
        Returns:
            True if field matches, False otherwise
        """
        doc_value = document_payload.get(field)
        if doc_value is None:
            return False
        
        try:
            if field == 'tags':
                # Tags: check if any soft filter tag matches any document tag
                if isinstance(value, list) and isinstance(doc_value, list):
                    doc_tags_lower = [tag.lower() for tag in doc_value]
                    return any(tag.lower() in doc_tags_lower for tag in value)
                elif isinstance(value, str) and isinstance(doc_value, list):
                    doc_tags_lower = [tag.lower() for tag in doc_value]
                    return value.lower() in doc_tags_lower
                    
            elif field == 'author':
                # Author: case-insensitive partial match
                if isinstance(doc_value, str) and isinstance(value, str):
                    return value.lower() in doc_value.lower()
                    
            elif field == 'publication_date':
                # Date: handle both string and DatetimeRange formats
                return self._date_matches(doc_value, value)
                
            elif field == 'title':
                # Title: case-insensitive partial match
                if isinstance(doc_value, str) and isinstance(value, str):
                    return value.lower() in doc_value.lower()
                    
            else:
                # Default: case-insensitive string match
                return str(value).lower() in str(doc_value).lower()
                
        except Exception as e:
            logger.warning(f"Error matching field {field}: {e}")
            return False
        
        return False
    
    def _date_matches(self, doc_date: Any, filter_date: Any) -> bool:
        """
        Check if document date falls within filter date range.
        
        Args:
            doc_date: Document date value (string or datetime)
            filter_date: Filter date value (string, dict with gte/lt, etc.)
            
        Returns:
            True if date matches, False otherwise
        """
        try:
            # Handle different date formats
            # This should match the logic used in qdrant_db.py for consistency
            
            if isinstance(filter_date, dict):
                # DatetimeRange format: {"gte": "2025-01-01", "lt": "2026-01-01"}
                doc_date_str = str(doc_date) if doc_date else ""
                
                if 'gte' in filter_date:
                    if doc_date_str < filter_date['gte']:
                        return False
                        
                if 'lt' in filter_date:
                    if doc_date_str >= filter_date['lt']:
                        return False
                        
                return True
                
            elif isinstance(filter_date, str):
                # String format: exact or partial match
                doc_date_str = str(doc_date) if doc_date else ""
                return filter_date in doc_date_str
                
            else:
                # Fallback: convert both to strings and compare
                return str(filter_date).lower() in str(doc_date).lower()
                
        except Exception as e:
            logger.warning(f"Error matching date {doc_date} with {filter_date}: {e}")
            return False
    
    def get_boost_statistics(self) -> Dict[str, Any]:
        """
        Get current boost configuration for debugging/monitoring.
        
        Returns:
            Dictionary with boost configuration details
        """
        return {
            'base_boost_per_match': self.config['base_boost_per_match'],
            'diminishing_factor': self.config['diminishing_factor'],
            'max_total_boost': self.config['max_total_boost'],
            'field_weights': self.config['field_weights'].copy(),
            'fetch_multiplier': self.config['fetch_multiplier']
        }
    
    def update_boost_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update boost configuration at runtime.
        
        Args:
            new_config: Dictionary with configuration updates
        """
        self.config.update(new_config)
        logger.info(f"Boost configuration updated: {new_config}")