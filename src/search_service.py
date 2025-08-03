"""
Search service with soft filtering and score boosting.

This module implements a unified search service that combines hard filtering,
negation filtering, and soft filtering with weighted diminishing returns boosting.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client.models import ScoredPoint

logger = logging.getLogger(__name__)

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
    Unified search service with hard filtering, negation filtering, and soft filter boosting.
    
    This service combines semantic vector search with metadata filtering and score boosting:
    - Hard filters: Must-match conditions (excludes documents if not matching)
    - Negation filters: Must-NOT-match conditions (excludes documents if matching)  
    - Soft filters: Boost-if-match conditions (increases score but doesn't exclude)
    """
    
    def __init__(self, qdrant_db, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SearchService.
        
        Args:
            qdrant_db: QdrantDB instance for vector search operations
            config: Optional configuration overrides for soft filtering
        """
        self.qdrant_db = qdrant_db
        self.config = {**SOFT_FILTER_CONFIG, **(config or {})}
        logger.info("SearchService initialized with soft filtering enabled")
        
    def unified_search(
        self, 
        query_vector: List[float], 
        top_k: int, 
        hard_filters: Optional[Dict[str, Any]] = None,
        negation_filters: Optional[Dict[str, Any]] = None, 
        soft_filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[ScoredPoint]:
        """
        Unified search with hard filtering, negation filtering, and soft filter boosting.
        
        Args:
            query_vector: The embedding vector for semantic search
            top_k: Number of final results to return
            hard_filters: Must-match metadata filters (excludes if not matching)
            negation_filters: Must-NOT-match metadata filters (excludes if matching)
            soft_filters: Boost-if-match metadata filters (boost score, don't exclude)
            score_threshold: Minimum similarity score threshold
        
        Returns:
            List of top_k ScoredPoint objects, re-ranked by boosted scores
        """
        try:
            # Step 1: Determine fetch limit for re-ranking
            has_soft_filters = soft_filters and any(soft_filters.values())
            fetch_limit = top_k * self.config['fetch_multiplier'] if has_soft_filters else top_k
            
            # Step 2: Apply hard filters and negation filters via Qdrant
            logger.debug(f"Performing initial search with limit={fetch_limit}")
            initial_results = self.qdrant_db.search(
                query_vector=query_vector,
                limit=fetch_limit,
                score_threshold=score_threshold,
                filters=hard_filters,
                negation_filters=negation_filters
            )
            
            if not initial_results:
                logger.info("No results found from initial search")
                return []
            
            logger.info(f"Initial search returned {len(initial_results)} results")
            
            # Step 3: Apply soft filter boosting if soft filters provided
            if has_soft_filters:
                logger.debug("Applying soft filter boosting")
                boosted_results = self._apply_soft_filter_boosting(initial_results, soft_filters)
                
                # Step 4: Re-rank by boosted scores and return top_k
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