import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, Filter, FieldCondition, MatchValue, MatchAny, DatetimeRange, Prefetch, FusionQuery, Fusion, SparseVector

logger = logging.getLogger(__name__)

class QdrantDB:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get('collection_name', 'knowledge_base')
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.client = None
        self._initialize_client()
        self._validate_payload_indexes()
    
    def _initialize_client(self):
        """Initialize Qdrant client based on configuration."""
        try:
            # Check if cloud URL is provided
            if 'url' in self.config and self.config['url']:
                # Qdrant Cloud setup
                self.client = QdrantClient(
                    url=self.config['url'],
                    api_key=self.config.get('api_key')
                )
                logger.info(f"Initialized Qdrant cloud client: {self.config['url']}")
            else:
                # Local Qdrant setup
                host = self.config.get('host', 'localhost')
                port = self.config.get('port', 6333)
                self.client = QdrantClient(host=host, port=port)
                logger.info(f"Initialized Qdrant local client: {host}:{port}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant server."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def create_collection(self, vector_size: int) -> bool:
        """Create a collection with the specified vector size."""
        try:
            if self.collection_exists():
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Map distance metric string to Qdrant Distance enum
            distance_map = {
                'cosine': Distance.COSINE,
                'euclidean': Distance.EUCLID,
                'dot': Distance.DOT
            }
            
            distance = distance_map.get(self.distance_metric, Distance.COSINE)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            logger.info(f"Created collection '{self.collection_name}' with vector size {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def search(
        self, 
        query_vector: List[float], 
        limit: int = 5, 
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        negation_filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPoint]:
        """
        Search for similar vectors in the collection with optional metadata filtering.
        
        This method now uses hybrid_search() internally with RRF fusion for improved ranking.
        It is kept for backward compatibility and easier single-vector calling.
        
        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Dictionary of positive filters (must match)
            negation_filters: Dictionary of negation filters (must NOT match)
        
        Returns:
            List of ScoredPoint objects matching the criteria
        """
        # Convert single vector search to hybrid search for RRF benefits
        return self.hybrid_search(
            dense_vectors=[query_vector],  # Single vector becomes multi-vector
            sparse_vector=None,           # Skip sparse vector completely
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
            negation_filters=negation_filters
        )

    def hybrid_search(
        self,
        dense_vectors: List[List[float]],  # Keywords + HyDE personas
        sparse_vector: Dict[str, List[int]],  # Keywords only: {"indices": [...], "values": [...]}
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        negation_filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPoint]:
        """
        Perform hybrid search using Qdrant's native RRF fusion with dense and sparse vectors.
        
        Args:
            dense_vectors: List of dense embeddings (keywords + HyDE personas)
            sparse_vector: Sparse vector dictionary with indices and values
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Dictionary of positive filters (must match)
            negation_filters: Dictionary of negation filters (must NOT match)
        
        Returns:
            List of ScoredPoint objects ranked by RRF fusion
        """
        try:
            if not self.collection_exists():
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return []
            
            # Check if we have any vectors to search with
            if not dense_vectors:
                logger.warning("No dense vectors provided for hybrid search")
                return []
            
            # Build prefetch queries for all dense vectors
            prefetch_queries = []
            
            # Add dense vector prefetches (keywords + HyDE personas)
            for i, dense_vector in enumerate(dense_vectors):
                prefetch_queries.append(
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit * 2  # Fetch more for better RRF fusion
                    )
                )
            
            # Add sparse vector prefetch if provided
            if sparse_vector and sparse_vector.get("indices") and sparse_vector.get("values"):
                sparse_query = SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"]
                )
                prefetch_queries.append(
                    Prefetch(
                        query=sparse_query,
                        using="sparse",
                        limit=limit * 2
                    )
                )
                logger.info(f"Added sparse vector to hybrid search with {len(sparse_vector['indices'])} dimensions")
            else:
                logger.warning("Sparse vector not provided or invalid, using dense-only hybrid search")
            
            # Build query parameters
            query_params = {
                "collection_name": self.collection_name,
                "prefetch": prefetch_queries,
                "query": FusionQuery(fusion=Fusion.RRF),
                "limit": limit,
                "with_payload": True
            }
            
            if score_threshold is not None:
                query_params["score_threshold"] = score_threshold
            
            # Build combined filter with both positive and negative conditions
            if filters or negation_filters:
                qdrant_filter = self._build_combined_filter(filters, negation_filters)
                if qdrant_filter:
                    # Apply filter to all prefetch queries
                    for prefetch in prefetch_queries:
                        prefetch.filter = qdrant_filter
                        
                    # Log applied filters
                    applied_filters = []
                    if filters:
                        applied_filters.extend([f"+{k}" for k in filters.keys()])
                    if negation_filters:
                        applied_filters.extend([f"-{k}" for k in negation_filters.keys()])
                    logger.info(f"Applied hybrid search filters: {applied_filters}")
            
            # Perform hybrid search with RRF fusion
            results = self.client.query_points(**query_params).points
            
            logger.info(f"Hybrid search found {len(results)} results using RRF fusion with {len(dense_vectors)} dense vectors")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def scroll_collection(self, limit: int = 10) -> List[PointStruct]:
        """Scroll through collection to get sample points (for testing)."""
        try:
            if not self.collection_exists():
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return []
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            points = results[0]  # First element contains the points
            logger.info(f"Retrieved {len(points)} points from collection")
            return points
            
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return []
    
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection."""
        try:
            if not self.collection_exists():
                return None
            
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'status': info.status,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def _validate_payload_indexes(self):
        """Validate that required payload indexes exist for hybrid search functionality."""
        required_indexes = ['tags', 'author', 'publication_date']
        
        try:
            if not self.collection_exists():
                logger.info("Collection does not exist yet, skipping payload index validation")
                return
            
            # Get collection info to check payload schema
            collection_info = self.client.get_collection(self.collection_name)
            payload_schema = getattr(collection_info.config, 'params', {}).get('payload_schema', {})
            
            missing_indexes = []
            for field in required_indexes:
                if field not in payload_schema:
                    missing_indexes.append(field)
            
            if missing_indexes:
                logger.warning(
                    f"Missing payload indexes for hybrid search: {missing_indexes}. "
                    f"Hybrid search will work but may have degraded performance. "
                    f"These indexes should be created during document ingestion."
                )
            else:
                logger.info("All required payload indexes for hybrid search are present")
                
        except Exception as e:
            logger.warning(f"Could not validate payload indexes: {e}. Hybrid search may have degraded performance.")
    
    def _build_combined_filter(self, filters: Optional[Dict[str, Any]], negation_filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """
        Build a combined Qdrant Filter object with both positive and negative conditions.
        
        Args:
            filters: Dictionary containing positive filter conditions (must match)
            negation_filters: Dictionary containing negative filter conditions (must NOT match)
        
        Returns:
            Qdrant Filter object or None if no valid filters
        """
        must_conditions = []
        must_not_conditions = []
        
        try:
            # Build positive conditions (must match)
            if filters:
                positive_conditions = self._build_filter_conditions(filters)
                must_conditions.extend(positive_conditions)
            
            # Build negative conditions (must NOT match)
            if negation_filters:
                negative_conditions = self._build_filter_conditions(negation_filters)
                must_not_conditions.extend(negative_conditions)
            
            # Create Filter object with both must and must_not conditions
            if must_conditions or must_not_conditions:
                filter_params = {}
                if must_conditions:
                    filter_params['must'] = must_conditions
                if must_not_conditions:
                    filter_params['must_not'] = must_not_conditions
                
                return Filter(**filter_params)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error building combined filter: {e}")
            return None

    def _build_filter_conditions(self, filters: Dict[str, Any]) -> List[FieldCondition]:
        """
        Build a list of FieldCondition objects from a filters dictionary.
        
        Args:
            filters: Dictionary containing filter conditions
        
        Returns:
            List of FieldCondition objects
        """
        conditions = []
        
        try:
            # Handle author filter
            if 'author' in filters and filters['author']:
                conditions.append(
                    FieldCondition(
                        key="author",
                        match=MatchValue(value=filters['author'])
                    )
                )
            
            # Handle tags filter (array field)
            if 'tags' in filters and filters['tags']:
                tags = filters['tags'] if isinstance(filters['tags'], list) else [filters['tags']]
                conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchAny(any=tags)
                    )
                )
            
            # Handle publication_date filter
            if 'publication_date' in filters and filters['publication_date']:
                pub_date = filters['publication_date']
                
                # Handle new date range format with gte/lt for DATETIME index
                if isinstance(pub_date, dict) and ('gte' in pub_date or 'lt' in pub_date):
                    # Use datetime range for efficient filtering with DATETIME index
                    range_conditions = {}
                    if 'gte' in pub_date:
                        range_conditions['gte'] = pub_date['gte']
                    if 'lt' in pub_date:
                        range_conditions['lt'] = pub_date['lt']
                    
                    conditions.append(
                        FieldCondition(
                            key="publication_date",
                            range=DatetimeRange(**range_conditions)
                        )
                    )
                    logger.debug(f"Applied publication_date range filter: {range_conditions}")
                
                # Handle legacy string format for backward compatibility
                elif isinstance(pub_date, str):
                    if len(pub_date) == 4:  # Year only (e.g., "2023")
                        # Convert to range for better performance with DATETIME index
                        conditions.append(
                            FieldCondition(
                                key="publication_date",
                                range=DatetimeRange(
                                    gte=f"{pub_date}-01-01",
                                    lt=f"{int(pub_date)+1}-01-01"
                                )
                            )
                        )
                        logger.debug(f"Converted year '{pub_date}' to date range")
                    else:  # Exact date match - keep as string match for compatibility
                        conditions.append(
                            FieldCondition(
                                key="publication_date",
                                match=MatchValue(value=pub_date)
                            )
                        )
            
            # Handle title filter (if provided)
            if 'title' in filters and filters['title']:
                conditions.append(
                    FieldCondition(
                        key="title",
                        match=MatchValue(value=filters['title'])
                    )
                )
            
            return conditions
                
        except Exception as e:
            logger.error(f"Error building filter conditions: {e}")
            return []

