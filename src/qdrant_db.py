import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, Filter, FieldCondition, MatchValue, MatchAny, DatetimeRange

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
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPoint]:
        """Search for similar vectors in the collection with optional metadata filtering."""
        try:
            if not self.collection_exists():
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return []
            
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": limit
            }
            
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            # Add filters if provided
            if filters:
                qdrant_filter = self._build_qdrant_filter(filters)
                if qdrant_filter:
                    search_params["query_filter"] = qdrant_filter
                    logger.info(f"Applied hybrid search filters: {list(filters.keys())}")
            
            results = self.client.search(**search_params)
            
            logger.info(f"Found {len(results)} results from search")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
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
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """
        Build Qdrant Filter object from filters dictionary.
        
        Args:
            filters: Dictionary containing filter conditions
                    e.g., {"author": "Smith", "tags": ["python"], "publication_date": "2023"}
        
        Returns:
            Qdrant Filter object or None if no valid filters
        """
        if not filters:
            return None
        
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
            
            if conditions:
                return Filter(must=conditions)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error building Qdrant filter: {e}")
            return None