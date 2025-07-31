import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

class QdrantDB:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get('collection_name', 'knowledge_base')
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.client = None
        self._initialize_client()
    
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
            # Try to get collections to test connection
            collections = self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
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
        score_threshold: Optional[float] = None
    ) -> List[ScoredPoint]:
        """Search for similar vectors in the collection."""
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