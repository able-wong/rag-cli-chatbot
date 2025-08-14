import logging
import requests
import warnings
import os
import sys
import contextlib
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Suppress sentence-transformers and transformers warnings and progress bars
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only show errors
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # Suppress advisory warnings

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

logger = logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(self, config: Dict[str, Any], sparse_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.provider = config.get('provider', 'ollama')
        self.client = None
        
        # Initialize sparse embedding if configuration is provided
        self.sparse_provider = None
        if sparse_config:
            self._initialize_sparse_provider(sparse_config)
        
        self._initialize_client()
    
    def _initialize_sparse_provider(self, sparse_config: Dict[str, Any]):
        """Initialize the sparse embedding provider if configured."""
        try:
            from config import SparseEmbeddingConfig, SpladeConfig
            from sparse_embedding_providers import create_sparse_embedding_provider
            
            # Convert dict config to dataclass
            splade_config = None
            if splade_dict := sparse_config.get('splade'):
                splade_config = SpladeConfig(**splade_dict)
            
            config_obj = SparseEmbeddingConfig(
                provider=sparse_config.get('provider', 'splade'),
                splade=splade_config
            )
            
            self.sparse_provider = create_sparse_embedding_provider(config_obj)
            logger.info(f"Initialized sparse embedding provider: {config_obj.provider}")
            
        except ImportError as e:
            logger.warning(f"Sparse embedding dependencies not available: {e}")
            self.sparse_provider = None
        except Exception as e:
            logger.error(f"Failed to initialize sparse embedding provider: {e}")
            self.sparse_provider = None
    
    def _initialize_client(self):
        """Initialize the embedding client based on provider."""
        try:
            if self.provider == 'ollama':
                self.base_url = self.config.get('base_url', 'http://localhost:11434')
                self.model = self.config.get('model', 'nomic-embed-text')
                self.timeout = self.config.get('timeout', 60)
                logger.info(f"Initialized Ollama embedding client with model: {self.model}")
                
            elif self.provider == 'gemini':
                gemini_config = self.config.get('gemini', {})
                api_key = gemini_config.get('api_key')
                if not api_key:
                    raise ValueError("Gemini API key is required")
                
                genai.configure(api_key=api_key)
                self.model = gemini_config.get('model', 'text-embedding-004')
                logger.info(f"Initialized Gemini embedding client with model: {self.model}")
                
            elif self.provider == 'sentence_transformers':
                st_config = self.config.get('sentence_transformers', {})
                model_name = st_config.get('model', 'all-MiniLM-L6-v2')
                device = st_config.get('device', 'cpu')
                
                # Initialize SentenceTransformer with suppressed output
                with suppress_stdout_stderr():
                    self.client = SentenceTransformer(model_name, device=device)
                logger.info(f"Initialized SentenceTransformers client with model: {model_name} on {device}")
                
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding client: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        try:
            if self.provider == 'ollama':
                return self._get_ollama_embedding(text)
            elif self.provider == 'gemini':
                return self._get_gemini_embedding(text)
            elif self.provider == 'sentence_transformers':
                return self._get_sentence_transformers_embedding(text)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result['embedding']
    
    def _get_gemini_embedding(self, text: str) -> List[float]:
        """Get embedding from Gemini."""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    def _get_sentence_transformers_embedding(self, text: str) -> List[float]:
        """Get embedding from SentenceTransformers."""
        # Suppress progress bars and warnings during encoding
        with suppress_stdout_stderr():
            embedding = self.client.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.get_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            # Return default dimensions based on known models
            if self.provider == 'ollama':
                if 'nomic-embed-text' in self.model:
                    return 768
                elif 'mxbai-embed-large' in self.model:
                    return 1024
            elif self.provider == 'gemini':
                return 768
            elif self.provider == 'sentence_transformers':
                if 'all-MiniLM-L6-v2' in self.config.get('sentence_transformers', {}).get('model', ''):
                    return 384
                elif 'all-mpnet-base-v2' in self.config.get('sentence_transformers', {}).get('model', ''):
                    return 768
            
            # Default fallback
            return 768
    
    def get_sparse_embedding(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate sparse embedding for the given text."""
        if not self.sparse_provider:
            return None
        
        try:
            return self.sparse_provider.generate_sparse_embedding(text)
        except Exception as e:
            logger.error(f"Failed to generate sparse embedding: {e}")
            return None
    
    def get_sparse_embeddings(self, texts: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Generate sparse embeddings for multiple texts."""
        if not self.sparse_provider:
            return None
        
        try:
            return self.sparse_provider.generate_sparse_embeddings(texts)
        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            return None
    
    def has_sparse_embedding(self) -> bool:
        """Check if sparse embedding is available."""
        return self.sparse_provider is not None
    
    def test_sparse_connection(self) -> bool:
        """Test if sparse embedding provider is working."""
        if not self.sparse_provider:
            return False
        
        try:
            return self.sparse_provider.test_connection()
        except Exception as e:
            logger.error(f"Sparse embedding connection test failed: {e}")
            return False
    
    def get_sparse_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the sparse embedding provider."""
        if not self.sparse_provider:
            return None
        
        try:
            return self.sparse_provider.get_info()
        except Exception as e:
            logger.error(f"Failed to get sparse embedding info: {e}")
            return None
