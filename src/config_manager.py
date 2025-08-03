import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file and environment variables."""
        try:
            # Load base configuration from YAML
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
            
            # Override with environment variables where applicable
            self._override_with_env_vars(config)
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def _override_with_env_vars(self, config: Dict[str, Any]) -> None:
        """Override configuration values with environment variables."""
        
        # LLM API keys
        if 'llm' in config:
            if config['llm'].get('provider') == 'gemini':
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    if 'gemini' not in config['llm']:
                        config['llm']['gemini'] = {}
                    config['llm']['gemini']['api_key'] = api_key
        
        # Embedding API keys
        if 'embedding' in config:
            if config['embedding'].get('provider') == 'gemini':
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    if 'gemini' not in config['embedding']:
                        config['embedding']['gemini'] = {}
                    config['embedding']['gemini']['api_key'] = api_key
        
        # Qdrant API key
        if 'vector_db' in config:
            api_key = os.getenv('QDRANT_API_KEY')
            if api_key:
                config['vector_db']['api_key'] = api_key
        
        # Query Rewriter configuration
        if 'query_rewriter' not in config:
            config['query_rewriter'] = {}
        
        # RAG Strategy override
        rag_strategy = os.getenv('RAG_STRATEGY')
        if rag_strategy and rag_strategy.lower() in ['hyde', 'rewrite']:
            config['query_rewriter']['retrieval_strategy'] = rag_strategy.lower()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'llm.provider')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.get('llm', {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self.get('embedding', {})
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self.get('vector_db', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG-specific configuration."""
        return self.get('rag', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def get_documents_config(self) -> Dict[str, Any]:
        """Get documents configuration."""
        return self.get('documents', {})