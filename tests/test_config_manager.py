import pytest
import tempfile
import os
import yaml
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config_manager import ConfigManager

def test_config_manager_loads_yaml():
    """Test that ConfigManager correctly loads YAML configuration."""
    config_data = {
        'llm': {
            'provider': 'ollama',
            'model': 'llama3.2'
        },
        'embedding': {
            'provider': 'ollama',
            'model': 'nomic-embed-text'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigManager(temp_path)
        assert config_manager.get('llm.provider') == 'ollama'
        assert config_manager.get('llm.model') == 'llama3.2'
        assert config_manager.get('embedding.provider') == 'ollama'
    finally:
        os.unlink(temp_path)

def test_config_manager_env_override():
    """Test that environment variables override config values."""
    config_data = {
        'llm': {
            'provider': 'gemini',
            'gemini': {}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    # Set environment variable
    os.environ['GEMINI_API_KEY'] = 'test-api-key'
    
    try:
        config_manager = ConfigManager(temp_path)
        assert config_manager.get('llm.gemini.api_key') == 'test-api-key'
    finally:
        os.unlink(temp_path)
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']

def test_config_manager_get_with_default():
    """Test that get() returns default value for missing keys."""
    config_data = {'existing': 'value'}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config_manager = ConfigManager(temp_path)
        assert config_manager.get('existing') == 'value'
        assert config_manager.get('missing', 'default') == 'default'
        assert config_manager.get('missing') is None
    finally:
        os.unlink(temp_path)

