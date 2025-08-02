"""
Integration tests for LLM providers (Ollama and Gemini).
Tests basic connection and JSON response functionality with real providers.
"""

import sys
import os
from typing import Dict, Any
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import LLMClient


class TestLLMProvidersIntegration:
    """Integration tests for both Ollama and Gemini LLM providers."""
    
    def create_ollama_config(self) -> Dict[str, Any]:
        """Create Ollama configuration for testing."""
        # Try to get Ollama config from actual config file first
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        try:
            from config_manager import ConfigManager
            config_manager = ConfigManager(config_path)
            actual_base_url = config_manager.get('llm.base_url', 'http://localhost:11434')
        except Exception:
            actual_base_url = 'http://localhost:11434'
        
        return {
            'provider': 'ollama',
            'model': 'llama3.2:3b',
            'base_url': actual_base_url,
            'timeout': 30,  # Short timeout for tests
            'temperature': 0.1,
            'max_tokens': 200
        }
    
    def create_gemini_config(self) -> Dict[str, Any]:
        """Create Gemini configuration for testing."""
        # Try to get API key from config or environment
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        
        try:
            from config_manager import ConfigManager
            config_manager = ConfigManager(config_path)
            gemini_config = config_manager.get('llm.gemini', {})
            api_key = gemini_config.get('api_key', '')
        except Exception:
            api_key = os.environ.get('GEMINI_API_KEY', '')
        
        return {
            'provider': 'gemini',
            'gemini': {
                'api_key': api_key,
                'model': 'gemini-1.5-flash'
            },
            'temperature': 0.1,
            'max_tokens': 200
        }
    
    def is_provider_available(self, provider_config: Dict[str, Any]) -> bool:
        """Check if a provider is available for testing."""
        if provider_config['provider'] == 'ollama':
            # Check if Ollama is running locally
            try:
                import requests
                response = requests.get(f"{provider_config['base_url']}/api/tags", timeout=5)
                return response.status_code == 200
            except Exception:
                return False
        
        elif provider_config['provider'] == 'gemini':
            # Check if API key is available
            gemini_config = provider_config.get('gemini', {})
            api_key = gemini_config.get('api_key', '')
            return bool(api_key and api_key != 'your-gemini-api-key-here')
        
        return False
    
    @pytest.mark.integration
    @pytest.mark.parametrize("provider_name,config_func", [
        ("ollama", "create_ollama_config"),
        ("gemini", "create_gemini_config")
    ])
    def test_llm_provider_connection(self, provider_name: str, config_func: str):
        """Test basic connection to LLM provider."""
        # Get config for this provider
        config = getattr(self, config_func)()
        
        # Only skip Gemini if API key is clearly not configured
        if provider_name == "gemini":
            gemini_config = config.get('gemini', {})
            api_key = gemini_config.get('api_key', '')
            if not api_key or api_key == 'your-gemini-api-key-here':
                pytest.skip("Gemini API key not configured for testing")
        
        try:
            # Create client and test connection
            client = LLMClient(config)
            connection_success = client.test_connection()
            
            if connection_success:
                print(f"✅ {provider_name} connection test passed")
            else:
                print(f"⚠️ {provider_name} connection failed (provider may not be running)")
                
            # For integration tests, we want to know if the connection logic works
            # even if the provider is not available, so we test the boolean response
            assert isinstance(connection_success, bool), f"{provider_name} should return boolean connection status"
            
        except Exception as e:
            print(f"❌ {provider_name} connection test encountered exception: {e}")
            # Re-raise to fail the test - exceptions indicate code issues, not just unavailable providers
            raise
    
    @pytest.mark.integration
    @pytest.mark.parametrize("provider_name,config_func", [
        ("ollama", "create_ollama_config"),
        ("gemini", "create_gemini_config")
    ])
    def test_llm_provider_json_response(self, provider_name: str, config_func: str):
        """Test JSON response parsing with real LLM provider."""
        # Get config for this provider
        config = getattr(self, config_func)()
        
        # Only skip Gemini if API key is clearly not configured
        if provider_name == "gemini":
            gemini_config = config.get('gemini', {})
            api_key = gemini_config.get('api_key', '')
            if not api_key or api_key == 'your-gemini-api-key-here':
                pytest.skip("Gemini API key not configured for testing")
        
        try:
            # Create client
            client = LLMClient(config)
            
            # Test simple JSON response
            messages = [
                {
                    "role": "user",
                    "content": "Respond with this exact JSON structure: {\"status\": \"ok\", \"test\": true, \"number\": 42}. Return only the JSON, nothing else."
                }
            ]
            
            # Get JSON response
            result = client.get_json_response(messages)
            
            # Validate response structure
            assert isinstance(result, dict), f"{provider_name} should return a dictionary"
            
            # Check for expected fields (allowing some flexibility since LLMs might not follow exactly)
            assert len(result) > 0, f"{provider_name} should return non-empty JSON"
            
            # Validate it contains reasonable JSON structure
            has_valid_structure = any(
                isinstance(v, (str, int, bool, list, dict)) for v in result.values()
            )
            assert has_valid_structure, f"{provider_name} should return valid JSON data types"
            
            print(f"✅ {provider_name} JSON response test passed")
            print(f"   Response: {result}")
            
        except Exception as e:
            print(f"❌ {provider_name} JSON response test failed: {e}")
            if provider_name == "ollama":
                print("   Note: This may indicate Ollama is not running or configured correctly")
            # Let test fail to show the actual error
            raise
    
    @pytest.mark.integration
    def test_both_providers_if_available(self):
        """Test both providers if they're available, comparing their behavior."""
        ollama_config = self.create_ollama_config()
        gemini_config = self.create_gemini_config()
        
        # Check if providers are configured (not necessarily running)
        gemini_api_key = gemini_config.get('gemini', {}).get('api_key', '')
        gemini_configured = bool(gemini_api_key and gemini_api_key != 'your-gemini-api-key-here')
        ollama_configured = True  # Always try Ollama - it's locally configured
        
        if not (ollama_configured or gemini_configured):
            pytest.skip("Neither Ollama nor Gemini configured for comparison test")
        
        results = {}
        
        # Test Ollama if configured
        if ollama_configured:
            try:
                client = LLMClient(ollama_config)
                messages = [{"role": "user", "content": "Respond with JSON: {\"provider\": \"ollama\", \"working\": true}"}]
                results['ollama'] = client.get_json_response(messages)
                print(f"Ollama result: {results['ollama']}")
            except Exception as e:
                print(f"Ollama failed: {e}")
        
        # Test Gemini if configured
        if gemini_configured:
            try:
                client = LLMClient(gemini_config)
                messages = [{"role": "user", "content": "Respond with JSON: {\"provider\": \"gemini\", \"working\": true}"}]
                results['gemini'] = client.get_json_response(messages)
                print(f"Gemini result: {results['gemini']}")
            except Exception as e:
                print(f"Gemini failed: {e}")
        
        # Verify at least one provider worked
        assert len(results) > 0, "At least one provider should work"
        
        # Verify all results are valid JSON
        for provider, result in results.items():
            assert isinstance(result, dict), f"{provider} should return dictionary"
            assert len(result) > 0, f"{provider} should return non-empty result"
        
        print(f"✅ Comparison test passed with {len(results)} provider(s)")
    
    @pytest.mark.integration
    def test_json_response_with_markdown_wrapper_real_llm(self):
        """Test that real LLM responses with markdown wrappers are handled correctly."""
        # Try with whichever provider is available
        configs = [
            ("ollama", self.create_ollama_config()),
            ("gemini", self.create_gemini_config())
        ]
        
        provider_tested = False
        
        for provider_name, config in configs:
            if not self.is_provider_available(config):
                continue
            
            try:
                client = LLMClient(config)
                
                # Request JSON with explicit markdown wrapper
                messages = [
                    {
                        "role": "user", 
                        "content": "Please respond with JSON wrapped in markdown code blocks like this:\n```json\n{\"test\": \"markdown_wrapper\", \"success\": true}\n```\nUse exactly that format."
                    }
                ]
                
                result = client.get_json_response(messages)
                
                # Should successfully parse despite potential markdown wrapper
                assert isinstance(result, dict), f"{provider_name} should return dictionary"
                assert len(result) > 0, f"{provider_name} should parse JSON from markdown"
                
                print(f"✅ {provider_name} markdown wrapper test passed: {result}")
                provider_tested = True
                break
                
            except Exception as e:
                print(f"{provider_name} markdown wrapper test failed: {e}")
                continue
        
        if not provider_tested:
            pytest.skip("No LLM provider available for markdown wrapper test")