import logging
import requests
from typing import List, Dict, Any, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'ollama')
        self.conversation_history = []
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        try:
            if self.provider == 'ollama':
                self.base_url = self.config.get('base_url', 'http://localhost:11434')
                self.model = self.config.get('model', 'llama3.2')
                self.timeout = self.config.get('timeout', 120)
                logger.info(f"Initialized Ollama LLM client with model: {self.model}")
                
            elif self.provider == 'gemini':
                gemini_config = self.config.get('gemini', {})
                api_key = gemini_config.get('api_key')
                if not api_key:
                    raise ValueError("Gemini API key is required")
                
                genai.configure(api_key=api_key)
                self.model = gemini_config.get('model', 'gemini-1.5-flash')
                logger.info(f"Initialized Gemini LLM client with model: {self.model}")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def get_llm_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response from the LLM."""
        try:
            if self.provider == 'ollama':
                return self._get_ollama_response(messages, temperature, max_tokens)
            elif self.provider == 'gemini':
                return self._get_gemini_response(messages, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise
    
    def _get_ollama_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Get response from Ollama."""
        url = f"{self.base_url}/api/chat"
        
        # Use config defaults if not provided
        temp = temperature if temperature is not None else self.config.get('temperature', 0.7)
        max_tok = max_tokens if max_tokens is not None else self.config.get('max_tokens', 2048)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
                "top_p": self.config.get('top_p', 0.9),
                "top_k": self.config.get('top_k', 40)
            }
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result['message']['content']
    
    def _get_gemini_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Get response from Gemini."""
        # Convert messages to Gemini format
        gemini_messages = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # Map roles
            if role == 'system':
                # Gemini doesn't have system role, prepend to first user message
                if gemini_messages and gemini_messages[-1]['role'] == 'user':
                    gemini_messages[-1]['parts'][0]['text'] = f"{content}\n\n{gemini_messages[-1]['parts'][0]['text']}"
                else:
                    gemini_messages.append({
                        'role': 'user',
                        'parts': [{'text': content}]
                    })
            elif role == 'user':
                gemini_messages.append({
                    'role': 'user',
                    'parts': [{'text': content}]
                })
            elif role == 'assistant':
                gemini_messages.append({
                    'role': 'model',
                    'parts': [{'text': content}]
                })
        
        # Use config defaults if not provided
        temp = temperature if temperature is not None else self.config.get('temperature', 0.7)
        max_tok = max_tokens if max_tokens is not None else self.config.get('max_tokens', 2048)
        
        model = genai.GenerativeModel(self.model)
        
        generation_config = genai.GenerationConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            top_p=self.config.get('top_p', 0.9),
            top_k=self.config.get('top_k', 40)
        )
        
        # Start chat with history (excluding the last message which will be sent)
        chat = model.start_chat(history=gemini_messages[:-1])
        
        # Send the last message
        response = chat.send_message(
            gemini_messages[-1]['parts'][0]['text'],
            generation_config=generation_config
        )
        
        return response.text
    
    def test_connection(self) -> bool:
        """Test connection to the LLM service."""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.get_llm_response(test_messages)
            
            if response and len(response.strip()) > 0:
                logger.info("Successfully connected to LLM service")
                return True
            else:
                logger.error("LLM returned empty response")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to LLM service: {e}")
            return False