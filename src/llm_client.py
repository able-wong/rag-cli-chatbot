import logging
import requests
import json
import re
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
    
    def get_json_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get JSON response from LLM, handling markdown wrappers automatically.
        
        This method:
        1. Gets raw text response from LLM
        2. Strips markdown code block wrappers (```json ... ```)
        3. Parses JSON and returns dictionary
        4. Provides clear error messages for debugging
        
        Args:
            messages: List of message dictionaries for the LLM
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Parsed JSON response as dictionary
            
        Raises:
            ValueError: If response is not valid JSON
            Exception: If LLM call fails
        """
        try:
            # Get raw text response from LLM
            raw_response = self.get_llm_response(messages, temperature, max_tokens)
            
            if not raw_response or not raw_response.strip():
                raise ValueError("LLM returned empty response")
            
            # Strip markdown code block wrappers if present
            cleaned_response = self._strip_markdown_json_wrapper(raw_response.strip())
            
            # Parse JSON
            try:
                parsed_json = json.loads(cleaned_response)
                logger.debug(f"Successfully parsed JSON response: {type(parsed_json)}")
                return parsed_json
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response. Raw: '{raw_response[:200]}...', Cleaned: '{cleaned_response[:200]}...', Error: {e}")
                raise ValueError(f"Invalid JSON response from LLM: {e}")
                
        except Exception as e:
            logger.error(f"Failed to get JSON response from LLM: {e}")
            raise
    
    def _strip_markdown_json_wrapper(self, text: str) -> str:
        """
        Strip markdown JSON code block wrappers from text.
        
        Handles these patterns:
        - ```json\n{...}\n```
        - ```\n{...}\n```
        - ````json\n{...}\n````
        - Raw JSON without wrappers
        
        Args:
            text: Raw text that may contain markdown wrappers
            
        Returns:
            Cleaned text with wrappers removed
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Pattern to match markdown code blocks with optional json specifier
        # Matches: ```json or ``` or ````json or ````
        # Captures the content between the backticks
        patterns = [
            r'```(?:json)?\s*\n(.*?)\n```',  # Standard triple backticks
            r'````(?:json)?\s*\n(.*?)\n````',  # Quad backticks (rare but possible)
            r'```(?:json)?\s*(.*?)```',  # Single line variant
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Stripped markdown wrapper, extracted: '{extracted[:100]}...'")
                return extracted
        
        # If no markdown wrapper found, return original text
        logger.debug("No markdown wrapper detected, using raw text")
        return text