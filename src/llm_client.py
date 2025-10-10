import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
import requests

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
        max_tokens: Optional[int] = None,
        tools: Optional[List[Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate response from the LLM.

        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            tools: Optional list of tool schemas (provider-specific format)

        Returns:
            str: Normal text response (when no tools called)
            Dict: {"text": str, "tool_calls": list} when tools called
        """
        try:
            if self.provider == 'ollama':
                return self._get_ollama_response(messages, temperature, max_tokens, tools)
            elif self.provider == 'gemini':
                return self._get_gemini_response(messages, temperature, max_tokens, tools)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise

    def _get_ollama_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Any]] = None
    ) -> Union[str, Dict[str, Any]]:
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

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        message = result.get('message', {})

        # Check for tool calls
        tool_calls = message.get('tool_calls')
        if tool_calls:
            logger.debug(f"Ollama requested {len(tool_calls)} tool calls")
            return {
                "text": message.get('content', ''),
                "tool_calls": tool_calls
            }

        # Normal text response
        return message.get('content', '')

    def _get_gemini_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Any]] = None
    ) -> Union[str, Dict[str, Any]]:
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

        # DEBUG: Log what we're sending to Gemini
        logger.debug("=" * 80)
        logger.debug("GEMINI REQUEST DEBUG")
        logger.debug("=" * 80)
        logger.debug(f"Messages being sent ({len(gemini_messages)} messages):")
        for i, msg in enumerate(gemini_messages):
            logger.debug(f"  Message {i+1}: role={msg['role']}")
            logger.debug(f"    content: {msg['parts'][0]['text'][:200]}...")
        logger.debug(f"\nTools provided: {len(tools) if tools else 0}")
        if tools:
            logger.debug(f"Tool names: {[t.name for t in tools]}")
            logger.debug("\nDetailed Tool Schemas:")
            for tool in tools:
                logger.debug(f"\n  Tool: {tool.name}")
                logger.debug(f"  Description: {tool.description}")
                # Convert parameters schema to dict for display
                logger.debug(f"  Parameters schema: {tool.parameters}")
        logger.debug("=" * 80)

        # Create model with tools if provided
        if tools:
            model = genai.GenerativeModel(self.model, tools=tools)
        else:
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

        # Check for function calls
        if response.parts and hasattr(response.parts[0], 'function_call') and response.parts[0].function_call:
            function_calls = []
            for part in response.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    # Convert Gemini format to Ollama-compatible format
                    function_calls.append({
                        "function": {
                            "name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {}
                        }
                    })

            if function_calls:
                logger.debug(f"Gemini requested {len(function_calls)} function calls")
                # Extract text from parts that are not function calls
                text_parts = []
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)

                return {
                    "text": ' '.join(text_parts) if text_parts else '',
                    "tool_calls": function_calls
                }

        # Normal text response
        return response.text

    def test_connection(self) -> bool:
        """Test connection to the LLM service."""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.get_llm_response(test_messages)

            # Handle both string and dict responses
            if isinstance(response, dict):
                response_text = response.get("text", "")
            else:
                response_text = response

            if response_text and len(response_text.strip()) > 0:
                logger.info("Successfully connected to LLM service")
                return True
            else:
                logger.error("LLM returned empty response")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to LLM service: {e}")
            return False

    def test_tool_support(self) -> bool:
        """
        Test if current model supports tools/function calling.

        For Gemini: Always returns True (all models support function calling)
        For Ollama: Sends test request with empty tools array to check for errors

        Returns:
            True if tools supported, False otherwise

        Raises:
            RuntimeError: If tools not supported and required
        """
        if self.provider == 'gemini':
            # All Gemini models support function calling
            logger.debug(f"Gemini model {self.model} supports tools")
            return True

        elif self.provider == 'ollama':
            # Test Ollama model by sending request with empty tools array
            try:
                url = f"{self.base_url}/api/chat"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False,
                    "tools": []  # Empty tools array to test support
                }

                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                # If we get here, tools are supported
                logger.debug(f"Ollama model {self.model} supports tools")
                return True

            except requests.exceptions.HTTPError as e:
                # Check if error is related to tools not being supported
                if e.response.status_code == 400:
                    error_msg = e.response.json().get('error', '')
                    if 'tool' in error_msg.lower() or 'function' in error_msg.lower():
                        logger.warning(
                            f"Ollama model {self.model} does not support tool calling"
                        )
                        return False

                # Re-raise if it's a different error
                logger.error(f"Error testing tool support: {e}")
                raise

            except Exception as e:
                logger.error(f"Failed to test tool support: {e}")
                raise

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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

            # Strip thinking blocks and markdown wrappers if present
            cleaned_response = self._extract_json_from_response(raw_response.strip())

            # Parse JSON
            try:
                parsed_json = json.loads(cleaned_response)
                logger.debug(f"Successfully parsed JSON response: {type(parsed_json)}")
                return parsed_json

            except json.JSONDecodeError as e:
                # Log error without sensitive data (only error type)
                logger.error(f"Failed to parse JSON response: {e}")
                # Log full details only at debug level (not in production logs)
                logger.debug(
                    f"JSON parse details - Raw length: {len(raw_response)}, "
                    f"Cleaned length: {len(cleaned_response)}"
                )
                # Create enhanced error with raw response for debugging
                json_error = ValueError(f"Invalid JSON response from LLM: {e}")
                json_error.raw_response = raw_response
                json_error.cleaned_response = cleaned_response
                raise json_error

        except Exception as e:
            logger.error(f"Failed to get JSON response from LLM: {e}")
            raise

    def _extract_json_from_response(self, text: str) -> str:
        """
        Extract JSON from LLM response, handling thinking blocks and markdown wrappers.

        This method handles:
        1. Thinking models with <think>...</think> blocks
        2. Markdown code block wrappers (```json ... ```)
        3. Raw JSON without wrappers
        4. Multiple extraction strategies for robustness

        Args:
            text: Raw text from LLM that may contain thinking blocks and/or markdown

        Returns:
            Cleaned JSON text ready for parsing
        """
        # First, remove thinking blocks if present
        text_without_thinking = self._strip_thinking_blocks(text)

        # Then try to extract JSON from the remaining content
        json_text = self._strip_markdown_json_wrapper(text_without_thinking)

        # Finally, clean up common JSON formatting issues
        return self._cleanup_json_formatting(json_text)

    def _strip_thinking_blocks(self, text: str) -> str:
        """
        Remove <think>...</think> blocks from text, preserving other content.

        Args:
            text: Text that may contain thinking blocks

        Returns:
            Text with thinking blocks removed
        """
        # Remove thinking blocks (case insensitive, multiline)
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned.strip())

        return cleaned

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
                logger.debug(f"Stripped markdown wrapper, extracted {len(extracted)} chars")
                return extracted

        # If no markdown wrapper found, return original text
        logger.debug("No markdown wrapper detected, using raw text")
        return text

    def _cleanup_json_formatting(self, text: str) -> str:
        """
        Clean up common JSON formatting issues that LLMs often produce.

        This handles:
        1. Trailing commas at the end of JSON objects
        2. Extra whitespace around braces

        Args:
            text: JSON text that may have formatting issues

        Returns:
            Cleaned JSON text
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove trailing commas before closing braces/brackets
        # This handles cases like: {"key": "value",} -> {"key": "value"}
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Remove any trailing commas at the very end
        text = text.rstrip(',').strip()

        logger.debug(f"Cleaned JSON formatting, final length: {len(text)} chars")
        return text
