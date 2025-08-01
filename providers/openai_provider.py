"""OpenAI provider implementation - unified API wrapper"""
import logging
from typing import Dict, Any, Optional

from config import OPENAI_MAX_TOKENS

logger = logging.getLogger("providers.openai")

class OpenAIProvider:
    """Unified OpenAI provider supporting all model types and capabilities"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "openai"
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning model (o-series)"""
        return model.startswith('o') and any(x in model for x in ['o4', 'o3', 'o1'])
    
    def _is_gpt_model(self, model: str) -> bool:
        """Check if model is a GPT model (gpt-series)"""
        return model.startswith('gpt-')
    
    def _is_deep_research_model(self, model: str) -> bool:
        """Check if model is a deep research variant"""
        return 'deep-research' in model
    
    def _build_input_messages(self, model: str, system_prompt: str, user_prompt: str) -> list:
        """Build the input messages list based on model type"""
        # Determine role for system message based on model family
        if self._is_reasoning_model(model):
            system_role = "developer"
        else:
            system_role = "system"
        
        # Construct input messages
        messages = [
            {
                "role": system_role,
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt
                    }
                ]
            }
        ]
        
        return messages
    
    def _build_reasoning_payload(self, model: str) -> dict:
        """Build reasoning object based on model type"""
        if self._is_gpt_model(model):
            # GPT models use empty reasoning object
            return {}
        elif self._is_deep_research_model(model):
            # Deep research models don't include effort
            return {
                "summary": "auto"
            }
        elif self._is_reasoning_model(model):
            # Standard reasoning models include effort
            return {
                "effort": "medium",
                "summary": "auto"
            }
        else:
            # Default to empty for unknown models
            return {}
    
    def _build_tools_payload(self, with_search: bool) -> list:
        """Build tools list for web search capability"""
        if with_search:
            return [
                {
                    "type": "web_search_preview",
                    "search_context_size": "medium", 
                    "user_location": None
                }
            ]
        else:
            return []
    
    def _extract_response_text(self, response) -> str:
        """Extract text content from response object"""
        try:
            # Try primary path: response.output[0].content[0].text
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text
            
            # Fallback: direct text attribute
            if hasattr(response, 'text'):
                return response.text
            
            # Last resort: try to find any text content
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
                
            return "No response text found in response object"
            
        except Exception as e:
            logger.warning(f"Error extracting response text: {str(e)}")
            return f"Error extracting response: {str(e)}"
    
    def generate_advanced(self, model: str, system_prompt: str, user_prompt: str, 
                with_search: bool = True, **kwargs) -> str:
        """
        Generate response using OpenAI responses API with dynamic model handling
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'o4-mini', 'o4-mini-deep-research')
            system_prompt: System/developer instructions 
            user_prompt: User query/content
            with_search: Enable web search capability
            **kwargs: Additional parameters (temperature, max_output_tokens, etc.)
            
        Returns:
            Generated response text
        """
        logger.info(f"OpenAI generating with model {model}, search={with_search}")
        
        try:
            # Build dynamic payload components
            input_messages = self._build_input_messages(model, system_prompt, user_prompt)
            reasoning_payload = self._build_reasoning_payload(model)
            tools_payload = self._build_tools_payload(with_search)
            
            # Base payload for responses.create
            payload = {
                "model": model,
                "input": input_messages,
                "text": {
                    "format": {
                        "type": "text"
                    }
                },
                "reasoning": reasoning_payload,
                "tools": tools_payload,
                "store": False
            }
            
            # Add additional parameters for GPT models
            if self._is_gpt_model(model):
                # GPT models can use temperature, max_output_tokens, etc.
                if 'temperature' in kwargs:
                    payload['temperature'] = kwargs['temperature']
                if 'max_output_tokens' in kwargs:
                    payload['max_output_tokens'] = kwargs['max_output_tokens']
                if 'top_p' in kwargs:
                    payload['top_p'] = kwargs['top_p']
            
            # Make API call
            response = self.client.responses.create(**payload)
            
            # Extract and return response text
            response_text = self._extract_response_text(response)
            logger.info(f"OpenAI response: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            error_msg = f"OpenAI generation error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"
    
    def generate_simple(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Simple interface method to match other providers (Gemini/Claude)
        Uses the advanced generate_advanced() method with default parameters
        """
        system_prompt = "You are a Security-Aware Log Schema Architect AI."
        return self.generate_advanced(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            with_search=False,
            temperature=temperature
        )
    
    # Main interface - matches other providers for orchestrator compatibility
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Main generate method - uses simple interface to match other providers (Gemini/Claude)
        Calls generate_simple which in turn uses the advanced generate_advanced method
        """
        return self.generate_simple(model, prompt, temperature)

    def generate_legacy(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Legacy method for backward compatibility with existing code
        Uses chat.completions.create for simple prompt completion
        """
        logger.info(f"OpenAI legacy generating with model {model}")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Security-Aware Log Schema Architect AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=OPENAI_MAX_TOKENS
            )
            
            response_text = response.choices[0].message.content
            logger.info(f"OpenAI legacy response: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            logger.error(f"OpenAI legacy generation error: {str(e)}")
            return f"Error: {str(e)}"


class OpenAIWithSearchProvider:
    """OpenAI provider with web search capability"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "openai_search"
        self._base_provider = None
    
    @property
    def base_provider(self):
        """Lazy initialization of base OpenAI provider"""
        if self._base_provider is None:
            self._base_provider = OpenAIProvider(self.api_key)
        return self._base_provider
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using OpenAI with web search enabled"""
        system_prompt = "You are a Security-Aware Log Schema Architect AI with advanced research capabilities."
        return self.base_provider.generate_advanced(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            with_search=True,
            temperature=temperature
        )

class OpenAIThinkingProvider:
    """OpenAI provider optimized for reasoning models (o-series)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "openai_thinking"
        self._base_provider = None
    
    @property
    def base_provider(self):
        """Lazy initialization of base OpenAI provider"""
        if self._base_provider is None:
            self._base_provider = OpenAIProvider(self.api_key)
        return self._base_provider
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using OpenAI reasoning models with enhanced prompting"""
        system_prompt = "You are an expert Security-Aware Log Schema Architect AI. Think step by step and reason through the log analysis carefully."
        return self.base_provider.generate_advanced(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            with_search=False,
            temperature=temperature
        )

# Example usage demonstrations:
# if __name__ == "__main__":
#     # Initialize provider
#     provider = OpenAIProvider(api_key="your-api-key-here")
    
#     # Example 1: o4-mini with search enabled
#     system_prompt = "You are a Senior Cyber Threat Analyst AI with advanced research capabilities."
#     user_prompt = "Analyze the latest cybersecurity threats in 2024."
    
#     response1 = provider.generate(
#         model="o4-mini",
#         system_prompt=system_prompt,
#         user_prompt=user_prompt,
#         with_search=True
#     )
#     print("o4-mini with search:", response1[:100] + "...")
    
#     # Example 2: gpt-4o with search disabled
#     response2 = provider.generate(
#         model="gpt-4o", 
#         system_prompt="You are a helpful security analyst.",
#         user_prompt="Explain common attack vectors.",
#         with_search=False,
#         temperature=0.5,
#         max_output_tokens=2000
#     )
#     print("gpt-4o without search:", response2[:100] + "...")
    
#     # Example 3: o4-mini-deep-research with search enabled
#     response3 = provider.generate(
#         model="o4-mini-deep-research",
#         system_prompt="You are an expert threat intelligence researcher.",
#         user_prompt="Research emerging APT groups and their tactics.",
#         with_search=True
#     )
#     print("o4-mini-deep-research with search:", response3[:100] + "...")