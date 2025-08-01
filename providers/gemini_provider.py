# providers/gemini_provider.py
"""Gemini provider implementations - simple API wrappers"""
import logging
from typing import Dict, Any

from config import GEMINI_MAX_TOKENS

logger = logging.getLogger("providers.gemini")

class GeminiProvider:
    """Basic Gemini provider using google.genai API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "gemini"
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using Gemini API"""
        logger.info(f"Gemini generating with model {model}")
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)
            
            msg_text = types.Part.from_text(text=prompt)
            si_text = types.Part.from_text(text="You are a Security-Aware Log Schema Architect AI.")
            
            contents = [
                types.Content(role="user", parts=[msg_text])
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=1,
                seed=0,
                max_output_tokens=GEMINI_MAX_TOKENS,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                ],
                system_instruction=[si_text],
            )
            
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text or ""
            elif hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                response_text += part.text or ""
            
            # Debug logging
            if not response_text:
                logger.warning(f"Gemini returned empty response. Full response: {response}")
                if hasattr(response, 'candidates'):
                    logger.warning(f"Candidates: {response.candidates}")
            
            logger.info(f"Gemini response: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            return f"Error: {str(e)}"

class GeminiWithSearchProvider:
    """Gemini provider with Google Search capability"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "gemini_search"
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using Gemini API with Google Search"""
        logger.info(f"Gemini+Search generating with model {model}")
        
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)
            
            msg_text = types.Part.from_text(text=prompt)
            si_text = types.Part.from_text(text="You are a Security-Aware Log Schema Architect AI.")
            
            contents = [
                types.Content(role="user", parts=[msg_text])
            ]
            
            # Add Google Search tool
            tools = [types.Tool(google_search=types.GoogleSearch())]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=1,
                seed=0,
                max_output_tokens=GEMINI_MAX_TOKENS,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                ],
                tools=tools,
                system_instruction=[si_text],
            )
            
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            response_text = ""
            if hasattr(response, 'text'):
                response_text = response.text or ""
            elif hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                response_text += part.text or ""
            
            # Debug logging
            if not response_text:
                logger.warning(f"Gemini+Search returned empty response. Full response: {response}")
                if hasattr(response, 'candidates'):
                    logger.warning(f"Candidates: {response.candidates}")
            
            logger.info(f"Gemini+Search response: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            logger.error(f"Gemini+Search generation error: {str(e)}")
            # Fallback to regular generation
            return GeminiProvider(self.api_key).generate(model, prompt, temperature)