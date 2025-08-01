# providers/claude_provider.py
"""Claude provider implementations - simple API wrappers"""
import logging
import time
from typing import Dict, Any

from config import ANTHROPIC_MAX_TOKENS

logger = logging.getLogger("providers.claude")

class ClaudeProvider:
    """Basic Claude provider using Anthropic API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "claude"
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using Claude API with improved error handling"""
        logger.info(f"Claude generating with model {model}")
        
        # Parse system message and user content from prompt
        system_msg, user_content = self._parse_prompt(prompt)
        
        # Add delay to avoid rate limiting
        time.sleep(2)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)
                
                message = client.messages.create(
                    model=model,
                    max_tokens=ANTHROPIC_MAX_TOKENS,
                    temperature=temperature,
                    system=system_msg,
                    messages=[
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ]
                )
                
                # Extract text from response
                response_text = ""
                if hasattr(message, 'content'):
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
                        elif isinstance(content_item, dict) and 'text' in content_item:
                            response_text += content_item['text']
                
                logger.info(f"Claude response: {len(response_text)} chars")
                return response_text
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Claude attempt {attempt + 1} failed: {error_str}")
                
                # Handle specific error types
                if "529" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                        logger.info(f"API overloaded, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                elif "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30  # Longer wait for rate limits: 30s, 60s, 90s
                        logger.info(f"Rate limit exceeded, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                elif "400" in error_str and "max_tokens" in error_str:
                    logger.error(f"Token limit error - cannot retry: {error_str}")
                    return f"Error: {error_str}"
                
                if attempt == max_retries - 1:
                    logger.error(f"Claude generation failed after {max_retries} attempts: {error_str}")
                    return f"Error: {error_str}"
    
    def _parse_prompt(self, prompt: str) -> tuple[str, str]:
        """Parse system message and user content from formatted prompt"""
        lines = prompt.strip().split('\n')
        
        # Look for system message at the start ("You are...")
        system_msg = "You are a Security-Aware Log Schema Architect AI."
        user_content = prompt
        
        if lines and lines[0].strip().startswith("You are"):
            # Find the end of the system message (usually ends with a separator like -----)
            system_lines = []
            user_lines = []
            in_system = True
            
            for line in lines:
                if in_system and ('-----' in line or '###' in line or '##' in line):
                    in_system = False
                    user_lines.append(line)
                elif in_system:
                    system_lines.append(line)
                else:
                    user_lines.append(line)
            
            if system_lines:
                system_msg = '\n'.join(system_lines).strip()
                user_content = '\n'.join(user_lines).strip()
        
        return system_msg, user_content

class ClaudeWithSearchProvider:
    """Claude provider with web search using beta API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "claude_search"
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate response using Claude beta API with web search"""
        logger.info(f"Claude+Search generating with model {model}")
        
        # Parse system message and user content from prompt
        system_msg, user_content = self._parse_prompt(prompt)
        
        # Add delay to avoid rate limiting
        time.sleep(2)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)
                
                # Try beta API with web search first
                try:
                    message = client.beta.messages.create(
                        model=model,
                        max_tokens=ANTHROPIC_MAX_TOKENS,
                        temperature=temperature,
                        system=system_msg,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": user_content
                                    }
                                ]
                            }
                        ],
                        tools=[
                            {
                                "name": "web_search",
                                "type": "web_search_20250305"
                            }
                        ],
                        betas=["web-search-2025-03-05"]
                    )
                except Exception as search_error:
                    logger.warning(f"Web search failed, using regular API: {search_error}")
                    # Fallback to regular API with same retry logic
                    return ClaudeProvider(self.api_key).generate(model, prompt, temperature)
                
                # Extract text from response
                response_text = ""
                if hasattr(message, 'content'):
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
                        elif isinstance(content_item, dict) and 'text' in content_item:
                            response_text += content_item['text']
                
                logger.info(f"Claude+Search response: {len(response_text)} chars")
                return response_text
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Claude+Search attempt {attempt + 1} failed: {error_str}")
                
                # Handle specific error types
                if "529" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # Exponential backoff: 10s, 20s, 30s
                        logger.info(f"API overloaded, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                elif "429" in error_str or "rate_limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 30  # Longer wait for rate limits: 30s, 60s, 90s
                        logger.info(f"Rate limit exceeded, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                elif "400" in error_str and "max_tokens" in error_str:
                    logger.error(f"Token limit error - cannot retry: {error_str}")
                    return f"Error: {error_str}"
                
                if attempt == max_retries - 1:
                    logger.error(f"Claude+Search generation failed after {max_retries} attempts: {error_str}")
                    # Final fallback to regular generation
                    return ClaudeProvider(self.api_key).generate(model, prompt, temperature)
    
    def _parse_prompt(self, prompt: str) -> tuple[str, str]:
        """Parse system message and user content from formatted prompt"""
        lines = prompt.strip().split('\n')
        
        # Look for system message at the start ("You are...")
        system_msg = "You are a Security-Aware Log Schema Architect AI."
        user_content = prompt
        
        if lines and lines[0].strip().startswith("You are"):
            # Find the end of the system message (usually ends with a separator like -----)
            system_lines = []
            user_lines = []
            in_system = True
            
            for line in lines:
                if in_system and ('-----' in line or '###' in line or '##' in line):
                    in_system = False
                    user_lines.append(line)
                elif in_system:
                    system_lines.append(line)
                else:
                    user_lines.append(line)
            
            if system_lines:
                system_msg = '\n'.join(system_lines).strip()
                user_content = '\n'.join(user_lines).strip()
        
        return system_msg, user_content