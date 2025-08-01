# few_shot_orchestrator.py
"""
🔄 FEW-SHOT MULTI-LLM PIPELINE SYSTEM
Orchestrator for three-stage schema generation pipeline

This file is part of System 1: Few-Shot Multi-LLM Pipeline
See SYSTEM_ARCHITECTURE.md for complete system documentation
"""
import logging
from typing import Dict, Any, List, Optional

from config import PROMPT_TEMPLATES, MODEL_CONFIG
from preprocessing import ProcessedLogData
from utils import extract_json_from_text
from providers.openai_provider import OpenAIProvider, OpenAIWithSearchProvider, OpenAIThinkingProvider
from providers.gemini_provider import GeminiProvider, GeminiWithSearchProvider
from providers.claude_provider import ClaudeProvider, ClaudeWithSearchProvider

logger = logging.getLogger("few_shot_orchestrator")

class FewShotOrchestrator:
    """Orchestrates the three-stage schema generation pipeline"""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize with API keys"""
        self.api_keys = api_keys
        self.providers = self._create_providers()
    
    def _create_providers(self) -> Dict[str, Any]:
        """Create provider instances based on MODEL_CONFIG"""
        providers = {}
        
        # Get all unique provider names from MODEL_CONFIG
        all_provider_names = set()
        for prompt_config in MODEL_CONFIG.values():
            all_provider_names.update(prompt_config.keys())
        
        # Create providers only for those specified in config
        for provider_name in all_provider_names:
            if provider_name.startswith('openai') and self.api_keys.get('openai'):
                if provider_name == 'openai':
                    providers[provider_name] = OpenAIWithSearchProvider(self.api_keys['openai'])
                else:
                    # For variants like openai_search, use search provider
                    providers[provider_name] = OpenAIWithSearchProvider(self.api_keys['openai'])
            
            elif provider_name.startswith('gemini') and self.api_keys.get('gemini'):
                if provider_name == 'gemini':
                    providers[provider_name] =GeminiWithSearchProvider(self.api_keys['gemini'])
                elif provider_name == 'gemini_flash':
                    providers[provider_name] = GeminiWithSearchProvider(self.api_keys['gemini'])  # Same provider, different model
                else:
                    # For variants like gemini_search, use search provider
                    providers[provider_name] = GeminiWithSearchProvider(self.api_keys['gemini'])
            
            elif provider_name.startswith('claude') and self.api_keys.get('claude'):
                if provider_name == 'claude':
                    providers[provider_name] = ClaudeWithSearchProvider(self.api_keys['claude'])
                else:
                    # For variants like claude_search, use search provider
                    providers[provider_name] = ClaudeWithSearchProvider(self.api_keys['claude'])
        
        return providers
    
    def _get_model_for_provider(self, provider_name: str, prompt_id: int) -> Optional[str]:
        """Get model name for provider and prompt"""
        try:
            # First try exact match
            if provider_name in MODEL_CONFIG[f"prompt{prompt_id}"]:
                return MODEL_CONFIG[f"prompt{prompt_id}"][provider_name]
            
            # Then try base name (openai_search -> openai)
            base_name = provider_name.split('_')[0]
            if base_name in MODEL_CONFIG[f"prompt{prompt_id}"]:
                return MODEL_CONFIG[f"prompt{prompt_id}"][base_name]
            
            logger.warning(f"No model config for {provider_name} or {base_name} in prompt{prompt_id}")
            return None
            
        except KeyError:
            logger.warning(f"No prompt{prompt_id} config found")
            return None
    
    def _format_prompt(self, prompt_id: int, **kwargs) -> str:
        """Format prompt template with provided variables"""
        template = PROMPT_TEMPLATES[prompt_id]
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt {prompt_id}")
            raise
    
    def run_prompt1(self, processed_data: ProcessedLogData, provider_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run prompt 1: Log type classification and entity extraction"""
        if provider_names is None:
            # Only use providers that are configured for prompt1
            provider_names = list(MODEL_CONFIG.get("prompt1", {}).keys())
        
        # Filter to only use available providers that have models configured
        valid_providers = []
        for provider_name in provider_names:
            if provider_name in self.providers and self._get_model_for_provider(provider_name, 1):
                valid_providers.append(provider_name)
        
        logger.info(f"Running Prompt 1 with providers: {valid_providers}")
        
        # Format prompt with log data
        prompt = self._format_prompt(1, log_data=processed_data.cleaned_data)
        
        results = {}
        for provider_name in valid_providers:
            provider = self.providers[provider_name]
            model = self._get_model_for_provider(provider_name, 1)
            
            try:
                # Generate response
                response = provider.generate(model, prompt, temperature=0.7)
                
                # Extract JSON from response
                json_data = extract_json_from_text(response)
                
                results[provider_name] = {
                    "parsed_json": json_data,
                    "raw_response": response,  # Save raw response for debugging
                    "model": model,
                    "provider": provider_name
                }
                
                logger.info(f"✓ Prompt 1 completed for {provider_name}")
                
            except Exception as e:
                logger.error(f"✗ Prompt 1 failed for {provider_name}: {str(e)}")
                results[provider_name] = {
                    "error": str(e),
                    "parsed_json": {},
                    "model": model,
                    "provider": provider_name
                }
        
        return results
    
    def run_prompt2(self, prompt1_results: Dict[str, Dict[str, Any]], provider_names: List[str] = None) -> Dict[str, str]:
        """Run prompt 2: Generate security analysis report from log_meta_data"""
        if provider_names is None:
            # Only use providers that are configured for prompt2
            provider_names = list(MODEL_CONFIG.get("prompt2", {}).keys())
        
        # Filter to only use available providers that have models configured
        valid_providers = []
        for provider_name in provider_names:
            if provider_name in self.providers and self._get_model_for_provider(provider_name, 2):
                valid_providers.append(provider_name)
        
        logger.info(f"Running Prompt 2 with providers: {valid_providers}")
        
        results = {}
        for provider_name in valid_providers:
            provider = self.providers[provider_name]
            model = self._get_model_for_provider(provider_name, 2)
            
            # Get log_meta_data from prompt 1 (prefer same provider, fallback to first available)
            base_name = provider_name.split('_')[0]
            p1_data = prompt1_results.get(provider_name, 
                                         prompt1_results.get(base_name, {}))
            
            if not p1_data.get('parsed_json'):
                # Try to get from any available result
                for result in prompt1_results.values():
                    if result.get('parsed_json'):
                        p1_data = result
                        break
                        
                if not p1_data.get('parsed_json'):
                    logger.warning(f"No prompt 1 data available for {provider_name}")
                    continue
            
            # Use the entire parsed JSON as log_meta_data
            log_meta_data = p1_data['parsed_json']
            
            try:
                # Format prompt with log_meta_data
                prompt = self._format_prompt(2, log_meta_data=log_meta_data)
                
                # Generate response
                response = provider.generate(model, prompt, temperature=0.7)
                
                results[provider_name] = response
                
                logger.info(f"✓ Prompt 2 completed for {provider_name}")
                
            except Exception as e:
                logger.error(f"✗ Prompt 2 failed for {provider_name}: {str(e)}")
                results[provider_name] = f"Error: {str(e)}"
        
        return results
    
    def run_prompt3(self, prompt1_results: Dict[str, Dict[str, Any]], prompt2_results: Dict[str, str], provider_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run prompt 3: Generate analysis schema from report and log_meta_data"""
        if provider_names is None:
            # Only use providers that are configured for prompt3
            provider_names = list(MODEL_CONFIG.get("prompt3", {}).keys())
        
        # Filter to only use available providers that have models configured
        valid_providers = []
        for provider_name in provider_names:
            if provider_name in self.providers and self._get_model_for_provider(provider_name, 3):
                valid_providers.append(provider_name)
        
        logger.info(f"Running Prompt 3 with providers: {valid_providers}")
        
        results = {}
        
        # Get log_meta_data once (same for all providers)
        log_meta_data = None
        for result in prompt1_results.values():
            if result.get('parsed_json'):
                log_meta_data = result['parsed_json']
                break
        
        if not log_meta_data:
            logger.error("No prompt 1 data available for any provider")
            return results
        
        # Process each provider with ALL prompt2 reports
        for provider_name in valid_providers:
            provider = self.providers[provider_name]
            model = self._get_model_for_provider(provider_name, 3)
            
            # Create one result per prompt2 report for this provider
            for report_provider, report_content in prompt2_results.items():
                try:
                    # Format prompt with this specific report and log_meta_data
                    prompt = self._format_prompt(3, report=report_content, log_meta_data=log_meta_data)
                    
                    # Generate response
                    response = provider.generate(model, prompt, temperature=0.7)
                    
                    # Extract JSON from response
                    json_data = extract_json_from_text(response)
                    
                    # Create unique key combining provider and report source
                    result_key = f"{provider_name}_from_{report_provider}_report"
                    
                    results[result_key] = {
                        "parsed_json": json_data,
                        "raw_response": response,  # Always store raw response for debugging
                        "model": model,
                        "provider": provider_name,
                        "report_source": report_provider
                    }
                    
                    logger.info(f"✓ Prompt 3 completed for {provider_name} using {report_provider} report")
                    
                except Exception as e:
                    logger.error(f"✗ Prompt 3 failed for {provider_name} with {report_provider} report: {str(e)}")
                    result_key = f"{provider_name}_from_{report_provider}_report"
                    results[result_key] = {
                        "error": str(e),
                        "parsed_json": {},
                        "raw_response": "",  # Empty response for error cases
                        "model": model,
                        "provider": provider_name,
                        "report_source": report_provider
                    }
        
        return results
    
    def run_full_pipeline(self, processed_data: ProcessedLogData, provider_names: List[str] = None) -> Dict[str, Any]:
        """Run the complete three-stage pipeline"""
        logger.info("Starting full three-stage pipeline")
        
        # Stage 1: Log analysis
        prompt1_results = self.run_prompt1(processed_data, provider_names)
        
        # Stage 2: Research
        prompt2_results = self.run_prompt2(prompt1_results, provider_names)
        
        # Stage 3: Schema generation (needs both prompt1 and prompt2 results)
        prompt3_results = self.run_prompt3(prompt1_results, prompt2_results, provider_names)
        
        return {
            "prompt1": prompt1_results,
            "prompt2": prompt2_results,
            "prompt3": prompt3_results
        }
    
    def run_openai_comparison(self, processed_data: ProcessedLogData, models: List[str] = None) -> Dict[str, Any]:
        """
        Run pipeline comparing different OpenAI models and capabilities
        
        Args:
            processed_data: Log data to process
            models: List of OpenAI model configurations to test
                   e.g., ['gpt-4o', 'o4-mini', 'o4-mini-deep-research']
        """
        if models is None:
            models = ['gpt-4o', 'o4-mini', 'o4-mini-deep-research']
        
        logger.info(f"Running OpenAI model comparison with models: {models}")
        
        # Map models to provider configurations
        provider_configs = []
        for model in models:
            if model.startswith('o') and any(x in model for x in ['o4', 'o3', 'o1']):
                # Reasoning models
                if 'deep-research' in model:
                    provider_configs.append(('openai_search', model))  # Use search for deep research
                else:
                    provider_configs.append(('openai_thinking', model))  # Use thinking for reasoning
            else:
                # GPT models
                provider_configs.append(('openai', model))
        
        results = {}
        for provider_name, model in provider_configs:
            logger.info(f"Testing {provider_name} with model {model}")
            
            # Update model config temporarily for this test
            original_config = MODEL_CONFIG.copy()
            try:
                # Set model for all prompts
                for prompt_id in [1, 2, 3]:
                    if f"prompt{prompt_id}" not in MODEL_CONFIG:
                        MODEL_CONFIG[f"prompt{prompt_id}"] = {}
                    MODEL_CONFIG[f"prompt{prompt_id}"]["openai"] = model
                
                # Run pipeline with single provider
                result = self.run_full_pipeline(processed_data, [provider_name])
                results[f"{provider_name}_{model}"] = result
                
            finally:
                # Restore original config
                MODEL_CONFIG.clear()
                MODEL_CONFIG.update(original_config)
        
        return results