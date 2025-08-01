#!/usr/bin/env python3
"""
LLM-based Conflict Arbitration System for Multi-Agent Consensus

This system uses Large Language Models to resolve conflicts when consensus algorithms
produce different or contradictory results. It can integrate with various LLM providers
and provides domain-specific arbitration logic.

Architecture:
- Conflict Detection: Identifies disagreements between consensus algorithms
- Context Generation: Creates structured prompts for LLM arbitration  
- LLM Integration: Supports multiple LLM providers (OpenAI, Anthropic, etc.)
- Resolution Logic: Applies domain-specific rules to LLM responses
- Quality Assessment: Evaluates arbitration quality and confidence
"""

import numpy as np
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of conflicts that can be arbitrated"""
    ALGORITHM_DISAGREEMENT = "algorithm_disagreement"
    CONFIDENCE_DISCREPANCY = "confidence_discrepancy"
    SEMANTIC_CONFLICT = "semantic_conflict"
    STRUCTURAL_MISMATCH = "structural_mismatch"
    THRESHOLD_BOUNDARY = "threshold_boundary"

class ArbitrationStrategy(Enum):
    """Strategies for LLM arbitration"""
    SINGLE_LLM = "single_llm"
    MULTI_LLM_CONSENSUS = "multi_llm_consensus"
    HIERARCHICAL_REVIEW = "hierarchical_review"
    ENSEMBLE_WEIGHTED = "ensemble_weighted"

@dataclass
class ConflictCase:
    """Represents a specific conflict that needs arbitration"""
    conflict_id: str
    conflict_type: ConflictType
    conflicting_algorithms: List[str]
    conflicting_results: Dict[str, Any]
    context_data: Dict[str, Any]
    domain: str = "security"
    priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate conflict hash for deduplication"""
        context_str = json.dumps(self.conflicting_results, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(context_str.encode()).hexdigest()[:12]

@dataclass
class ArbitrationResult:
    """Result of LLM arbitration"""
    conflict_id: str
    resolution: Dict[str, Any]
    confidence_score: float
    arbitration_reasoning: str
    llm_responses: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate quality metrics"""
        self.quality_score = min(1.0, self.confidence_score * 0.8 + (1.0 if self.arbitration_reasoning else 0.0) * 0.2)

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.request_count = 0
        self.success_count = 0
        
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response from LLM provider"""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider usage statistics"""
        success_rate = self.success_count / max(1, self.request_count)
        return {
            'provider_name': self.provider_name,
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'success_rate': success_rate
        }

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and development"""
    
    def __init__(self, provider_name: str = "mock_llm", config: Dict[str, Any] = None):
        super().__init__(provider_name, config or {})
        
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock response"""
        self.request_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock response based on prompt content
        if "malicious" in prompt.lower():
            response = {
                'text': 'Based on the conflicting results, I recommend selecting the pattern with higher specificity and clearer indicators. The pattern mentioning specific attack techniques should be prioritized.',
                'confidence': 0.85,
                'reasoning': 'Selected based on technical specificity and indicator clarity'
            }
        elif "network" in prompt.lower():
            response = {
                'text': 'For network-related conflicts, prioritize patterns that include both source and destination context with specific protocol information.',
                'confidence': 0.78,
                'reasoning': 'Network patterns require bidirectional context for accuracy'
            }
        else:
            response = {
                'text': 'Recommend merging the conflicting elements by taking the most comprehensive aspects from each algorithm result.',
                'confidence': 0.70,
                'reasoning': 'General merge strategy for unknown domain conflicts'
            }
        
        self.success_count += 1
        return response

class LLMConflictArbitrator:
    """
    Main arbitration system for resolving conflicts between consensus algorithms
    
    This system can work with multiple LLM providers and provides domain-specific
    arbitration logic for security-related conflicts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM conflict arbitrator"""
        self.config = config or self._get_default_config()
        
        # LLM providers
        self.llm_providers = {}
        
        # Arbitration cache
        self.arbitration_cache = {}
        
        # Conflict resolution templates
        self.prompt_templates = self._init_prompt_templates()
        
        # Domain-specific rules
        self.domain_rules = self._init_domain_rules()
        
        # Statistics
        self.arbitration_stats = {
            'total_conflicts': 0,
            'resolved_conflicts': 0,
            'cache_hits': 0,
            'llm_requests': 0
        }
        
        # Initialize mock provider for development
        self.register_llm_provider("mock", MockLLMProvider())
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'cache_enabled': True,
            'cache_ttl_seconds': 3600,
            'max_parallel_llm_requests': 3,
            'arbitration_timeout_seconds': 30,
            'confidence_threshold': 0.6,
            'domain_weights': {
                'security': 1.0,
                'network': 0.9,
                'authentication': 0.95,
                'endpoint': 0.8
            },
            'llm_request_timeout': 15
        }
    
    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialize domain-specific prompt templates"""
        return {
            'security_pattern_conflict': '''
You are an expert cybersecurity analyst resolving conflicts between different AI models analyzing security patterns.

## Conflict Context:
{context}

## Conflicting Results:
{conflicting_results}

## Task:
Analyze the conflicting results and provide:
1. Your recommended resolution
2. Confidence score (0.0-1.0)
3. Detailed reasoning

Focus on security relevance, technical accuracy, and practical applicability.

## Response Format:
```json
{
    "resolution": {your_recommended_result},
    "confidence": confidence_score,
    "reasoning": "detailed_explanation"
}
```
''',
            'algorithm_disagreement': '''
You are resolving a disagreement between consensus algorithms: {algorithms}

## Original Data:
{original_data}

## Algorithm Results:
{algorithm_results}

## Your Task:
1. Evaluate which algorithm result is most reliable
2. Consider merging compatible elements
3. Provide confidence assessment

## Response Format:
```json
{
    "resolution": {final_result},
    "confidence": confidence_score,
    "reasoning": "explanation_of_decision"
}
```
''',
            'confidence_discrepancy': '''
Multiple algorithms agree on content but show different confidence levels:

## Content: 
{content}

## Confidence Scores:
{confidence_scores}

## Additional Context:
{context}

Determine the most appropriate confidence level considering:
- Algorithm reliability
- Data quality indicators
- Domain expertise requirements

## Response Format:
```json
{
    "resolution": {
        "content": original_content,
        "confidence": adjusted_confidence
    },
    "confidence": meta_confidence,
    "reasoning": "confidence_adjustment_rationale"
}
```
'''
        }
    
    def _init_domain_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize domain-specific arbitration rules"""
        return {
            'security': {
                'priority_fields': ['attack_type', 'severity', 'indicators', 'mitre_technique'],
                'merge_strategies': {
                    'attack_patterns': 'union_with_dedup',
                    'indicators': 'union_all',
                    'severity': 'max_value'
                },
                'conflict_weights': {
                    'embedding_similarity': 0.25,
                    'bft_consensus': 0.30,
                    'dempster_shafer': 0.25,
                    'mcts_optimization': 0.20
                }
            },
            'authentication': {
                'priority_fields': ['logon_type', 'event_code', 'authentication_package'],
                'merge_strategies': {
                    'logon_patterns': 'technical_specificity',
                    'event_codes': 'union_all',
                    'risk_level': 'max_value'
                },
                'conflict_weights': {
                    'embedding_similarity': 0.20,
                    'bft_consensus': 0.35,
                    'dempster_shafer': 0.25,
                    'mcts_optimization': 0.20
                }
            },
            'network': {
                'priority_fields': ['protocol', 'direction', 'ports', 'ip_ranges'],
                'merge_strategies': {
                    'traffic_patterns': 'bidirectional_context',
                    'ports': 'union_all',
                    'protocols': 'specific_over_general'
                },
                'conflict_weights': {
                    'embedding_similarity': 0.30,
                    'bft_consensus': 0.25,
                    'dempster_shafer': 0.25,
                    'mcts_optimization': 0.20
                }
            }
        }
    
    def register_llm_provider(self, provider_name: str, provider_instance: LLMProvider):
        """Register an LLM provider for arbitration"""
        self.llm_providers[provider_name] = provider_instance
        logger.info(f"Registered LLM provider: {provider_name}")
    
    def detect_conflicts(self, 
                        algorithm_results: Dict[str, Any],
                        consensus_threshold: float = 0.1) -> List[ConflictCase]:
        """
        Detect conflicts between algorithm results
        
        Args:
            algorithm_results: Results from different consensus algorithms
            consensus_threshold: Threshold for detecting significant disagreements
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Extract consensus items from each algorithm
        algorithm_items = {}
        for algo_name, result in algorithm_results.items():
            if 'error' not in result and 'consensus_items' in result:
                algorithm_items[algo_name] = result['consensus_items']
        
        if len(algorithm_items) < 2:
            return conflicts  # Need at least 2 algorithms to have conflicts
        
        # Find algorithm disagreements
        conflicts.extend(self._detect_algorithm_disagreements(algorithm_items))
        
        # Find confidence discrepancies
        conflicts.extend(self._detect_confidence_discrepancies(algorithm_items))
        
        # Find semantic conflicts
        conflicts.extend(self._detect_semantic_conflicts(algorithm_items))
        
        logger.info(f"Detected {len(conflicts)} conflicts requiring arbitration")
        return conflicts
    
    def _detect_algorithm_disagreements(self, algorithm_items: Dict[str, Dict]) -> List[ConflictCase]:
        """Detect disagreements between algorithm results"""
        conflicts = []
        
        # Group items by normalized names
        item_groups = defaultdict(lambda: defaultdict(list))
        
        for algo_name, items in algorithm_items.items():
            for item_id, item_data in items.items():
                # Normalize item name for comparison
                normalized_name = self._normalize_item_name(item_id)
                item_groups[normalized_name][algo_name].append((item_id, item_data))
        
        # Find groups where algorithms disagree
        for item_name, algo_results in item_groups.items():
            if len(algo_results) > 1:  # Multiple algorithms have this item
                # Check for significant disagreements
                if self._has_significant_disagreement(algo_results):
                    conflict = ConflictCase(
                        conflict_id=f"disagreement_{item_name}_{int(time.time())}",
                        conflict_type=ConflictType.ALGORITHM_DISAGREEMENT,
                        conflicting_algorithms=list(algo_results.keys()),
                        conflicting_results={algo: results for algo, results in algo_results.items()},
                        context_data={'item_name': item_name},
                        domain=self._infer_domain(algo_results)
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_confidence_discrepancies(self, algorithm_items: Dict[str, Dict]) -> List[ConflictCase]:
        """Detect significant confidence score discrepancies"""
        conflicts = []
        
        # Group items and check confidence variations
        item_groups = defaultdict(list)
        
        for algo_name, items in algorithm_items.items():
            for item_id, item_data in items.items():
                confidence = self._extract_confidence(item_data)
                if confidence is not None:
                    item_groups[self._normalize_item_name(item_id)].append((algo_name, confidence, item_data))
        
        # Find items with high confidence variation
        for item_name, algo_confidences in item_groups.items():
            if len(algo_confidences) > 1:
                confidences = [conf for _, conf, _ in algo_confidences]
                if max(confidences) - min(confidences) > 0.3:  # Significant discrepancy
                    conflict = ConflictCase(
                        conflict_id=f"confidence_{item_name}_{int(time.time())}",
                        conflict_type=ConflictType.CONFIDENCE_DISCREPANCY,
                        conflicting_algorithms=[algo for algo, _, _ in algo_confidences],
                        conflicting_results={
                            algo: {'confidence': conf, 'data': data}
                            for algo, conf, data in algo_confidences
                        },
                        context_data={'item_name': item_name, 'confidence_range': (min(confidences), max(confidences))}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_semantic_conflicts(self, algorithm_items: Dict[str, Dict]) -> List[ConflictCase]:
        """Detect semantic conflicts in content"""
        conflicts = []
        
        # This would involve more sophisticated semantic analysis
        # For now, detect conflicts based on contradictory field values
        
        field_conflicts = defaultdict(lambda: defaultdict(list))
        
        for algo_name, items in algorithm_items.items():
            for item_id, item_data in items.items():
                if isinstance(item_data, dict):
                    for field, value in item_data.items():
                        if field in ['severity', 'risk_level', 'attack_type', 'category']:
                            field_conflicts[field][str(value)].append((algo_name, item_id, item_data))
        
        # Find fields with multiple contradictory values
        for field, value_groups in field_conflicts.items():
            if len(value_groups) > 1:
                conflict = ConflictCase(
                    conflict_id=f"semantic_{field}_{int(time.time())}",
                    conflict_type=ConflictType.SEMANTIC_CONFLICT,
                    conflicting_algorithms=list(set(algo for _, values in value_groups.items() for algo, _, _ in values)),
                    conflicting_results={
                        f"{field}_values": {value: [(algo, item_id) for algo, item_id, _ in items] 
                                           for value, items in value_groups.items()}
                    },
                    context_data={'field': field, 'value_count': len(value_groups)}
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def arbitrate_conflicts(self, 
                                 conflicts: List[ConflictCase],
                                 strategy: ArbitrationStrategy = ArbitrationStrategy.SINGLE_LLM,
                                 preferred_provider: str = "mock") -> Dict[str, ArbitrationResult]:
        """
        Arbitrate detected conflicts using LLM providers
        
        Args:
            conflicts: List of conflicts to resolve
            strategy: Arbitration strategy to use
            preferred_provider: Preferred LLM provider
            
        Returns:
            Dictionary mapping conflict IDs to arbitration results
        """
        start_time = time.time()
        
        if not conflicts:
            return {}
        
        logger.info(f"Arbitrating {len(conflicts)} conflicts using strategy: {strategy.value}")
        
        results = {}
        
        # Check cache first
        cached_results, uncached_conflicts = self._check_arbitration_cache(conflicts)
        results.update(cached_results)
        
        if not uncached_conflicts:
            logger.info(f"All {len(conflicts)} conflicts resolved from cache")
            return results
        
        # Process uncached conflicts
        if strategy == ArbitrationStrategy.SINGLE_LLM:
            new_results = await self._single_llm_arbitration(uncached_conflicts, preferred_provider)
        elif strategy == ArbitrationStrategy.MULTI_LLM_CONSENSUS:
            new_results = await self._multi_llm_arbitration(uncached_conflicts)
        elif strategy == ArbitrationStrategy.HIERARCHICAL_REVIEW:
            new_results = await self._hierarchical_arbitration(uncached_conflicts, preferred_provider)
        elif strategy == ArbitrationStrategy.ENSEMBLE_WEIGHTED:
            new_results = await self._ensemble_arbitration(uncached_conflicts)
        else:
            raise ValueError(f"Unknown arbitration strategy: {strategy}")
        
        # Cache new results
        self._cache_arbitration_results(new_results)
        
        results.update(new_results)
        
        # Update statistics
        self.arbitration_stats['total_conflicts'] += len(conflicts)
        self.arbitration_stats['resolved_conflicts'] += len(results)
        self.arbitration_stats['cache_hits'] += len(cached_results)
        
        processing_time = time.time() - start_time
        logger.info(f"Arbitrated {len(conflicts)} conflicts in {processing_time:.2f}s")
        
        return results
    
    async def _single_llm_arbitration(self, 
                                     conflicts: List[ConflictCase], 
                                     provider_name: str) -> Dict[str, ArbitrationResult]:
        """Arbitrate conflicts using a single LLM provider"""
        
        if provider_name not in self.llm_providers:
            logger.error(f"LLM provider '{provider_name}' not registered")
            return {}
        
        provider = self.llm_providers[provider_name]
        results = {}
        
        # Process conflicts in parallel
        max_workers = self.config.get('max_parallel_llm_requests', 3)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_conflict = {}
            
            for conflict in conflicts:
                future = executor.submit(self._process_single_conflict, conflict, provider)
                future_to_conflict[future] = conflict
            
            timeout = self.config.get('arbitration_timeout_seconds', 30)
            
            for future in as_completed(future_to_conflict, timeout=timeout):
                conflict = future_to_conflict[future]
                try:
                    result = future.result()
                    if result:
                        results[conflict.conflict_id] = result
                except Exception as e:
                    logger.error(f"Failed to arbitrate conflict {conflict.conflict_id}: {e}")
        
        return results
    
    def _process_single_conflict(self, conflict: ConflictCase, provider: LLMProvider) -> Optional[ArbitrationResult]:
        """Process a single conflict with an LLM provider"""
        start_time = time.time()
        
        try:
            # Generate prompt based on conflict type and domain
            prompt = self._generate_arbitration_prompt(conflict)
            
            # Get LLM response (using asyncio.run for sync context)
            llm_response = asyncio.run(provider.generate_response(
                prompt, 
                temperature=0.3,
                max_tokens=1000
            ))
            
            # Parse and validate response
            resolution = self._parse_llm_response(llm_response, conflict)
            
            if resolution:
                result = ArbitrationResult(
                    conflict_id=conflict.conflict_id,
                    resolution=resolution['resolution'],
                    confidence_score=resolution.get('confidence', 0.5),
                    arbitration_reasoning=resolution.get('reasoning', ''),
                    llm_responses={provider.provider_name: llm_response},
                    processing_time=time.time() - start_time,
                    metadata={
                        'provider': provider.provider_name,
                        'conflict_type': conflict.conflict_type.value,
                        'domain': conflict.domain
                    }
                )
                
                return result
            
        except Exception as e:
            logger.error(f"Error processing conflict {conflict.conflict_id}: {e}")
        
        return None
    
    async def _multi_llm_arbitration(self, conflicts: List[ConflictCase]) -> Dict[str, ArbitrationResult]:
        """Arbitrate conflicts using multiple LLM providers and consensus"""
        # Implementation would involve querying multiple providers and finding consensus
        # For now, fall back to single LLM
        return await self._single_llm_arbitration(conflicts, list(self.llm_providers.keys())[0])
    
    async def _hierarchical_arbitration(self, 
                                       conflicts: List[ConflictCase], 
                                       primary_provider: str) -> Dict[str, ArbitrationResult]:
        """Hierarchical arbitration with escalation for difficult conflicts"""
        # Start with primary provider, escalate complex conflicts to more sophisticated models
        return await self._single_llm_arbitration(conflicts, primary_provider)
    
    async def _ensemble_arbitration(self, conflicts: List[ConflictCase]) -> Dict[str, ArbitrationResult]:
        """Weighted ensemble arbitration across multiple providers"""
        return await self._single_llm_arbitration(conflicts, list(self.llm_providers.keys())[0])
    
    def _generate_arbitration_prompt(self, conflict: ConflictCase) -> str:
        """Generate appropriate prompt for conflict arbitration"""
        
        conflict_type_key = conflict.conflict_type.value
        domain = conflict.domain
        
        # Select appropriate template
        template_key = conflict_type_key
        if domain == 'security' and 'pattern' in str(conflict.context_data):
            template_key = 'security_pattern_conflict'
        
        template = self.prompt_templates.get(template_key, self.prompt_templates['algorithm_disagreement'])
        
        # Prepare context data
        context = {
            'domain': domain,
            'conflict_type': conflict_type_key,
            'algorithms': conflict.conflicting_algorithms,
            'priority': conflict.priority
        }
        
        # Format conflicting results for readability
        formatted_results = self._format_conflicting_results(conflict.conflicting_results)
        
        # Fill template
        prompt = template.format(
            context=json.dumps(context, indent=2),
            conflicting_results=formatted_results,
            algorithms=', '.join(conflict.conflicting_algorithms),
            original_data=json.dumps(conflict.context_data, indent=2),
            algorithm_results=formatted_results,
            content=conflict.context_data.get('item_name', ''),
            confidence_scores=self._extract_confidence_info(conflict.conflicting_results)
        )
        
        return prompt
    
    def _format_conflicting_results(self, conflicting_results: Dict[str, Any]) -> str:
        """Format conflicting results for LLM prompt"""
        formatted = {}
        
        for key, value in conflicting_results.items():
            if isinstance(value, dict):
                # Simplify complex nested structures
                simplified = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool)):
                        simplified[k] = v
                    elif isinstance(v, list) and len(v) < 10:
                        simplified[k] = v
                    elif isinstance(v, dict):
                        # Take only key fields
                        key_fields = ['name', 'description', 'type', 'confidence', 'severity']
                        simplified[k] = {kf: v.get(kf) for kf in key_fields if kf in v}
                formatted[key] = simplified
            else:
                formatted[key] = value
        
        return json.dumps(formatted, indent=2, default=str)
    
    def _extract_confidence_info(self, conflicting_results: Dict[str, Any]) -> str:
        """Extract confidence information for prompt"""
        confidence_info = {}
        
        for key, value in conflicting_results.items():
            if isinstance(value, dict):
                conf = self._extract_confidence(value)
                if conf is not None:
                    confidence_info[key] = conf
        
        return json.dumps(confidence_info, indent=2)
    
    def _parse_llm_response(self, llm_response: Dict[str, Any], conflict: ConflictCase) -> Optional[Dict[str, Any]]:
        """Parse and validate LLM response"""
        try:
            response_text = llm_response.get('text', '')
            
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields
                if 'resolution' in parsed and 'confidence' in parsed:
                    # Ensure confidence is within valid range
                    parsed['confidence'] = max(0.0, min(1.0, float(parsed.get('confidence', 0.5))))
                    return parsed
            
            # Fallback: create response from raw text
            return {
                'resolution': {'arbitration_text': response_text},
                'confidence': llm_response.get('confidence', 0.5),
                'reasoning': response_text[:200] + '...' if len(response_text) > 200 else response_text
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
    
    def _check_arbitration_cache(self, conflicts: List[ConflictCase]) -> Tuple[Dict[str, ArbitrationResult], List[ConflictCase]]:
        """Check cache for existing arbitration results"""
        cached_results = {}
        uncached_conflicts = []
        
        if not self.config.get('cache_enabled', True):
            return {}, conflicts
        
        cache_ttl = self.config.get('cache_ttl_seconds', 3600)
        current_time = time.time()
        
        for conflict in conflicts:
            cache_key = f"{conflict.content_hash}_{conflict.conflict_type.value}"
            
            if cache_key in self.arbitration_cache:
                cached_result, timestamp = self.arbitration_cache[cache_key]
                if current_time - timestamp < cache_ttl:
                    cached_results[conflict.conflict_id] = cached_result
                    self.arbitration_stats['cache_hits'] += 1
                    continue
            
            uncached_conflicts.append(conflict)
        
        return cached_results, uncached_conflicts
    
    def _cache_arbitration_results(self, results: Dict[str, ArbitrationResult]):
        """Cache arbitration results"""
        if not self.config.get('cache_enabled', True):
            return
        
        current_time = time.time()
        
        for conflict_id, result in results.items():
            # Create cache key based on result metadata
            cache_key = f"{result.metadata.get('content_hash', conflict_id)}_{result.metadata.get('conflict_type', 'unknown')}"
            self.arbitration_cache[cache_key] = (result, current_time)
    
    def _normalize_item_name(self, item_name: str) -> str:
        """Normalize item names for comparison"""
        # Remove provider prefixes and normalize
        if '::' in item_name:
            item_name = item_name.split('::', 1)[1]
        
        return item_name.lower().replace('_', ' ').strip()
    
    def _has_significant_disagreement(self, algo_results: Dict[str, List]) -> bool:
        """Check if algorithms have significant disagreement"""
        # Simple heuristic: if different algorithms produce different numbers of items
        # or have different confidence levels, consider it a disagreement
        
        item_counts = [len(results) for results in algo_results.values()]
        if max(item_counts) - min(item_counts) > 1:
            return True
        
        # Check confidence variations
        all_confidences = []
        for results in algo_results.values():
            for _, item_data in results:
                conf = self._extract_confidence(item_data)
                if conf is not None:
                    all_confidences.append(conf)
        
        if len(all_confidences) > 1:
            return max(all_confidences) - min(all_confidences) > 0.3
        
        return False
    
    def _extract_confidence(self, item_data: Any) -> Optional[float]:
        """Extract confidence score from item data"""
        if isinstance(item_data, dict):
            for conf_key in ['confidence', 'confidence_score', 'score', 'belief', 'weight']:
                if conf_key in item_data:
                    try:
                        return float(item_data[conf_key])
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _infer_domain(self, algo_results: Dict[str, List]) -> str:
        """Infer domain from algorithm results"""
        # Analyze content to determine domain
        all_content = []
        for results in algo_results.values():
            for _, item_data in results:
                if isinstance(item_data, dict):
                    all_content.extend(str(v).lower() for v in item_data.values() if v)
                else:
                    all_content.append(str(item_data).lower())
        
        content_text = ' '.join(all_content)
        
        if any(term in content_text for term in ['logon', 'authentication', 'kerberos', 'ntlm']):
            return 'authentication'
        elif any(term in content_text for term in ['network', 'traffic', 'protocol', 'port']):
            return 'network'
        elif any(term in content_text for term in ['malicious', 'attack', 'threat', 'exploit']):
            return 'security'
        else:
            return 'generic'
    
    def get_arbitration_statistics(self) -> Dict[str, Any]:
        """Get arbitration system statistics"""
        provider_stats = {name: provider.get_provider_stats() 
                         for name, provider in self.llm_providers.items()}
        
        return {
            'arbitration_stats': self.arbitration_stats.copy(),
            'provider_stats': provider_stats,
            'cache_size': len(self.arbitration_cache),
            'registered_providers': list(self.llm_providers.keys())
        }
    
    def clear_cache(self):
        """Clear arbitration cache"""
        self.arbitration_cache.clear()
        logger.info("Arbitration cache cleared")
    
    def create_custom_similarity_function(self, arbitration_results: Dict[str, ArbitrationResult]) -> callable:
        """Create a similarity function based on arbitration learnings"""
        
        def learned_similarity(item1: Any, item2: Any) -> float:
            """Similarity function enhanced by arbitration results"""
            base_similarity = 0.0
            
            # Apply lessons learned from arbitration
            if isinstance(item1, dict) and isinstance(item2, dict):
                # Check if these item types have been arbitrated before
                for result in arbitration_results.values():
                    if 'pattern_similarity_boost' in result.metadata:
                        boost_rules = result.metadata['pattern_similarity_boost']
                        # Apply boost rules
                        for field, boost in boost_rules.items():
                            if field in item1 and field in item2:
                                if item1[field] == item2[field]:
                                    base_similarity += boost
                
                # Default similarity calculation
                common_fields = set(item1.keys()) & set(item2.keys())
                total_fields = set(item1.keys()) | set(item2.keys())
                
                if total_fields:
                    base_similarity += len(common_fields) / len(total_fields) * 0.5
            
            return min(1.0, base_similarity)
        
        return learned_similarity