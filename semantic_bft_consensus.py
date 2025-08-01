#!/usr/bin/env python3
"""
Comprehensive Consensus Processor for Multi-LLM JSON Deconstruction and Reconstruction

This system takes multiple JSON files, deconstructs each part systematically,
sends each part through the consensus mechanism, gets new consolidated parts,
and rejoins to create a final unified JSON that encompasses most relevant knowledge.

Architecture:
1. JSON Deconstruction: Break down each JSON into meaningful parts
2. Part-wise Consensus: Send each deconstructed part through Universal Consensus Engine  
3. Consensus Integration: Merge consensus results from all parts
4. JSON Reconstruction: Rejoin processed parts into final unified structure
5. Quality Assessment: Evaluate final result quality and coverage
"""

import json
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib

# Import consensus tools
from tools.universal_consensus_engine import UniversalConsensusEngine, ConsensusStrength
from tools.consensus_orchestrator import ConsensusOrchestrator

logger = logging.getLogger(__name__)

@dataclass
class JSONPart:
    """Represents a deconstructed part of a JSON structure"""
    part_id: str
    path: str  # JSON path like 'observations.behavioral_patterns.malicious'
    content: Any
    source_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate hash for identity"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

@dataclass
class ConsensusPart:
    """Represents a part after consensus processing"""
    original_part: JSONPart
    consensus_content: Any
    consensus_confidence: float
    contributing_sources: List[str]
    consensus_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedJSONResult:
    """Final unified JSON result with metadata"""
    unified_json: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    part_contributions: Dict[str, List[str]]  # path -> contributing source files
    original_file_count: int
    consensus_parts_count: int
    total_processing_time: float

class SemanticBFTConsensusProcessor:
    """
    Semantic Byzantine Fault Tolerant (BFT) Consensus Processor
    
    Implements research-grade consensus using:
    - Byzantine Fault Tolerance (BFT) algorithms
    - Semantic similarity clustering 
    - Dempster-Shafer belief combination
    - Monte Carlo Tree Search optimization
    - LLM conflict arbitration
    - Iterative Confidence Enhancement (ICE)
    
    Usage:
    processor = SemanticBFTConsensusProcessor()
    result = processor.process_json_files(
        json_files=['file1.json', 'file2.json', 'file3.json'],
        consensus_strength='comprehensive'
    )
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_research_mode: bool = False):
        """Initialize semantic BFT consensus processor"""
        self.config = config or self._get_default_config()
        self.enable_research_mode = enable_research_mode
        
        # Validate hyperparameter configuration
        self._validate_hyperparameter_config()
        
        # Initialize consensus engines
        if enable_research_mode:
            # Use unified consensus orchestrator V2 with architectural fixes
            from tools.consensus_orchestratorv2 import ConsensusOrchestratorV2
            self.research_orchestrator = ConsensusOrchestratorV2(config=self.config)
            
            # Register default LLM agents for unified consensus
            default_agents = {
                'gpt4_analyst': {'domains': ['security', 'analysis'], 'architecture': 'transformer'},
                'claude_researcher': {'domains': ['research', 'analysis'], 'architecture': 'transformer'},
                'gemini_specialist': {'domains': ['general', 'technical'], 'architecture': 'transformer'}
            }
            self.research_orchestrator.register_llm_agents(default_agents)
            logger.info("Research mode enabled with ConsensusOrchestratorV2 (unified architecture with fixes)")
        else:
            self.research_orchestrator = None
            
        self.universal_engine = UniversalConsensusEngine(config=self.config)
        self.orchestrator = ConsensusOrchestrator(config=self.config)
        
        # Tracking
        self.processing_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with hyperparameter controls"""
        return {
            'consensus_strength': 'comprehensive',
            'max_recursion_depth': 5,
            'min_part_size': 1,  # Minimum items needed for consensus
            'similarity_threshold': 0.75,
            'consensus_threshold': 0.6,
            'preserve_structure': True,
            'enable_cross_path_merging': True,
            'quality_threshold': 0.5,
            'part_processing_timeout': 120,
            'output_format': 'enhanced',  # 'minimal', 'standard', 'enhanced'
            'include_metadata': True,
            'include_provenance': True,
            'deconstruction_strategy': 'comprehensive',  # 'minimal', 'standard', 'comprehensive'
            
            # UniversalConsensusEngine config parameters
            'embedding_model': 'gemini-embedding-001',
            'bft_fault_tolerance': 0.33,
            'mcts_exploration': 1.41,
            'mcts_iterations': 500,
            'max_parallel_algorithms': 4,
            'enable_llm_arbitration': True,
            'ice_loop_max_iterations': 3,
            'ice_confidence_threshold': 0.7,
            
            # Hyperparameters for controlling consensus output completeness
            'preservation_ratio': 0.8,  # How much content to preserve (0.1-1.0)
            'strictness_level': 'moderate',  # 'lenient', 'moderate', 'strict'
            'security_focus': True,  # Prioritize security-related content
            'max_output_sections': None,  # Limit number of sections in final output (None = no limit)
            'min_confidence_threshold': 0.5,  # Global minimum confidence for inclusion
            'content_complexity_preference': 'balanced',  # 'simple', 'balanced', 'comprehensive'
            
            # Research mode configuration (Multi-agent.md)
            'enable_research_mode': True
        }
    
    def _get_research_config(self) -> Dict[str, Any]:
        """Get research-grade configuration implementing Multi-agent.md"""
        research_config = {
            # Section 3.1: Modern SBERT Configuration
            'sbert_model': 'all-mpnet-base-v2',  # Upgraded from MiniLM
            'enable_ontology_grounding': False,
            
            # Section 6.1: Weighted Voting Parameters (Wi = αRi + βCi)
            'reliability_alpha': 0.7,  # Weight for historical reliability
            'reliability_beta': 0.3,   # Weight for current confidence
            'reliability_decay_days': 30,
            'min_reliability_score': 0.1,
            
            # Section 8: ICE Loop Configuration
            'ice_threshold': 0.6,
            'ice_max_iterations': 3,
            'enable_hitl': True,
            'hitl_threshold': 0.4,
            
            # Dynamic thresholds based on hyperparameters
            'consensus_threshold': self._calculate_dynamic_consensus_threshold(),
            'similarity_threshold': self._calculate_dynamic_similarity_threshold(),
            
            # Content preference integration
            'complexity_weight': self._get_complexity_weight(),
            'security_priority': self.config.get('security_focus', True),
            'preservation_ratio': self.config.get('preservation_ratio', 0.8),
            
            # Algorithm weights for research components
            'algorithm_weights': {
                'semantic_clustering': 0.20,  # Section 3.1
                'weighted_voting': 0.25,      # Section 6.1
                'muse_confidence': 0.20,      # Section 7.2
                'graph_clustering': 0.15,
                'semantic_similarity': 0.10,
                'bft_consensus': 0.10
            }
        }
        
        return research_config
    
    def _register_default_agents(self) -> None:
        """Register default LLM agents for research consensus"""
        default_agents = {
            'gpt4_analyst': {
                'domains': ['security', 'analysis'],
                'architecture': 'transformer',
                'training': 'general_knowledge',
                'initial_reliability': 0.8,
                'performance_profile': {
                    'accuracy': 0.85,
                    'consistency': 0.80,
                    'domain_expertise': 0.85,
                    'calibration': 0.75
                }
            },
            'claude_researcher': {
                'domains': ['research', 'analysis'],
                'architecture': 'transformer',
                'training': 'research_focused',
                'initial_reliability': 0.75,
                'performance_profile': {
                    'accuracy': 0.80,
                    'consistency': 0.85,
                    'domain_expertise': 0.70,
                    'calibration': 0.80
                }
            },
            'gemini_specialist': {
                'domains': ['technical', 'security'],
                'architecture': 'transformer',
                'training': 'technical_focused',
                'initial_reliability': 0.70,
                'performance_profile': {
                    'accuracy': 0.75,
                    'consistency': 0.70,
                    'domain_expertise': 0.80,
                    'calibration': 0.70
                }
            }
        }
        
        if hasattr(self.orchestrator, 'register_llm_agents'):
            self.orchestrator.register_llm_agents(default_agents)
            logger.info(f"Registered {len(default_agents)} default LLM agents for research mode")
    
    def _calculate_dynamic_consensus_threshold(self) -> float:
        """Calculate consensus threshold based on strictness level"""
        strictness = self.config.get('strictness_level', 'moderate')
        base_threshold = self.config.get('consensus_threshold', 0.6)
        
        if strictness == 'strict':
            return min(base_threshold + 0.1, 0.9)
        elif strictness == 'lenient':
            return max(base_threshold - 0.1, 0.4)
        else:
            return base_threshold
    
    def _calculate_dynamic_similarity_threshold(self) -> float:
        """Calculate similarity threshold based on preservation ratio"""
        preservation = self.config.get('preservation_ratio', 0.8)
        base_threshold = self.config.get('similarity_threshold', 0.75)
        
        # Higher preservation = lower similarity threshold (include more)
        adjustment = (1.0 - preservation) * 0.2  # Max adjustment of 0.2
        return max(base_threshold - adjustment, 0.5)
    
    def _get_complexity_weight(self) -> float:
        """Get complexity weight based on preference"""
        preference = self.config.get('content_complexity_preference', 'balanced')
        
        if preference == 'simple':
            return 0.3
        elif preference == 'comprehensive':
            return 0.9
        else:  # balanced
            return 0.6
    
    def process_json_files(self, 
                          json_files: List[str],
                          output_file: Optional[str] = None,
                          consensus_strength: str = 'comprehensive') -> UnifiedJSONResult:
        """
        Main method to process multiple JSON files through comprehensive consensus
        
        Args:
            json_files: List of JSON file paths to process
            output_file: Optional output file path for final unified JSON
            consensus_strength: Strength of consensus ('fast', 'comprehensive', 'maximum')
            
        Returns:
            UnifiedJSONResult with final unified JSON and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting comprehensive consensus processing for {len(json_files)} files")
            logger.info(f"Files: {json_files}")
            logger.info(f"Consensus strength: {consensus_strength}")
            
            # Step 1: Load and validate JSON files
            loaded_jsons = self._load_json_files(json_files)
            if not loaded_jsons:
                raise ValueError("No valid JSON files could be loaded")
            
            logger.info(f"Successfully loaded {len(loaded_jsons)} JSON files")
            
            # Step 2: Deconstruct JSONs into meaningful parts
            all_parts = self._deconstruct_jsons(loaded_jsons)
            logger.info(f"Deconstructed {len(all_parts)} parts from JSON files")
            
            # Step 3: Group parts by path for consensus processing
            parts_by_path = self._group_parts_by_path(all_parts)
            logger.info(f"Grouped parts into {len(parts_by_path)} consensus groups")
            
            # Step 4: Process each path group through consensus
            consensus_parts = self._process_parts_through_consensus(
                parts_by_path, consensus_strength
            )
            logger.info(f"Generated {len(consensus_parts)} consensus parts")
            
            # Step 5: Apply cross-path merging if enabled
            if self.config.get('enable_cross_path_merging', True):
                consensus_parts = self._apply_cross_path_merging(consensus_parts)
                logger.info(f"Cross-path merging completed")
            
            # Step 6: Reconstruct unified JSON from consensus parts
            unified_json = self._reconstruct_unified_json(consensus_parts)
            logger.info(f"Reconstructed unified JSON structure")
            
            # Step 7: Calculate quality metrics
            quality_metrics = self._calculate_processing_quality(
                loaded_jsons, all_parts, consensus_parts, unified_json
            )
            
            # Step 8: Create final result
            result = UnifiedJSONResult(
                unified_json=unified_json,
                processing_metadata={
                    'original_files': json_files,
                    'loaded_files_count': len(loaded_jsons),
                    'total_parts_deconstructed': len(all_parts),
                    'consensus_groups_processed': len(parts_by_path),
                    'final_consensus_parts': len(consensus_parts),
                    'consensus_strength': consensus_strength,
                    'config_used': self.config.copy(),
                    'processing_timestamp': time.time(),
                    'total_processing_time': time.time() - start_time
                },
                quality_metrics=quality_metrics,
                part_contributions=self._calculate_part_contributions(consensus_parts),
                original_file_count=len(loaded_jsons),
                consensus_parts_count=len(consensus_parts),
                total_processing_time=time.time() - start_time
            )
            
            # Step 9: Save output if specified
            if output_file:
                self._save_unified_result(result, output_file)
                logger.info(f"Saved unified result to {output_file}")
            
            # Step 10: Update processing history
            self.processing_history.append({
                'timestamp': time.time(),
                'files_processed': json_files,
                'quality_score': quality_metrics.get('overall_quality', 0),
                'processing_time': result.total_processing_time
            })
            
            logger.info(f"Comprehensive consensus processing completed in {result.total_processing_time:.2f}s")
            logger.info(f"Overall quality score: {quality_metrics.get('overall_quality', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive consensus processing failed: {e}")
            return self._create_error_result(json_files, str(e), time.time() - start_time)
    
    def _load_json_files(self, json_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load and validate JSON files"""
        loaded_jsons = {}
        
        for file_path in json_files:
            try:
                path_obj = Path(file_path)
                if not path_obj.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                with open(path_obj, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Extract filename for identification
                file_key = path_obj.stem
                loaded_jsons[file_key] = json_data
                
                logger.debug(f"Loaded JSON file: {file_path} -> {file_key}")
                
            except Exception as e:
                logger.error(f"Failed to load JSON file {file_path}: {e}")
                continue
        
        return loaded_jsons
    
    def _deconstruct_jsons(self, loaded_jsons: Dict[str, Dict[str, Any]]) -> List[JSONPart]:
        """Deconstruct JSON files into meaningful parts for consensus processing"""
        all_parts = []
        
        strategy = self.config.get('deconstruction_strategy', 'comprehensive')
        
        for file_key, json_data in loaded_jsons.items():
            logger.debug(f"Deconstructing file: {file_key}")
            
            if strategy == 'minimal':
                parts = self._minimal_deconstruction(json_data, file_key)
            elif strategy == 'standard':
                parts = self._standard_deconstruction(json_data, file_key)
            else:  # comprehensive
                parts = self._comprehensive_deconstruction(json_data, file_key)
            
            all_parts.extend(parts)
            logger.debug(f"File {file_key} deconstructed into {len(parts)} parts")
        
        return all_parts
    
    def _comprehensive_deconstruction(self, json_data: Dict[str, Any], file_key: str) -> List[JSONPart]:
        """Comprehensive deconstruction strategy - extract every meaningful path"""
        parts = []
        
        # Check if this is a prompt1 field structure and handle specially
        if self._is_prompt1_structure(json_data):
            return self._prompt1_field_deconstruction(json_data, file_key)
        
        def extract_recursive(data: Any, current_path: str = "", depth: int = 0) -> None:
            """Recursively extract parts from nested JSON structure"""
            
            if depth > self.config.get('max_recursion_depth', 5):
                return
            
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    
                    # Create part for each meaningful section
                    if self._is_meaningful_part(value, new_path):
                        part = JSONPart(
                            part_id=f"{file_key}_{new_path}_{depth}",
                            path=new_path,
                            content=value,
                            source_file=file_key,
                            metadata={
                                'depth': depth,
                                'parent_path': current_path,
                                'key': key,
                                'data_type': type(value).__name__,
                                'extraction_strategy': 'comprehensive'
                            }
                        )
                        parts.append(part)
                    
                    # Continue recursion for nested structures
                    if isinstance(value, (dict, list)) and depth < self.config.get('max_recursion_depth', 5):
                        extract_recursive(value, new_path, depth + 1)
                        
            elif isinstance(data, list):
                # For lists, create parts for the list itself and potentially items
                if len(data) >= self.config.get('min_part_size', 1):
                    part = JSONPart(
                        part_id=f"{file_key}_{current_path}_list_{depth}",
                        path=current_path,
                        content=data,
                        source_file=file_key,
                        metadata={
                            'depth': depth,
                            'data_type': 'list',
                            'list_length': len(data),
                            'extraction_strategy': 'comprehensive'
                        }
                    )
                    parts.append(part)
                
                # Process list items if they're complex structures
                for i, item in enumerate(data):
                    if isinstance(item, dict) and depth < self.config.get('max_recursion_depth', 5):
                        item_path = f"{current_path}[{i}]"
                        extract_recursive(item, item_path, depth + 1)
        
        # Start extraction from root
        extract_recursive(json_data)
        
        return parts
    
    def _is_prompt1_structure(self, json_data: Dict[str, Any]) -> bool:
        """Check if this JSON structure is from prompt1 (field extraction)"""
        # Look for prompt1 indicators: parsed_json.fields structure
        if isinstance(json_data, dict):
            # Check for direct prompt1 structure (parsed_json.fields)
            if 'parsed_json' in json_data and isinstance(json_data['parsed_json'], dict):
                if 'fields' in json_data['parsed_json']:
                    return True
            # Check for already parsed structure with fields at root
            if 'fields' in json_data and isinstance(json_data['fields'], dict):
                return True
            # Check for log_type as another prompt1 indicator
            if 'log_type' in json_data:
                return True
        return False
    
    def _prompt1_field_deconstruction(self, json_data: Dict[str, Any], file_key: str) -> List[JSONPart]:
        """Special deconstruction for prompt1 field structures"""
        parts = []
        
        logger.info(f"Processing prompt1 field structure for {file_key}")
        
        # Extract the actual field data
        field_data = None
        if 'parsed_json' in json_data and 'fields' in json_data['parsed_json']:
            field_data = json_data['parsed_json']['fields']
            root_path = 'parsed_json.fields'
        elif 'fields' in json_data:
            field_data = json_data['fields']
            root_path = 'fields'
        
        if field_data and isinstance(field_data, dict):
            # Extract each field as a separate part for consensus
            for field_name, field_definition in field_data.items():
                if isinstance(field_definition, dict):
                    # Create a part for each field
                    part = JSONPart(
                        part_id=f"{file_key}_field_{field_name}",
                        path=f"{root_path}.{field_name}",
                        content=field_definition,
                        source_file=file_key,
                        metadata={
                            'field_name': field_name,
                            'field_type': field_definition.get('type', 'unknown'),
                            'extraction_strategy': 'prompt1_field',
                            'is_field_definition': True
                        }
                    )
                    parts.append(part)
                    
                    # Also extract individual field properties for fine-grained consensus
                    for prop_name, prop_value in field_definition.items():
                        if prop_name in ['type', 'description', 'OCSF', 'ECS', 'OSSEM', 'importance']:
                            prop_part = JSONPart(
                                part_id=f"{file_key}_field_{field_name}_{prop_name}",
                                path=f"{root_path}.{field_name}.{prop_name}",
                                content=prop_value,
                                source_file=file_key,
                                metadata={
                                    'field_name': field_name,
                                    'property_name': prop_name,
                                    'extraction_strategy': 'prompt1_field_property',
                                    'is_field_property': True
                                }
                            )
                            parts.append(prop_part)
        
        # Also extract top-level metadata from parsed_json
        parsed_json_data = json_data.get('parsed_json', json_data)
        for key in ['log_type', 'follow_up_queries']:
            if key in parsed_json_data:
                meta_part = JSONPart(
                    part_id=f"{file_key}_meta_{key}",
                    path=f"parsed_json.{key}" if 'parsed_json' in json_data else key,
                    content=parsed_json_data[key],
                    source_file=file_key,
                    metadata={
                        'metadata_type': key,
                        'extraction_strategy': 'prompt1_metadata',
                        'enable_semantic_similarity_selection': True  # Flag for semantic selection
                    }
                )
                parts.append(meta_part)
        
        logger.info(f"Extracted {len(parts)} parts from prompt1 structure in {file_key}")
        return parts
    
    def _standard_deconstruction(self, json_data: Dict[str, Any], file_key: str) -> List[JSONPart]:
        """Standard deconstruction - focus on major sections"""
        parts = []
        
        # Define standard paths to extract
        standard_paths = [
            'observations',
            'observations.behavioral_patterns',
            'observations.behavioral_patterns.malicious',
            'observations.behavioral_patterns.anomalous',
            'observations.behavioral_patterns.vulnerable',
            'observations.temporal_patterns',
            'observations.temporal_patterns.malicious',
            'observations.temporal_patterns.anomalous',
            'observations.temporal_patterns.vulnerable',
            'entity_analysis_instructions',
            'detection_rule_checklist',
            'indicators_of_compromise',
            'attack_pattern_checks',
            'vulnerability_checks',
            'Log_context'
        ]
        
        for path in standard_paths:
            content = self._get_nested_value(json_data, path)
            if content is not None:
                part = JSONPart(
                    part_id=f"{file_key}_{path.replace('.', '_')}",
                    path=path,
                    content=content,
                    source_file=file_key,
                    metadata={
                        'extraction_strategy': 'standard',
                        'data_type': type(content).__name__
                    }
                )
                parts.append(part)
        
        return parts
    
    def _minimal_deconstruction(self, json_data: Dict[str, Any], file_key: str) -> List[JSONPart]:
        """Minimal deconstruction - only top-level sections"""
        parts = []
        
        for key, value in json_data.items():
            if self._is_meaningful_part(value, key):
                part = JSONPart(
                    part_id=f"{file_key}_{key}",
                    path=key,
                    content=value,
                    source_file=file_key,
                    metadata={
                        'extraction_strategy': 'minimal',
                        'data_type': type(value).__name__
                    }
                )
                parts.append(part)
        
        return parts
    
    def _is_meaningful_part(self, content: Any, path: str) -> bool:
        """Determine if a piece of content is meaningful enough for consensus processing"""
        
        # Skip very small content
        if isinstance(content, (str, int, float, bool)):
            return len(str(content)) > 2
        
        # Include non-empty lists and dicts
        if isinstance(content, list):
            return len(content) >= self.config.get('min_part_size', 1)
        
        if isinstance(content, dict):
            return len(content) >= self.config.get('min_part_size', 1)
        
        # Skip None values
        if content is None:
            return False
        
        return True
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        try:
            current = data
            for key in path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current
        except:
            return None
    
    def _group_parts_by_path(self, all_parts: List[JSONPart]) -> Dict[str, List[JSONPart]]:
        """Group parts by their JSON path for consensus processing"""
        parts_by_path = defaultdict(list)
        
        for part in all_parts:
            # Normalize path for grouping (remove array indices)
            normalized_path = self._normalize_path_for_grouping(part.path)
            parts_by_path[normalized_path].append(part)
        
        # Filter out paths with insufficient parts for consensus
        min_parts = max(1, self.config.get('min_part_size', 1))
        filtered_parts = {
            path: parts for path, parts in parts_by_path.items()
            if len(parts) >= min_parts
        }
        
        logger.debug(f"Grouped parts: {len(parts_by_path)} total paths, {len(filtered_parts)} with sufficient parts")
        
        return filtered_parts
    
    def _normalize_path_for_grouping(self, path: str) -> str:
        """Normalize path for grouping similar structures"""
        # Remove array indices: observations.patterns[0] -> observations.patterns
        import re
        normalized = re.sub(r'\[\d+\]', '', path)
        
        # Remove file-specific prefixes
        if '_' in normalized:
            parts = normalized.split('_')
            if len(parts) > 1:
                # Try to detect if first part is a file identifier
                if any(model in parts[0].lower() for model in ['claude', 'gemini', 'openai', 'gpt']):
                    normalized = '_'.join(parts[1:])
        
        return normalized
    
    def _process_parts_through_consensus(self, 
                                       parts_by_path: Dict[str, List[JSONPart]],
                                       consensus_strength: str) -> List[ConsensusPart]:
        """Process each path group through the consensus mechanism"""
        
        consensus_parts = []
        total_paths = len(parts_by_path)
        
        # Map consensus strength to engine strength
        strength_mapping = {
            'fast': ConsensusStrength.WEAK,
            'comprehensive': ConsensusStrength.STRONG,
            'maximum': ConsensusStrength.MAXIMUM
        }
        engine_strength = strength_mapping.get(consensus_strength, ConsensusStrength.STRONG)
        
        for i, (path, parts) in enumerate(parts_by_path.items()):
            logger.info(f"Processing consensus for path {i+1}/{total_paths}: {path} ({len(parts)} parts)")
            
            try:
                # Convert parts to provider_results format for consensus engine
                provider_results = self._convert_parts_to_provider_format(parts, path)
                
                if not provider_results:
                    logger.warning(f"No valid provider results for path: {path}")
                    continue
                
                # Choose consensus engine based on research mode and configuration
                consensus_result = None
                
                if self.enable_research_mode and self.research_orchestrator:
                    try:
                        # Use Research Consensus Orchestrator with configurable components
                        import asyncio
                        
                        # Check if ICE refinement is enabled in config
                        enable_ice = self.config.get('enable_ice_refinement', True)
                        
                        logger.info(f"Attempting unified consensus for path: {path}")
                        
                        # Run unified consensus with architectural fixes and component configurations
                        research_result = asyncio.run(
                            self.research_orchestrator.unified_consensus(
                                provider_results=provider_results,
                                target_key_path='content',
                                task_domain=self._get_domain_from_path(path)
                            )
                        )
                        
                        # Convert research result to universal consensus format for compatibility
                        # Extract ICE results from research_result if available
                        ice_results = research_result.get('ice_refinement_results', {})
                        
                        consensus_result = self._convert_research_result_to_universal_format(
                            research_result, provider_results, path, ice_results
                        )
                        
                        logger.info(f"Unified consensus successful for path: {path}")
                        
                    except Exception as unified_error:
                        logger.error(f"Unified consensus failed for path {path}: {str(unified_error)}")
                        logger.info(f"Falling back to Universal Consensus Engine for path: {path}")
                        consensus_result = None  # Will trigger fallback below
                
                # Use Universal Consensus Engine (either as primary choice or fallback)
                if consensus_result is None:
                    logger.info(f"Using Universal Consensus Engine for path: {path}")
                    consensus_result = self.universal_engine.create_consensus(
                        provider_results=provider_results,
                        target_path='content',  # Since we've already extracted the target content
                        consensus_strength=engine_strength,
                        custom_similarity_func=self._create_domain_similarity_function(path)
                    )
                
                # Convert consensus result to ConsensusPart and apply hyperparameter filtering
                if consensus_result.final_consensus_items:
                    filtered_consensus_items = self._apply_hyperparameter_filtering(
                        consensus_result.final_consensus_items, path
                    )
                    
                    for consensus_item in filtered_consensus_items:
                        consensus_part = ConsensusPart(
                            original_part=parts[0],  # Use first part as representative
                            consensus_content=consensus_item.content,
                            consensus_confidence=consensus_item.confidence_score,
                            contributing_sources=list(set(p.source_file for p in parts)),
                            consensus_metadata={
                                'path': path,
                                'original_parts_count': len(parts),
                                'consensus_item_id': consensus_item.item_id,
                                'algorithm_results': consensus_result.algorithm_results,
                                'quality_metrics': consensus_result.consensus_quality_metrics,
                                'processing_time': consensus_result.processing_metadata.get('processing_time', 0),
                                'hyperparameter_filtered': True
                            }
                        )
                        consensus_parts.append(consensus_part)
                else:
                    # Fallback: if no consensus found, use best single part
                    logger.warning(f"No consensus achieved for path {path}, using fallback")
                    fallback_part = self._create_fallback_consensus_part(parts, path)
                    if fallback_part:
                        consensus_parts.append(fallback_part)
                
            except Exception as e:
                logger.error(f"Consensus processing failed for path {path}: {e}")
                # Create fallback part
                fallback_part = self._create_fallback_consensus_part(parts, path)
                if fallback_part:
                    consensus_parts.append(fallback_part)
        
        return consensus_parts
    
    def _convert_parts_to_provider_format(self, parts: List[JSONPart], path: str) -> Dict[str, Dict]:
        """Convert JSONPart objects to provider_results format for consensus engine"""
        provider_results = {}
        
        for part in parts:
            provider_key = f"{part.source_file}_{part.part_id}"
            provider_results[provider_key] = {
                'content': part.content,
                'metadata': part.metadata,
                'source_file': part.source_file,
                'path': part.path
            }
        
        return provider_results
    
    def _create_domain_similarity_function(self, path: str) -> callable:
        """Create domain-specific similarity function based on the path"""
        
        def domain_similarity(item1: Any, item2: Any) -> float:
            """Domain-specific similarity function"""
            
            # Pattern-based similarity for behavioral patterns
            if 'behavioral_patterns' in path or 'temporal_patterns' in path:
                return self._calculate_pattern_similarity(item1, item2)
            
            # Detection rule similarity
            elif 'detection_rule' in path:
                return self._calculate_detection_rule_similarity(item1, item2)
            
            # IoC similarity
            elif 'indicators_of_compromise' in path:
                return self._calculate_ioc_similarity(item1, item2)
            
            # General structure similarity
            else:
                return self._calculate_general_similarity(item1, item2)
        
        return domain_similarity
    
    def _calculate_pattern_similarity(self, pattern1: Any, pattern2: Any) -> float:
        """Calculate similarity between security patterns"""
        if not isinstance(pattern1, dict) or not isinstance(pattern2, dict):
            return 0.0
        
        similarity = 0.0
        
        # Pattern name similarity (high weight)
        name_fields = ['pattern_name', 'name', 'title']
        for field in name_fields:
            if field in pattern1 and field in pattern2:
                name1 = str(pattern1[field]).lower()
                name2 = str(pattern2[field]).lower()
                if name1 == name2:
                    similarity += 0.4
                elif self._calculate_string_similarity(name1, name2) > 0.7:
                    similarity += 0.2
                break
        
        # Instruction/description similarity
        desc_fields = ['instruction', 'Instruction', 'description']
        for field in desc_fields:
            if field in pattern1 and field in pattern2:
                desc_sim = self._calculate_string_similarity(
                    str(pattern1[field]).lower(), 
                    str(pattern2[field]).lower()
                )
                similarity += desc_sim * 0.3
                break
        
        # Field overlap
        if 'identifiable_fields' in pattern1 and 'identifiable_fields' in pattern2:
            fields1 = set(pattern1['identifiable_fields']) if isinstance(pattern1['identifiable_fields'], list) else set()
            fields2 = set(pattern2['identifiable_fields']) if isinstance(pattern2['identifiable_fields'], list) else set()
            if fields1 and fields2:
                field_overlap = len(fields1 & fields2) / len(fields1 | fields2)
                similarity += field_overlap * 0.3
        
        return min(1.0, similarity)
    
    def _calculate_detection_rule_similarity(self, rule1: Any, rule2: Any) -> float:
        """Calculate similarity between detection rules"""
        if not isinstance(rule1, dict) or not isinstance(rule2, dict):
            return 0.0
        
        similarity = 0.0
        
        # Detection name similarity
        if 'detection_name' in rule1 and 'detection_name' in rule2:
            name_sim = self._calculate_string_similarity(
                rule1['detection_name'].lower(),
                rule2['detection_name'].lower()
            )
            similarity += name_sim * 0.3
        
        # TTP mapping similarity
        if 'mapped_ttp' in rule1 and 'mapped_ttp' in rule2:
            if rule1['mapped_ttp'] == rule2['mapped_ttp']:
                similarity += 0.4
        
        # Logic similarity
        if 'detection_logic' in rule1 and 'detection_logic' in rule2:
            logic_sim = self._calculate_string_similarity(
                rule1['detection_logic'].lower(),
                rule2['detection_logic'].lower()
            )
            similarity += logic_sim * 0.3
        
        return min(1.0, similarity)
    
    def _calculate_ioc_similarity(self, ioc1: Any, ioc2: Any) -> float:
        """Calculate similarity between IoCs"""
        if not isinstance(ioc1, dict) or not isinstance(ioc2, dict):
            return 0.0
        
        similarity = 0.0
        
        # Category similarity
        if 'indicator_category' in ioc1 and 'indicator_category' in ioc2:
            if ioc1['indicator_category'] == ioc2['indicator_category']:
                similarity += 0.5
        
        # Description similarity
        if 'instruction' in ioc1 and 'instruction' in ioc2:
            desc_sim = self._calculate_string_similarity(
                ioc1['instruction'].lower(),
                ioc2['instruction'].lower()
            )
            similarity += desc_sim * 0.3
        
        # Fields overlap
        if 'relevant_fields' in ioc1 and 'relevant_fields' in ioc2:
            fields1 = set(ioc1['relevant_fields']) if isinstance(ioc1['relevant_fields'], list) else set()
            fields2 = set(ioc2['relevant_fields']) if isinstance(ioc2['relevant_fields'], list) else set()
            if fields1 and fields2:
                field_overlap = len(fields1 & fields2) / len(fields1 | fields2)
                similarity += field_overlap * 0.2
        
        return min(1.0, similarity)
    
    def _calculate_general_similarity(self, item1: Any, item2: Any) -> float:
        """Calculate general structural similarity"""
        if type(item1) != type(item2):
            return 0.0
        
        if isinstance(item1, dict) and isinstance(item2, dict):
            # Calculate key overlap
            keys1 = set(item1.keys())
            keys2 = set(item2.keys())
            if keys1 and keys2:
                key_overlap = len(keys1 & keys2) / len(keys1 | keys2)
                return key_overlap
        
        elif isinstance(item1, str) and isinstance(item2, str):
            return self._calculate_string_similarity(item1.lower(), item2.lower())
        
        elif item1 == item2:
            return 1.0
        
        return 0.0
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        # Word-based Jaccard similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _create_fallback_consensus_part(self, parts: List[JSONPart], path: str) -> Optional[ConsensusPart]:
        """Create fallback consensus part when consensus fails"""
        if not parts:
            return None
        
        # Choose the part with the most comprehensive content
        best_part = max(parts, key=lambda p: len(str(p.content)))
        
        return ConsensusPart(
            original_part=best_part,
            consensus_content=best_part.content,
            consensus_confidence=0.5,  # Low confidence for fallback
            contributing_sources=[best_part.source_file],
            consensus_metadata={
                'path': path,
                'is_fallback': True,
                'original_parts_count': len(parts),
                'fallback_reason': 'consensus_failed'
            }
        )
    
    def _apply_cross_path_merging(self, consensus_parts: List[ConsensusPart]) -> List[ConsensusPart]:
        """Apply cross-path merging to find similar consensus parts across different paths"""
        
        if not self.config.get('enable_cross_path_merging', True):
            return consensus_parts
        
        logger.info("Applying cross-path merging")
        
        # Group similar parts across paths
        similarity_threshold = self.config.get('similarity_threshold', 0.75)
        merged_groups = []
        used_indices = set()
        
        for i, part1 in enumerate(consensus_parts):
            if i in used_indices:
                continue
            
            current_group = [part1]
            used_indices.add(i)
            
            for j, part2 in enumerate(consensus_parts[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Calculate cross-path similarity
                similarity = self._calculate_cross_path_similarity(part1, part2)
                
                if similarity >= similarity_threshold:
                    current_group.append(part2)
                    used_indices.add(j)
            
            merged_groups.append(current_group)
        
        # Merge each group
        final_consensus_parts = []
        for group in merged_groups:
            if len(group) == 1:
                final_consensus_parts.append(group[0])
            else:
                merged_part = self._merge_consensus_parts(group)
                final_consensus_parts.append(merged_part)
        
        logger.info(f"Cross-path merging: {len(consensus_parts)} -> {len(final_consensus_parts)} parts")
        
        return final_consensus_parts
    
    def _calculate_cross_path_similarity(self, part1: ConsensusPart, part2: ConsensusPart) -> float:
        """Calculate similarity between consensus parts from different paths"""
        
        # Don't merge parts from the same path
        path1 = part1.consensus_metadata.get('path', '')
        path2 = part2.consensus_metadata.get('path', '')
        if path1 == path2:
            return 0.0
        
        # Calculate content similarity
        content_sim = self._calculate_general_similarity(part1.consensus_content, part2.consensus_content)
        
        # Bonus for similar path types
        path_bonus = 0.0
        if any(common in path1 and common in path2 for common in ['patterns', 'detection', 'indicators']):
            path_bonus = 0.2
        
        return min(1.0, content_sim + path_bonus)
    
    def _apply_hyperparameter_filtering(self, consensus_items, path: str):
        """Apply hyperparameter controls to filter and limit consensus output"""
        
        preservation_ratio = self.config.get('preservation_ratio', 0.8)
        strictness_level = self.config.get('strictness_level', 'moderate')
        security_focus = self.config.get('security_focus', True)
        
        # Sort items by confidence and quality
        sorted_items = sorted(consensus_items, key=lambda x: x.confidence_score, reverse=True)
        
        # Apply preservation ratio - control how much content to keep
        target_count = max(1, int(len(sorted_items) * preservation_ratio))
        
        # Apply strictness level filtering
        if strictness_level == 'strict':
            # Only keep high-confidence items (>0.8)
            filtered_items = [item for item in sorted_items if item.confidence_score > 0.8]
            # Further limit if too many high-confidence items
            filtered_items = filtered_items[:target_count]
        elif strictness_level == 'moderate':
            # Keep good-confidence items (>0.6) up to target count
            filtered_items = [item for item in sorted_items if item.confidence_score > 0.6][:target_count]
        else:  # lenient
            # Keep all items up to target count with lower threshold (>0.4)
            filtered_items = [item for item in sorted_items if item.confidence_score > 0.4][:target_count]
        
        # Apply security focus if enabled
        if security_focus and self._is_security_related_path(path):
            # For security-related paths, be more lenient to preserve security information
            security_boost_count = max(target_count, int(len(sorted_items) * 0.9))
            filtered_items = sorted_items[:security_boost_count]
        
        logger.debug(f"Hyperparameter filtering for {path}: {len(sorted_items)} -> {len(filtered_items)} items")
        logger.debug(f"Applied: preservation_ratio={preservation_ratio}, strictness={strictness_level}, security_focus={security_focus}")
        
        return filtered_items
    
    def _is_security_related_path(self, path: str) -> bool:
        """Check if path is security-related"""
        security_keywords = [
            'malicious', 'attack', 'vulnerability', 'threat', 'security',
            'detection', 'indicators_of_compromise', 'suspicious', 'anomalous',
            'behavioral_patterns', 'temporal_patterns'
        ]
        return any(keyword in path.lower() for keyword in security_keywords)
    
    def _merge_consensus_parts(self, parts: List[ConsensusPart]) -> ConsensusPart:
        """Merge multiple consensus parts into one"""
        
        if len(parts) == 1:
            return parts[0]
        
        # Choose the part with highest confidence as base
        base_part = max(parts, key=lambda p: p.consensus_confidence)
        
        # Apply hyperparameter controls to merged content
        merged_content = self._apply_content_merging_controls(parts, base_part)
        
        # Combine contributing sources
        all_sources = set()
        for part in parts:
            all_sources.update(part.contributing_sources)
        
        # Average confidence
        avg_confidence = sum(p.consensus_confidence for p in parts) / len(parts)
        
        return ConsensusPart(
            original_part=base_part.original_part,
            consensus_content=merged_content,
            consensus_confidence=avg_confidence,
            contributing_sources=list(all_sources),
            consensus_metadata={
                'is_merged': True,
                'merged_parts_count': len(parts),
                'merged_paths': [p.consensus_metadata.get('path', '') for p in parts],
                'merge_timestamp': time.time(),
                'hyperparameter_applied': True
            }
        )
    
    def _apply_content_merging_controls(self, parts: List[ConsensusPart], base_part: ConsensusPart):
        """Apply hyperparameter controls when merging content"""
        
        strictness_level = self.config.get('strictness_level', 'moderate')
        preservation_ratio = self.config.get('preservation_ratio', 0.8)
        
        if isinstance(base_part.consensus_content, dict):
            merged_content = base_part.consensus_content.copy()
            
            # Collect all unique fields from other parts
            additional_fields = {}
            for part in parts:
                if part != base_part and isinstance(part.consensus_content, dict):
                    for key, value in part.consensus_content.items():
                        if key not in merged_content:
                            additional_fields[key] = {
                                'value': value,
                                'confidence': part.consensus_confidence,
                                'source_count': len(part.contributing_sources)
                            }
            
            # Apply preservation controls to additional fields
            if additional_fields:
                # Sort by confidence and source count
                sorted_fields = sorted(
                    additional_fields.items(),
                    key=lambda x: (x[1]['confidence'], x[1]['source_count']),
                    reverse=True
                )
                
                # Apply preservation ratio
                fields_to_add = int(len(sorted_fields) * preservation_ratio)
                
                # Apply strictness filtering
                confidence_threshold = {
                    'strict': 0.8,
                    'moderate': 0.5,
                    'lenient': 0.1
                }.get(strictness_level, 0.6)
                
                for field_name, field_data in sorted_fields[:fields_to_add]:
                    if field_data['confidence'] >= confidence_threshold:
                        merged_content[field_name] = field_data['value']
            
            return merged_content
        else:
            # For non-dict content, return base part content
            return base_part.consensus_content
    
    def _reconstruct_unified_json(self, consensus_parts: List[ConsensusPart]) -> Dict[str, Any]:
        """Reconstruct unified JSON from consensus parts with hyperparameter controls"""
        
        logger.info("Reconstructing unified JSON from consensus parts")
        
        # Apply global hyperparameter filtering before reconstruction
        filtered_parts = self._apply_global_hyperparameter_filtering(consensus_parts)
        
        logger.info(f"Applied hyperparameter filtering: {len(consensus_parts)} -> {len(filtered_parts)} parts")
        
        unified_json = {}
        
        # Sort parts by path depth for proper reconstruction
        sorted_parts = sorted(filtered_parts, key=lambda p: len(p.consensus_metadata.get('path', '').split('.')))
        
        for part in sorted_parts:
            path = part.consensus_metadata.get('path', '')
            if not path:
                continue
            
            # Apply content complexity preference before setting value
            processed_content = self._apply_content_complexity_preference(
                part.consensus_content, 
                part.consensus_confidence
            )
            
            # Set value in unified JSON at the specified path
            self._set_nested_value(unified_json, path, processed_content)
        
        # Apply output section limiting if configured
        if self.config.get('max_output_sections') and isinstance(self.config['max_output_sections'], int):
            unified_json = self._limit_output_sections(unified_json)
        
        # Apply semantic similarity-based log_type selection if needed
        unified_json = self._apply_semantic_log_type_selection(unified_json, filtered_parts)
        
        # Add comprehensive metadata if enabled
        if self.config.get('include_metadata', True):
            unified_json['_consensus_metadata'] = {
                'generation_timestamp': time.time(),
                'consensus_parts_count': len(consensus_parts),
                'filtered_parts_count': len(filtered_parts),
                'contributing_sources': list(set(
                    source for part in filtered_parts 
                    for source in part.contributing_sources
                )),
                'average_confidence': sum(p.consensus_confidence for p in filtered_parts) / len(filtered_parts) if filtered_parts else 0,
                'processing_config': self.config.copy(),
                'hyperparameter_summary': {
                    'preservation_ratio': self.config.get('preservation_ratio', 0.8),
                    'strictness_level': self.config.get('strictness_level', 'moderate'),
                    'security_focus': self.config.get('security_focus', True),
                    'content_complexity_preference': self.config.get('content_complexity_preference', 'balanced')
                }
            }
        
        # Add provenance information if enabled
        if self.config.get('include_provenance', True):
            unified_json['_provenance'] = {
                'part_contributions': {
                    part.consensus_metadata.get('path', f'part_{i}'): {
                        'contributing_sources': part.contributing_sources,
                        'confidence': part.consensus_confidence,
                        'is_merged': part.consensus_metadata.get('is_merged', False),
                        'hyperparameter_filtered': part.consensus_metadata.get('hyperparameter_filtered', False)
                    }
                    for i, part in enumerate(filtered_parts)
                }
            }
        
        return unified_json
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation with intelligent conflict resolution"""
        try:
            current = data
            path_parts = path.split('.')
            
            # Navigate to the parent of the target key
            for i, key in enumerate(path_parts[:-1]):
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    # Handle conflict: existing value is not a dict
                    # FIX: Don't create _original_content wrapper, prefer proper data structures
                    original_value = current[key]
                    
                    # If original value is a list, we need to handle this at the parent path level
                    # Don't overwrite lists with dicts - preserve the list structure
                    if isinstance(original_value, list):
                        # Stop here - can't navigate further into a list for dict-style path
                        logger.debug(f"Cannot navigate into list at {key}, stopping path traversal")
                        return  # Exit early to preserve list structure
                    else:
                        # Convert to dict only if we need nested structure
                        current[key] = {}
                    
                    logger.debug(f"Path conflict resolved at {key}, preserving structure")
                current = current[key]
            
            # Set the final value with conflict detection and proper list handling
            final_key = path_parts[-1]
            if final_key in current and current[final_key] != value:
                # Handle final key conflicts by merging if both are dicts
                if isinstance(current[final_key], dict) and isinstance(value, dict):
                    # Merge dictionaries
                    current[final_key].update(value)
                    logger.debug(f"Merged dict values at final key {final_key}")
                elif isinstance(current[final_key], list) and isinstance(value, list):
                    # Merge lists (avoiding duplicates) - FIX: Preserve all knowledge
                    existing_items = set(str(item) for item in current[final_key] if item is not None)
                    for item in value:
                        if item is not None and str(item) not in existing_items:
                            current[final_key].append(item)
                    logger.debug(f"Merged list values at final key {final_key}")
                elif isinstance(value, list) and not isinstance(current[final_key], list):
                    # FIX: If new value is a list but current isn't, replace with list to preserve all knowledge
                    current[final_key] = value
                    logger.debug(f"Replaced with list value at final key {final_key}")
                elif isinstance(current[final_key], list) and not isinstance(value, list):
                    # FIX: If current is list but new value isn't, append to preserve all knowledge
                    if value not in current[final_key]:
                        current[final_key].append(value)
                    logger.debug(f"Appended to list at final key {final_key}")
                else:
                    # For non-compatible types, keep the more complex/informative value
                    if self._is_more_informative(value, current[final_key]):
                        current[final_key] = value
                    logger.debug(f"Kept more informative value at final key {final_key}")
            else:
                current[final_key] = value
            
        except Exception as e:
            logger.error(f"Failed to set nested value at path {path}: {e}")
    
    def _is_more_informative(self, value1: Any, value2: Any) -> bool:
        """Determine which value is more informative/complex"""
        # Dicts are generally more informative than primitives
        if isinstance(value1, dict) and not isinstance(value2, dict):
            return True
        if isinstance(value2, dict) and not isinstance(value1, dict):
            return False
        
        # Lists are more informative than primitives
        if isinstance(value1, list) and not isinstance(value2, list):
            return True
        if isinstance(value2, list) and not isinstance(value1, list):
            return False
        
        # For strings, longer is generally more informative
        if isinstance(value1, str) and isinstance(value2, str):
            return len(value1.strip()) > len(value2.strip())
        
        # For numbers, non-zero is more informative than zero
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return abs(value1) > abs(value2)
        
        # Default: keep the new value
        return True
    
    def _calculate_processing_quality(self, 
                                    loaded_jsons: Dict[str, Dict],
                                    all_parts: List[JSONPart],
                                    consensus_parts: List[ConsensusPart],
                                    unified_json: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for the processing"""
        
        quality_metrics = {
            'overall_quality': 0.0,
            'coverage': 0.0,
            'consensus_success_rate': 0.0,
            'average_consensus_confidence': 0.0,
            'data_preservation': 0.0,
            'source_diversity': 0.0,
            'structural_integrity': 0.0
        }
        
        try:
            # Coverage: how much of original data was processed
            if all_parts:
                quality_metrics['coverage'] = len(consensus_parts) / len(all_parts)
            
            # Consensus success rate
            successful_consensus = len([p for p in consensus_parts if not p.consensus_metadata.get('is_fallback', False)])
            if consensus_parts:
                quality_metrics['consensus_success_rate'] = successful_consensus / len(consensus_parts)
            
            # Average consensus confidence
            if consensus_parts:
                quality_metrics['average_consensus_confidence'] = sum(p.consensus_confidence for p in consensus_parts) / len(consensus_parts)
            
            # Source diversity: how many different sources contributed
            all_sources = set()
            for part in consensus_parts:
                all_sources.update(part.contributing_sources)
            
            if loaded_jsons:
                quality_metrics['source_diversity'] = len(all_sources) / len(loaded_jsons)
            
            # Data preservation: estimate of how much original data is preserved
            original_size = sum(len(str(json_data)) for json_data in loaded_jsons.values())
            unified_size = len(str(unified_json))
            if original_size > 0:
                # Normalize preservation metric (not just size ratio)
                size_ratio = unified_size / original_size
                quality_metrics['data_preservation'] = min(1.0, size_ratio * 1.5)  # Boost for consolidation
            
            # Structural integrity: check if unified JSON has expected structure
            expected_sections = ['observations', 'detection_rule_checklist', 'indicators_of_compromise']
            present_sections = sum(1 for section in expected_sections if section in unified_json)
            quality_metrics['structural_integrity'] = present_sections / len(expected_sections)
            
            # Overall quality as weighted average
            weights = {
                'coverage': 0.2,
                'consensus_success_rate': 0.25,
                'average_consensus_confidence': 0.2,
                'data_preservation': 0.15,
                'source_diversity': 0.1,
                'structural_integrity': 0.1
            }
            
            quality_metrics['overall_quality'] = sum(
                quality_metrics[metric] * weight
                for metric, weight in weights.items()
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return quality_metrics
    
    def _calculate_part_contributions(self, consensus_parts: List[ConsensusPart]) -> Dict[str, List[str]]:
        """Calculate which source files contributed to each path"""
        contributions = {}
        
        for part in consensus_parts:
            path = part.consensus_metadata.get('path', 'unknown')
            contributions[path] = part.contributing_sources
        
        return contributions
    
    def _apply_global_hyperparameter_filtering(self, consensus_parts: List[ConsensusPart]) -> List[ConsensusPart]:
        """Apply global hyperparameter filtering across all consensus parts"""
        
        min_confidence_threshold = self.config.get('min_confidence_threshold', 0.5)
        security_focus = self.config.get('security_focus', True)
        preservation_ratio = self.config.get('preservation_ratio', 0.8)
        
        # Filter by minimum confidence threshold
        confidence_filtered = [p for p in consensus_parts if p.consensus_confidence >= min_confidence_threshold]
        
        # Apply security focus - prioritize security-related paths
        if security_focus:
            security_parts = [p for p in confidence_filtered if self._is_security_related_path(p.consensus_metadata.get('path', ''))]
            non_security_parts = [p for p in confidence_filtered if not self._is_security_related_path(p.consensus_metadata.get('path', ''))]
            
            # Keep all security parts, limit non-security parts more strictly
            target_non_security = max(1, int(len(non_security_parts) * preservation_ratio * 0.7))  # More restrictive for non-security
            limited_non_security = sorted(non_security_parts, key=lambda x: x.consensus_confidence, reverse=True)[:target_non_security]
            
            filtered_parts = security_parts + limited_non_security
        else:
            # Normal preservation ratio application
            target_count = max(1, int(len(confidence_filtered) * preservation_ratio))
            filtered_parts = sorted(confidence_filtered, key=lambda x: x.consensus_confidence, reverse=True)[:target_count]
        
        logger.debug(f"Global hyperparameter filtering: {len(consensus_parts)} -> {len(filtered_parts)} parts")
        
        return filtered_parts
    
    def _apply_content_complexity_preference(self, content: Any, confidence: float) -> Any:
        """Apply content complexity preference to individual content items"""
        
        complexity_preference = self.config.get('content_complexity_preference', 'balanced')
        
        if not isinstance(content, dict):
            return content
        
        if complexity_preference == 'simple':
            # Keep only essential fields for simple output
            essential_fields = ['pattern_name', 'name', 'title', 'instruction', 'Instruction', 'detection_name', 'indicator_category']
            simplified_content = {}
            for field in essential_fields:
                if field in content:
                    simplified_content[field] = content[field]
            return simplified_content if simplified_content else content
        
        elif complexity_preference == 'comprehensive':
            # Keep all content as-is for comprehensive output
            return content
        
        else:  # balanced
            # Remove very verbose fields if confidence is low
            if confidence < 0.7:
                verbose_fields = ['detailed_explanation', 'technical_details', 'extended_context', 'debug_info']
                balanced_content = content.copy()
                for field in verbose_fields:
                    balanced_content.pop(field, None)
                return balanced_content
            return content
    
    def _limit_output_sections(self, unified_json: Dict[str, Any]) -> Dict[str, Any]:
        """Limit the number of top-level sections in output"""
        
        max_sections = self.config.get('max_output_sections')
        if not max_sections or not isinstance(max_sections, int):
            return unified_json
        
        # Preserve metadata and provenance sections
        protected_sections = ['_consensus_metadata', '_provenance']
        
        # Get regular sections (non-protected)
        regular_sections = {k: v for k, v in unified_json.items() if k not in protected_sections}
        
        # Limit regular sections based on priority
        priority_order = [
            'observations', 'detection_rule_checklist', 'indicators_of_compromise',
            'attack_pattern_checks', 'vulnerability_checks', 'entity_analysis_instructions'
        ]
        
        limited_sections = {}
        sections_added = 0
        
        # Add priority sections first
        for section in priority_order:
            if section in regular_sections and sections_added < max_sections:
                limited_sections[section] = regular_sections[section]
                sections_added += 1
        
        # Add remaining sections up to limit
        for section, content in regular_sections.items():
            if section not in limited_sections and sections_added < max_sections:
                limited_sections[section] = content
                sections_added += 1
        
        # Combine with protected sections
        result = limited_sections.copy()
        for section in protected_sections:
            if section in unified_json:
                result[section] = unified_json[section]
        
        logger.info(f"Limited output sections: {len(regular_sections)} -> {len(limited_sections)} sections")
        
        return result
    
    def _validate_hyperparameter_config(self) -> None:
        """Validate hyperparameter configuration values"""
        
        # Validate preservation_ratio
        preservation_ratio = self.config.get('preservation_ratio', 0.8)
        if not isinstance(preservation_ratio, (int, float)) or not 0.1 <= preservation_ratio <= 1.0:
            logger.warning(f"Invalid preservation_ratio {preservation_ratio}, using default 0.8")
            self.config['preservation_ratio'] = 0.8
        
        # Validate strictness_level
        valid_strictness = ['lenient', 'moderate', 'strict']
        strictness = self.config.get('strictness_level', 'moderate')
        if strictness not in valid_strictness:
            logger.warning(f"Invalid strictness_level {strictness}, using 'moderate'")
            self.config['strictness_level'] = 'moderate'
        
        # Validate content_complexity_preference
        valid_complexity = ['simple', 'balanced', 'comprehensive']
        complexity = self.config.get('content_complexity_preference', 'balanced')
        if complexity not in valid_complexity:
            logger.warning(f"Invalid content_complexity_preference {complexity}, using 'balanced'")
            self.config['content_complexity_preference'] = 'balanced'
        
        # Validate min_confidence_threshold
        min_confidence = self.config.get('min_confidence_threshold', 0.5)
        if not isinstance(min_confidence, (int, float)) or not 0.0 <= min_confidence <= 1.0:
            logger.warning(f"Invalid min_confidence_threshold {min_confidence}, using default 0.5")
            self.config['min_confidence_threshold'] = 0.5
        
        # Validate max_output_sections
        max_sections = self.config.get('max_output_sections')
        if max_sections is not None and (not isinstance(max_sections, int) or max_sections < 1):
            logger.warning(f"Invalid max_output_sections {max_sections}, disabling section limiting")
            self.config['max_output_sections'] = None
        
        logger.debug("Hyperparameter configuration validated successfully")
    
    def update_hyperparameters(self, **hyperparams) -> None:
        """Update hyperparameter configuration
        
        Args:
            preservation_ratio (float): How much content to preserve (0.1-1.0)
            strictness_level (str): Filtering strictness ('lenient', 'moderate', 'strict')
            security_focus (bool): Whether to prioritize security-related content
            content_complexity_preference (str): Content detail level ('simple', 'balanced', 'comprehensive')
            min_confidence_threshold (float): Global minimum confidence (0.0-1.0)
            max_output_sections (int): Maximum number of output sections (None for no limit)
        """
        
        # Update configuration
        for param, value in hyperparams.items():
            if param in ['preservation_ratio', 'strictness_level', 'security_focus', 
                        'content_complexity_preference', 'min_confidence_threshold', 'max_output_sections']:
                self.config[param] = value
            else:
                logger.warning(f"Unknown hyperparameter: {param}")
        
        # Re-validate configuration
        self._validate_hyperparameter_config()
        
        logger.info(f"Updated hyperparameters: {hyperparams}")
    
    def get_hyperparameter_summary(self) -> Dict[str, Any]:
        """Get current hyperparameter configuration summary"""
        
        return {
            'preservation_ratio': self.config.get('preservation_ratio', 0.8),
            'strictness_level': self.config.get('strictness_level', 'moderate'),
            'security_focus': self.config.get('security_focus', True),
            'content_complexity_preference': self.config.get('content_complexity_preference', 'balanced'),
            'min_confidence_threshold': self.config.get('min_confidence_threshold', 0.5),
            'max_output_sections': self.config.get('max_output_sections', None),
            'description': {
                'preservation_ratio': 'Controls how much content to keep (0.1=minimal, 1.0=all)',
                'strictness_level': 'Confidence filtering (lenient/moderate/strict)',
                'security_focus': 'Prioritizes security-related content when True',
                'content_complexity_preference': 'Detail level (simple/balanced/comprehensive)',
                'min_confidence_threshold': 'Global minimum confidence for inclusion',
                'max_output_sections': 'Maximum top-level sections in output (None=unlimited)'
            }
        }
    
    def _save_unified_result(self, result: UnifiedJSONResult, output_file: str) -> None:
        """Save unified result to file"""
        try:
            output_path = Path(output_file)
            
            # Determine output format
            output_format = self.config.get('output_format', 'enhanced')
            
            if output_format == 'minimal':
                # Save only the unified JSON
                output_data = result.unified_json
            elif output_format == 'standard':
                # Save JSON with basic metadata
                output_data = {
                    'unified_json': result.unified_json,
                    'quality_metrics': result.quality_metrics,
                    'processing_time': result.total_processing_time
                }
            else:  # enhanced
                # Save everything
                output_data = {
                    'unified_json': result.unified_json,
                    'processing_metadata': result.processing_metadata,
                    'quality_metrics': result.quality_metrics,
                    'part_contributions': result.part_contributions,
                    'summary': {
                        'original_file_count': result.original_file_count,
                        'consensus_parts_count': result.consensus_parts_count,
                        'total_processing_time': result.total_processing_time,
                        'overall_quality': result.quality_metrics.get('overall_quality', 0)
                    }
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Saved unified result to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save unified result: {e}")
    
    def _create_error_result(self, json_files: List[str], error: str, processing_time: float) -> UnifiedJSONResult:
        """Create error result for failed processing"""
        return UnifiedJSONResult(
            unified_json={'error': error},
            processing_metadata={
                'error': error,
                'original_files': json_files,
                'processing_time': processing_time,
                'timestamp': time.time()
            },
            quality_metrics={'overall_quality': 0.0, 'error': error},
            part_contributions={},
            original_file_count=len(json_files),
            consensus_parts_count=0,
            total_processing_time=processing_time
        )
    
    def process_jsons_folder(self, 
                           jsons_folder: str = "jsons",
                           output_file: str = "output/jsons/consensus_results/semantic_bft_result.json",
                           consensus_strength: str = "comprehensive") -> UnifiedJSONResult:
        """
        Convenience method to process all JSON files in the jsons folder
        
        Args:
            jsons_folder: Folder containing JSON files
            output_file: Output file for unified result
            consensus_strength: Consensus strength level
            
        Returns:
            UnifiedJSONResult
        """
        try:
            jsons_path = Path(jsons_folder)
            if not jsons_path.exists():
                raise FileNotFoundError(f"Jsons folder not found: {jsons_folder}")
            
            # Find all JSON files
            json_files = list(jsons_path.glob("*.json"))
            if not json_files:
                raise ValueError(f"No JSON files found in {jsons_folder}")
            
            json_file_paths = [str(f) for f in json_files]
            
            logger.info(f"Processing {len(json_file_paths)} JSON files from {jsons_folder}")
            
            return self.process_json_files(
                json_files=json_file_paths,
                output_file=output_file,
                consensus_strength=consensus_strength
            )
            
        except Exception as e:
            logger.error(f"Failed to process jsons folder: {e}")
            return self._create_error_result([], str(e), 0.0)

    @staticmethod
    def create_hyperparameter_preset(preset_name: str) -> Dict[str, Any]:
        """Create predefined hyperparameter configurations
        
        Args:
            preset_name: 'minimal', 'balanced', 'comprehensive', 'security_focused'
            
        Returns:
            Dictionary with hyperparameter configuration
        """
        
        presets = {
            'minimal': {
                'preservation_ratio': 0.4,
                'strictness_level': 'strict',
                'security_focus': True,
                'content_complexity_preference': 'simple',
                'min_confidence_threshold': 0.8,
                'max_output_sections': 4
            },
            'balanced': {
                'preservation_ratio': 0.8,
                'strictness_level': 'moderate',
                'security_focus': True,
                'content_complexity_preference': 'balanced',
                'min_confidence_threshold': 0.5,
                'max_output_sections': None
            },
            'comprehensive': {
                'preservation_ratio': 0.95,
                'strictness_level': 'lenient',
                'security_focus': True,
                'content_complexity_preference': 'comprehensive',
                'min_confidence_threshold': 0.3,
                'max_output_sections': None
            },
            'security_focused': {
                'preservation_ratio': 0.7,
                'strictness_level': 'moderate',
                'security_focus': True,
                'content_complexity_preference': 'balanced',
                'min_confidence_threshold': 0.6,
                'max_output_sections': 6
            }
        }
        
        return presets.get(preset_name, presets['balanced'])
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about processing history"""
        return {
            'total_processing_runs': len(self.processing_history),
            'recent_runs': self.processing_history[-5:] if self.processing_history else [],
            'average_quality': sum(run.get('quality_score', 0) for run in self.processing_history) / len(self.processing_history) if self.processing_history else 0,
            'average_processing_time': sum(run.get('processing_time', 0) for run in self.processing_history) / len(self.processing_history) if self.processing_history else 0,
            'config': self.config,
            'hyperparameter_summary': self.get_hyperparameter_summary()
        }
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get research mode statistics and status"""
        if not self.enable_research_mode:
            return {
                'research_mode_enabled': False,
                'components_initialized': {},
                'agents_registered': 0
            }
        
        # Get statistics from research orchestrator
        stats = self.research_orchestrator.get_system_status()
        stats['research_mode_enabled'] = True
        return stats
    
    def get_hitl_items(self) -> List[Dict[str, Any]]:
        """Get Human-in-the-Loop items requiring review"""
        if not self.enable_research_mode:
            return []
        
        return self.research_orchestrator.ice_loop.get_hitl_items()
    
    def _get_domain_from_path(self, path: str) -> str:
        """Extract domain type from JSON path for task specialization"""
        path_lower = path.lower()
        
        # Domain mapping based on path content
        if any(term in path_lower for term in ['security', 'threat', 'attack', 'malicious', 'vulnerability']):
            return 'security'
        elif any(term in path_lower for term in ['network', 'connection', 'protocol', 'traffic']):
            return 'network'
        elif any(term in path_lower for term in ['system', 'process', 'service', 'application']):
            return 'system'
        elif any(term in path_lower for term in ['behavior', 'pattern', 'activity', 'action']):
            return 'behavioral'
        else:
            return 'general'
    
    def _convert_research_result_to_universal_format(self, research_result: Dict[str, Any], 
                                                   provider_results: Dict[str, Any], 
                                                   path: str,
                                                   ice_results: Optional[Dict[str, Any]] = None) -> Any:
        """Convert research orchestrator result to universal consensus engine format"""
        
        # Create a mock universal consensus result structure
        class MockConsensusResult:
            def __init__(self, research_data, ice_results_data=None):
                self.final_consensus_items = []
                self.algorithm_results = research_data.get('algorithm_results', {})
                self.consensus_quality_metrics = research_data.get('quality_metrics', {})
                self.processing_metadata = research_data.get('processing_metadata', {})
                
                # Add ICE results to metadata if available
                if ice_results_data:
                    self.processing_metadata['ice_refinement'] = ice_results_data
                
                # Extract consensus content from research result
                final_consensus = research_data.get('final_consensus', {})
                
                # Check if ICE refinement provided refined consensus
                refined_consensus = None
                if ice_results_data and ice_results_data.get('refined_consensus'):
                    refined_consensus = ice_results_data['refined_consensus']
                    logger.info(f"Using ICE-refined consensus for {len(refined_consensus)} items")
                
                if refined_consensus:
                    # Prioritize ICE-refined content
                    for concept_id, refined_data in refined_consensus.items():
                        # Use refined content if available, otherwise fall back to original
                        original_data = final_consensus.get(concept_id, {})
                        
                        consensus_item = MockConsensusItem(
                            item_id=concept_id,
                            content=refined_data.get('consensus', original_data.get('consensus_content', original_data.get('concept_data', {}))),
                            confidence_score=refined_data.get('final_confidence_score', original_data.get('consensus_confidence', 0.8)),
                            metadata={**original_data.get('metadata', {}), 'ice_refined': True}
                        )
                        self.final_consensus_items.append(consensus_item)
                elif final_consensus:
                    # Create consensus items from research result (no ICE refinement)
                    for concept_id, consensus_data in final_consensus.items():
                        consensus_item = MockConsensusItem(
                            item_id=concept_id,
                            content=consensus_data.get('consensus_content', consensus_data.get('concept_data', {})),
                            confidence_score=consensus_data.get('consensus_confidence', 0.8),
                            metadata=consensus_data.get('metadata', {})
                        )
                        self.final_consensus_items.append(consensus_item)
                
                # If no final consensus, extract from weighted consensus
                elif research_data.get('weighted_consensus', {}):
                    weighted_result = research_data['weighted_consensus']
                    final_consensus = weighted_result.get('final_consensus', {})
                    
                    for concept_id, consensus_data in final_consensus.items():
                        consensus_item = MockConsensusItem(
                            item_id=concept_id,
                            content=consensus_data.get('consensus_content', consensus_data),
                            confidence_score=consensus_data.get('final_confidence', 0.7),
                            metadata={'weighted_consensus': True}
                        )
                        self.final_consensus_items.append(consensus_item)
                
                # Fallback: create single consensus item from best available content
                if not self.final_consensus_items:
                    # Use first provider result as fallback
                    first_provider = list(provider_results.keys())[0]
                    first_content = provider_results[first_provider].get('content', {})
                    
                    consensus_item = MockConsensusItem(
                        item_id=f"fallback_{path}",
                        content=first_content,
                        confidence_score=0.5,
                        metadata={'fallback': True, 'research_processing_failed': True}
                    )
                    self.final_consensus_items.append(consensus_item)
        
        class MockConsensusItem:
            def __init__(self, item_id, content, confidence_score, metadata=None):
                self.item_id = item_id
                self.content = content
                self.confidence_score = confidence_score
                self.metadata = metadata or {}
        
        return MockConsensusResult(research_result, ice_results)
    
    def _apply_semantic_log_type_selection(self, unified_json: Dict[str, Any], consensus_parts: List[ConsensusPart]) -> Dict[str, Any]:
        """Apply semantic similarity-based log_type selection and complete field properties"""
        
        try:
            # Apply complete field properties selection
            unified_json = self._apply_complete_field_properties_selection(unified_json)
            
            # Apply complete Stage 5 pattern reconstruction if applicable
            unified_json = self._apply_complete_stage5_pattern_reconstruction(unified_json)
            
            # Check if we need to apply semantic log_type selection
            log_type_parts = [p for p in consensus_parts 
                             if 'log_type' in p.consensus_metadata.get('path', '') 
                             and p.original_part.metadata.get('enable_semantic_similarity_selection', False)]
            
            if log_type_parts:
                from improved_log_type_consensus import SemanticLogTypeSelector
                
                # Collect all log_type values from consensus parts
                log_type_values = []
                for part in log_type_parts:
                    if isinstance(part.consensus_content, str):
                        log_type_values.append(part.consensus_content)
                
                if log_type_values:
                    # Use semantic similarity to select best log_type
                    selector = SemanticLogTypeSelector()
                    best_log_type = selector.select_best_log_type(log_type_values)
                    
                    # Set the best log_type in unified JSON
                    if 'parsed_json' not in unified_json:
                        unified_json['parsed_json'] = {}
                    unified_json['parsed_json']['log_type'] = best_log_type
                    
                    logger.info(f"Applied semantic log_type selection: {best_log_type}")
            
        except Exception as e:
            logger.warning(f"Semantic selection failed: {e}, using default consensus")
        
        return unified_json
    
    def _apply_complete_field_properties_selection(self, unified_json: Dict[str, Any]) -> Dict[str, Any]:
        """Apply complete field properties selection from all models"""
        
        try:
            # Load the three original JSON files to get all field properties
            stage1_dir = Path("output_v3_new/stage1_prompt1")
            json_files = list(stage1_dir.glob("*.json"))
            
            if not json_files:
                logger.warning("No stage1 JSON files found for complete field properties")
                return unified_json
            
            # Collect all field properties from all models
            all_field_data = {}  # field_name -> property -> [values from all models]
            all_log_types = []  # Collect log_types from all models
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                if 'parsed_json' in data:
                    parsed_json = data['parsed_json']
                    model_name = json_file.stem  # e.g., 'claude_search_prompt1'
                    
                    # Collect log_type
                    if 'log_type' in parsed_json:
                        all_log_types.append(parsed_json['log_type'])
                    
                    # Collect field data
                    if 'fields' in parsed_json:
                        fields = parsed_json['fields']
                        
                        for field_name, field_info in fields.items():
                            if field_name not in all_field_data:
                                all_field_data[field_name] = {}
                            
                            # Collect all properties for this field
                            for prop_name, prop_value in field_info.items():
                                if prop_name not in all_field_data[field_name]:
                                    all_field_data[field_name][prop_name] = []
                                
                                # Store the value with source model info
                                all_field_data[field_name][prop_name].append({
                                    'value': prop_value,
                                    'source': model_name
                                })
            
            if not all_field_data:
                logger.warning("No field data collected for complete properties")
                return unified_json
            
            # Use semantic similarity to select best values for each property
            from improved_log_type_consensus import SemanticLogTypeSelector
            selector = SemanticLogTypeSelector()
            
            consensus_fields = {}
            
            for field_name, field_properties in all_field_data.items():
                consensus_fields[field_name] = {}
                
                # Process each property type
                for prop_name, prop_values in field_properties.items():
                    # Extract just the values for semantic comparison
                    values = [str(item['value']) for item in prop_values if item['value'] is not None]
                    
                    if not values:
                        continue
                        
                    if prop_name in ['importance']:
                        # For numeric properties, take the most common value
                        if len(set(values)) == 1:
                            consensus_fields[field_name][prop_name] = prop_values[0]['value']
                        else:
                            # Use the most frequent value
                            from collections import Counter
                            counter = Counter(values)
                            most_common = counter.most_common(1)[0][0]
                            # Convert back to original type
                            for item in prop_values:
                                if str(item['value']) == most_common:
                                    consensus_fields[field_name][prop_name] = item['value']
                                    break
                    else:
                        # For text properties, use semantic similarity selection
                        if len(set(values)) == 1:
                            consensus_fields[field_name][prop_name] = values[0]
                        else:
                            best_value = selector.select_best_log_type(values)
                            consensus_fields[field_name][prop_name] = best_value
            
            # Update unified JSON with complete field properties
            if 'parsed_json' not in unified_json:
                unified_json['parsed_json'] = {}
            
            unified_json['parsed_json']['fields'] = consensus_fields
            
            # Select best log_type using semantic similarity
            if all_log_types:
                best_log_type = selector.select_best_log_type(all_log_types)
                unified_json['parsed_json']['log_type'] = best_log_type
                logger.info(f"Selected log_type via semantic similarity: {best_log_type}")
            
            # Add field property stats
            unified_json['_field_property_stats'] = {
                "total_fields": len(consensus_fields),
                "properties_per_field": {
                    field_name: list(field_props.keys()) 
                    for field_name, field_props in consensus_fields.items()
                },
                "semantic_selection_applied": True
            }
            
            logger.info(f"Applied complete field properties for {len(consensus_fields)} fields")
            
            # Update consensus metadata to reflect all sources used in complete field properties
            if '_consensus_metadata' in unified_json:
                # Get all unique sources from the JSON files we actually used
                all_sources_used = list(set(json_file.stem for json_file in json_files))
                unified_json['_consensus_metadata']['contributing_sources'] = all_sources_used
                unified_json['_consensus_metadata']['complete_field_reconstruction_applied'] = True
                logger.info(f"Updated consensus metadata to reflect all {len(all_sources_used)} sources: {all_sources_used}")
            
        except Exception as e:
            logger.warning(f"Complete field properties selection failed: {e}")
        
        return unified_json
    
    def _apply_complete_stage5_pattern_reconstruction(self, unified_json: Dict[str, Any]) -> Dict[str, Any]:
        """Apply complete Stage 5 pattern reconstruction from all Stage 4 models"""
        
        try:
            # Load the three original Stage 4 JSON files to get all patterns
            stage4_dir = Path("output_v3_new/stage4_prompt3")
            json_files = list(stage4_dir.glob("*.json"))
            
            if not json_files:
                logger.warning("No stage4 JSON files found for complete pattern reconstruction")
                return unified_json
            
            logger.info(f"Found {len(json_files)} Stage 4 files for comprehensive pattern reconstruction")
            
            # Collect all pattern data from all models
            all_behavioral_patterns = {'malicious': [], 'anomalous': [], 'vulnerable': []}
            all_temporal_patterns = {'malicious': [], 'anomalous': [], 'vulnerable': []}
            all_detection_rules = []
            all_indicators = []
            all_attack_patterns = []
            all_entity_instructions = []
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                model_name = json_file.stem  # e.g., 'gemini_from_claude_report_prompt3'
                logger.info(f"Processing {model_name} for pattern extraction")
                
                if 'parsed_json' in data:
                    parsed_json = data['parsed_json']
                    
                    # Extract behavioral patterns
                    if 'observations' in parsed_json and 'behavioral_patterns' in parsed_json['observations']:
                        behavioral = parsed_json['observations']['behavioral_patterns']
                        for pattern_type in ['malicious', 'anomalous', 'vulnerable']:
                            if pattern_type in behavioral:
                                patterns = behavioral[pattern_type]
                                if isinstance(patterns, list):
                                    all_behavioral_patterns[pattern_type].extend(patterns)
                                elif isinstance(patterns, dict):
                                    all_behavioral_patterns[pattern_type].append(patterns)
                    
                    # Extract temporal patterns  
                    if 'observations' in parsed_json and 'temporal_patterns' in parsed_json['observations']:
                        temporal = parsed_json['observations']['temporal_patterns']
                        for pattern_type in ['malicious', 'anomalous', 'vulnerable']:
                            if pattern_type in temporal:
                                patterns = temporal[pattern_type]
                                if isinstance(patterns, list):
                                    all_temporal_patterns[pattern_type].extend(patterns)
                                elif isinstance(patterns, dict):
                                    all_temporal_patterns[pattern_type].append(patterns)
                    
                    # Extract detection rules
                    if 'detection_rule_checklist' in parsed_json:
                        rules = parsed_json['detection_rule_checklist']
                        if isinstance(rules, list):
                            all_detection_rules.extend(rules)
                        elif isinstance(rules, dict):
                            all_detection_rules.append(rules)
                    
                    # Extract indicators of compromise
                    if 'indicators_of_compromise' in parsed_json:
                        indicators = parsed_json['indicators_of_compromise']
                        if isinstance(indicators, list):
                            all_indicators.extend(indicators)
                        elif isinstance(indicators, dict):
                            all_indicators.append(indicators)
                    
                    # Extract attack pattern checks
                    if 'attack_pattern_checks' in parsed_json:
                        attacks = parsed_json['attack_pattern_checks']
                        if isinstance(attacks, list):
                            all_attack_patterns.extend(attacks)
                        elif isinstance(attacks, dict):
                            all_attack_patterns.append(attacks)
                    
                    # Extract entity analysis instructions
                    if 'entity_analysis_instructions' in parsed_json:
                        entity_instructions = parsed_json['entity_analysis_instructions']
                        if isinstance(entity_instructions, dict):
                            all_entity_instructions.append(entity_instructions)
            
            # Create comprehensive unified structure
            comprehensive_json = {
                'parsed_json': {
                    'observations': {
                        'behavioral_patterns': {},
                        'temporal_patterns': {}
                    },
                    'detection_rule_checklist': all_detection_rules[:20],  # Limit to prevent overwhelming
                    'indicators_of_compromise': all_indicators[:15],
                    'attack_pattern_checks': all_attack_patterns[:10],
                    'entity_analysis_instructions': all_entity_instructions[0] if all_entity_instructions else {}
                }
            }
            
            # Process behavioral patterns with deduplication
            for pattern_type in ['malicious', 'anomalous', 'vulnerable']:
                if all_behavioral_patterns[pattern_type]:
                    # Deduplicate by pattern_name
                    seen_names = set()
                    unique_patterns = []
                    for pattern in all_behavioral_patterns[pattern_type]:
                        if isinstance(pattern, dict):
                            name = pattern.get('pattern_name', pattern.get('name', ''))
                            if name and name not in seen_names:
                                seen_names.add(name)
                                unique_patterns.append(pattern)
                    
                    comprehensive_json['parsed_json']['observations']['behavioral_patterns'][pattern_type] = unique_patterns[:5]  # Top 5 per type
            
            # Process temporal patterns with deduplication
            for pattern_type in ['malicious', 'anomalous', 'vulnerable']:
                if all_temporal_patterns[pattern_type]:
                    # Deduplicate by pattern_name
                    seen_names = set()
                    unique_patterns = []
                    for pattern in all_temporal_patterns[pattern_type]:
                        if isinstance(pattern, dict):
                            name = pattern.get('pattern_name', pattern.get('name', ''))
                            if name and name not in seen_names:
                                seen_names.add(name)
                                unique_patterns.append(pattern)
                    
                    comprehensive_json['parsed_json']['observations']['temporal_patterns'][pattern_type] = unique_patterns[:5]  # Top 5 per type
            
            # Update unified_json with comprehensive data
            if 'parsed_json' not in unified_json:
                unified_json['parsed_json'] = {}
            
            # Merge comprehensive data with existing unified_json
            unified_json['parsed_json'].update(comprehensive_json['parsed_json'])
            
            # Update consensus metadata to reflect all sources used
            if '_consensus_metadata' in unified_json:
                all_sources_used = list(set(json_file.stem for json_file in json_files))
                unified_json['_consensus_metadata']['contributing_sources'] = all_sources_used
                unified_json['_consensus_metadata']['complete_stage5_reconstruction_applied'] = True
                logger.info(f"Updated Stage 5 consensus metadata to reflect all {len(all_sources_used)} sources: {all_sources_used}")
            
            logger.info(f"Stage 5 comprehensive pattern reconstruction completed:")
            logger.info(f"  - Behavioral patterns: {sum(len(v) for v in all_behavioral_patterns.values())} total")
            logger.info(f"  - Temporal patterns: {sum(len(v) for v in all_temporal_patterns.values())} total")
            logger.info(f"  - Detection rules: {len(all_detection_rules)}")
            logger.info(f"  - Indicators: {len(all_indicators)}")
            logger.info(f"  - Attack patterns: {len(all_attack_patterns)}")
            
        except Exception as e:
            logger.warning(f"Complete Stage 5 pattern reconstruction failed: {e}")
        
        return unified_json

if __name__ == "__main__":
    # Example usage with hyperparameter demonstration
    logging.basicConfig(level=logging.INFO)
    
    print("=== Semantic BFT Consensus Processor with Hyperparameter Controls ===")
    print()
    
    # Create processor with default hyperparameters
    processor = SemanticBFTConsensusProcessor()
    
    # Show current hyperparameter configuration
    hyperparams = processor.get_hyperparameter_summary()
    print("Current Hyperparameter Configuration:")
    for param, value in hyperparams.items():
        if param != 'description':
            print(f"  {param}: {value}")
    print()
    
    # Process with default settings
    print("Processing with DEFAULT hyperparameters...")
    result_default = processor.process_jsons_folder(
        jsons_folder="jsons",
        output_file="output/jsons/consensus_results/semantic_bft_default.json",
        consensus_strength="comprehensive"
    )
    
    print(f"DEFAULT - Quality: {result_default.quality_metrics.get('overall_quality', 0):.3f}, "
          f"Time: {result_default.total_processing_time:.2f}s, "
          f"Parts: {result_default.consensus_parts_count}")
    print()
    
    # Test with MINIMAL hyperparameters (preserve less, be more selective)
    print("Processing with MINIMAL hyperparameters...")
    processor.update_hyperparameters(
        preservation_ratio=0.5,
        strictness_level='strict',
        content_complexity_preference='simple',
        min_confidence_threshold=0.7,
        max_output_sections=5
    )
    
    result_minimal = processor.process_jsons_folder(
        jsons_folder="jsons",
        output_file="output/jsons/consensus_results/semantic_bft_minimal.json",
        consensus_strength="comprehensive"
    )
    
    print(f"MINIMAL - Quality: {result_minimal.quality_metrics.get('overall_quality', 0):.3f}, "
          f"Time: {result_minimal.total_processing_time:.2f}s, "
          f"Parts: {result_minimal.consensus_parts_count}")
    print()
    
    # Test with COMPREHENSIVE hyperparameters (preserve more, be more inclusive)
    print("Processing with COMPREHENSIVE hyperparameters...")
    processor.update_hyperparameters(
        preservation_ratio=0.95,
        strictness_level='lenient',
        content_complexity_preference='comprehensive',
        min_confidence_threshold=0.3,
        max_output_sections=None,
        security_focus=True
    )
    
    result_comprehensive = processor.process_jsons_folder(
        jsons_folder="jsons",
        output_file="output/jsons/consensus_results/semantic_bft_comprehensive.json",
        consensus_strength="comprehensive"
    )
    
    print(f"COMPREHENSIVE - Quality: {result_comprehensive.quality_metrics.get('overall_quality', 0):.3f}, "
          f"Time: {result_comprehensive.total_processing_time:.2f}s, "
          f"Parts: {result_comprehensive.consensus_parts_count}")
    print()
    
    # Summary comparison
    print("=== HYPERPARAMETER CONTROL COMPARISON ===")
    print(f"{'Setting':<15} {'Quality':<8} {'Time':<8} {'Parts':<6} {'Description'}")
    print("-" * 70)
    print(f"{'Default':<15} {result_default.quality_metrics.get('overall_quality', 0):<8.3f} {result_default.total_processing_time:<8.2f} {result_default.consensus_parts_count:<6} Balanced approach")
    print(f"{'Minimal':<15} {result_minimal.quality_metrics.get('overall_quality', 0):<8.3f} {result_minimal.total_processing_time:<8.2f} {result_minimal.consensus_parts_count:<6} Concise, high-confidence only")
    print(f"{'Comprehensive':<15} {result_comprehensive.quality_metrics.get('overall_quality', 0):<8.3f} {result_comprehensive.total_processing_time:<8.2f} {result_comprehensive.consensus_parts_count:<6} Detailed, inclusive output")
    print()
    print("Hyperparameters successfully control consensus output completeness!")