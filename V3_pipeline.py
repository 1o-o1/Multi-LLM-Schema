#!/usr/bin/env python3
"""
V3 Multi-LLM JSON Consensus Pipeline
Clean, unified architecture using working components

Pipeline Architecture:
1. Raw logs → Preprocessing → Log type detection
2. Prompt1 (Field Extraction) → JSON1 → SEMANTIC_CORE consensus → JSON1_consensus  
3. JSON1_consensus + Prompt2 → Reports.md
4. Reports.md + Prompt3 → JSON2 → OPTIMIZED_CONSENSUS → Final unified schema
"""

import json
import sys
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import root-level components that we know work
try:
    from config import MODEL_CONFIG, PROMPT_TEMPLATES
    from utils import write_json, write_markdown, get_nested_value
    from preprocessing import LogPreprocessor
    from semantic_bft_consensus import SemanticBFTConsensusProcessor
    from few_shot_orchestrator import FewShotOrchestrator
    
    # Import provider modules (these are used by the inline class if needed)
    from providers.claude_provider import ClaudeProvider
    from providers.openai_provider import OpenAIProvider
    from providers.gemini_provider import GeminiProvider
    
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure all required files exist in the root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_api_keys():
    """Get API keys from api_key.txt file (copied from test_few_shot.py)"""
    import re
    
    keys = {}
    try:
        with open("api_key.txt", "r") as f:
            content = f.read()
            
        # Extract keys using regex
        openai_match = re.search(r'OPENAI_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
        gemini_match = re.search(r'GEMINI_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
        anthropic_match = re.search(r'ANTHROPIC_API_KEY\s*=\s*["\']([^"\']+)["\']', content)
        
        if openai_match:
            keys['openai'] = openai_match.group(1)
        if gemini_match:
            keys['gemini'] = gemini_match.group(1)
        if anthropic_match:
            keys['claude'] = anthropic_match.group(1)
            
    except FileNotFoundError:
        logger.error("api_key.txt not found, trying environment variables")
        keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'gemini': os.getenv('GEMINI_API_KEY'),
            'claude': os.getenv('ANTHROPIC_API_KEY')
        }
    
    missing = [k for k, v in keys.items() if not v]
    if missing:
        logger.warning(f"Missing API keys for: {missing}")
    
    return keys

@dataclass
class PipelineStageResult:
    """Result from a pipeline stage"""
    stage_name: str
    content: Dict[str, Any]
    confidence: float
    processing_time: float
    method_used: str
    source_files: List[str]
    metadata: Dict[str, Any]

class V3ConsensusPipeline:
    """
    V3 Multi-LLM JSON Consensus Pipeline
    
    Implements the complete 5-stage pipeline:
    1. Stage 1: Prompt1 execution (field extraction)
    2. Stage 2: JSON1 SEMANTIC_CORE consensus
    3. Stage 3: Prompt2 execution (report generation)
    4. Stage 4: Prompt3 execution (security pattern detection)
    5. Stage 5: JSON2 OPTIMIZED_CONSENSUS
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize V3 consensus pipeline"""
        print("Initializing V3 Multi-LLM JSON Consensus Pipeline")
        print("=" * 60)
        
        # Load configuration - default to config.json if available
        if config_path is None and Path("config.json").exists():
            config_path = "config.json"
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.log_preprocessor = LogPreprocessor()
        
        # Initialize SEMANTIC_CORE consensus processor (for JSON1)
        semantic_core_config = self.config.get('semantic_core', {})
        self.semantic_core_processor = SemanticBFTConsensusProcessor(
            config=semantic_core_config,
            enable_research_mode=True  # Enable research mode to use SBERT, clustering, TED, BFT
        )
        
        # Initialize OPTIMIZED_CONSENSUS processor (for JSON2) with architectural fixes
        optimized_config = self.config.get('optimized_consensus', {})
        
        # RESTORE ORIGINAL WORKING CONFIGURATION - DO NOT MODIFY
        architectural_fixes_config = {
            'consensus_strength': 'comprehensive',
            'use_sbert_embeddings': True,
            'use_semantic_clustering': True,
            'use_semantic_ted': True,
            'use_bft_consensus': True,  # Keep original BFT settings
            'use_weighted_voting': True,
            'use_mcts_optimization': False,  # Keep MCTS disabled as it was working
            'bft_agreement_threshold': 0.67,
            'similarity_threshold': 0.75,  # Original working threshold
            'consensus_threshold': 0.6,    # Original working threshold
            'enable_research_mode': True,
            # RESTORE ORIGINAL WORKING PARAMETERS
            'preservation_ratio': 0.8,
            'min_confidence_threshold': 0.0,
            'min_part_size': 1,
            'clustering_method': 'distance_based'  # Original method
        }
        
        # Merge with user config (user config takes precedence)
        final_config = {**architectural_fixes_config, **optimized_config}
        
        self.optimized_consensus_processor = SemanticBFTConsensusProcessor(
            config=final_config,
            enable_research_mode=True  # Ensures ConsensusOrchestratorV2 is used
        )
        
        # Initialize few-shot orchestrator for prompt execution
        api_keys = get_api_keys()
        self.few_shot_orchestrator = FewShotOrchestrator(api_keys)
        
        print("V3 Pipeline initialized successfully")
        print("SEMANTIC_CORE processor ready for JSON1 consensus")
        print("OPTIMIZED_CONSENSUS processor ready for JSON2 consensus")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            # SEMANTIC_CORE configuration (for JSON1 field consensus)
            'semantic_core': {
                'consensus_strength': 'comprehensive',  # Use comprehensive consensus
                'use_sbert_embeddings': True,  # SBERT for semantic similarity
                'use_semantic_clustering': True,  # Clustering as requested
                'use_semantic_ted': True,  # TED (Tree Edit Distance) as requested
                'use_bft_consensus': False,  # Temporarily disable BFT due to data format issues
                'use_weighted_voting': True,  # Enable weighted voting as fallback
                'use_mcts_optimization': False,  # Disable MCTS to avoid the zero reward issue
                'use_ice_refinement': False,  # Disable ICE - use SBERT+clustering+TED instead
                'use_muse_adaptation': False,  # Disable MUSE for simpler consensus
                'similarity_threshold': 0.5,  # Lower threshold for more matches
                'consensus_threshold': 0.4,   # Lower threshold for more consensus
                'enable_research_mode': True,  # Enable research mode for advanced algorithms
                'min_confidence_threshold': 0.0,  # Accept all confidence levels
                'preservation_ratio': 1.0,  # Keep all fields
                'security_focus': False,  # Disable security focus to avoid restrictive filtering
                'min_part_size': 1,
                'enable_prompt1_mode': True,
                'deconstruction_strategy': 'comprehensive'  # Use comprehensive deconstruction for prompt1 support
            },
            
            # OPTIMIZED_CONSENSUS configuration (for JSON2 security patterns) - ORIGINAL WORKING CONFIG
            'optimized_consensus': {
                'consensus_strength': 'comprehensive',
                'use_sbert_embeddings': True,
                'use_semantic_clustering': True,
                'use_semantic_ted': True,
                'use_bft_consensus': True,
                'use_weighted_voting': True,
                'use_mcts_optimization': False,  # Keep disabled as original
                'bft_agreement_threshold': 0.67,
                'similarity_threshold': 0.75,
                'consensus_threshold': 0.6,
                'enable_research_mode': True,
                # RESTORE ORIGINAL WORKING PARAMETERS
                'preservation_ratio': 0.8,
                'strictness_level': 'moderate',
                'security_focus': True,
                'content_complexity_preference': 'balanced'
            },
            
            # Pipeline settings
            'output_directory': 'output',
            'enable_caching': True,
            'max_parallel_models': 3
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _create_prompt2_format(self, consensus_json: Dict[str, Any], stage1_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create the proper format for Prompt2 from consensus JSON with semantic log_type selection"""
        from improved_log_type_consensus import SemanticLogTypeSelector
        
        # Extract log_types and follow_up_queries from original stage1 results
        log_types = []
        all_follow_up_queries = []
        
        # Collect all log_types and follow_up_queries from stage1 results
        for provider_result in stage1_results.values():
            parsed_json = provider_result.get('parsed_json', {})
            if isinstance(parsed_json, dict):
                if 'log_type' in parsed_json:
                    log_types.append(parsed_json['log_type'])
                if 'follow_up_queries' in parsed_json:
                    all_follow_up_queries.extend(parsed_json['follow_up_queries'])
        
        # Use semantic similarity to select best log_type
        if log_types:
            log_type_selector = SemanticLogTypeSelector()
            similarities = log_type_selector.calculate_pairwise_similarities(log_types)
            best_log_type = max(similarities.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected log_type via semantic similarity: {best_log_type}")
        else:
            best_log_type = consensus_json.get('parsed_json', {}).get('log_type', "Unknown Log Type")
        
        # Deduplicate and limit follow_up_queries
        unique_queries = list(set(all_follow_up_queries))
        final_queries = unique_queries[:2] if unique_queries else [
            f"threat hunting queries for {best_log_type}",
            f"common attack vectors related to {best_log_type}"
        ]
        
        # Extract fields from consensus
        fields = consensus_json.get('parsed_json', {}).get('fields', {})
        
        # Create the proper format
        prompt2_format = {
            "log_type": best_log_type,
            "follow_up_queries": final_queries,
            "fields": fields
        }
        
        return prompt2_format
    
    def _fix_original_content_wrapper(self, consensus_json: Dict[str, Any]) -> Dict[str, Any]:
        """Fix _original_content wrapper issue in consensus results"""
        if not isinstance(consensus_json, dict):
            return consensus_json
        
        # Deep copy to avoid modifying original
        import copy
        fixed_json = copy.deepcopy(consensus_json)
        
        parsed_json = fixed_json.get('parsed_json', {})
        
        # Fix detection_rule_checklist if it has _original_content wrapper
        if 'detection_rule_checklist' in parsed_json:
            detection_rules = parsed_json['detection_rule_checklist']
            
            if isinstance(detection_rules, dict) and '_original_content' in detection_rules:
                logger.info("Fixing _original_content wrapper in detection_rule_checklist")
                rules_list = detection_rules['_original_content']
                
                # Convert to proper list format with consensus metadata
                merged_rules = []
                for i, rule in enumerate(rules_list):
                    merged_rule = rule.copy()
                    merged_rule['consensus_votes'] = 1
                    merged_rule['consensus_sources'] = [f'optimized_consensus_{i}']
                    merged_rules.append(merged_rule)
                
                # Replace the problematic structure
                parsed_json['detection_rule_checklist'] = merged_rules
                logger.info(f"Fixed detection_rule_checklist: Converted to list with {len(merged_rules)} rules")
        
        # Fix observations structure (both behavioral_patterns and temporal_patterns)
        if 'observations' in parsed_json:
            observations = parsed_json['observations']
            
            # Handle both behavioral_patterns and temporal_patterns
            for pattern_category in ['behavioral_patterns', 'temporal_patterns']:
                if pattern_category in observations:
                    pattern_section = observations[pattern_category]
                    
                    for pattern_type in ['malicious', 'anomalous', 'vulnerable']:
                        if pattern_type in pattern_section:
                            pattern_data = pattern_section[pattern_type]
                            
                            if isinstance(pattern_data, dict) and '_original_content' in pattern_data:
                                logger.info(f"Fixing _original_content wrapper in observations.{pattern_category}.{pattern_type}")
                                
                                # Convert to proper list format with consensus metadata
                                original_patterns = pattern_data['_original_content']
                                merged_patterns = []
                                
                                for i, pattern in enumerate(original_patterns):
                                    merged_pattern = pattern.copy()
                                    # Add consensus metadata if not already present
                                    if 'consensus_votes' not in merged_pattern:
                                        merged_pattern['consensus_votes'] = 1
                                    if 'consensus_sources' not in merged_pattern:
                                        merged_pattern['consensus_sources'] = [f'optimized_consensus_{pattern_category}_{pattern_type}_{i}']
                                    merged_patterns.append(merged_pattern)
                                
                                # Replace the problematic structure
                                pattern_section[pattern_type] = merged_patterns
                                logger.info(f"Fixed {pattern_category}.{pattern_type}: Converted to list with {len(merged_patterns)} patterns")
        
        # Apply same fix to other top-level sections that might have similar issues
        for section_name in ['indicators_of_compromise', 'attack_pattern_checks', 'vulnerability_checks']:
            if section_name in parsed_json:
                section_data = parsed_json[section_name]
                if isinstance(section_data, dict) and '_original_content' in section_data:
                    logger.info(f"Fixing _original_content wrapper in {section_name}")
                    parsed_json[section_name] = section_data['_original_content']
        
        return fixed_json
    
    def execute_pipeline(self, input_data: str, output_dir: str = "output") -> Dict[str, Any]:
        """
        Execute the complete V3 consensus pipeline
        
        Args:
            input_data: Path to log file or log content
            output_dir: Output directory for results
            
        Returns:
            Complete pipeline results with all stages
        """
        pipeline_start_time = time.time()
        
        try:
            print("\n" + " STARTING V3 CONSENSUS PIPELINE ")
            print("=" * 80)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Stage 1: Preprocessing and Prompt1 execution
            stage1_result = self._execute_stage1_prompt1(input_data, output_path)
            
            # Stage 2: JSON1 SEMANTIC_CORE consensus
            stage2_result = self._execute_stage2_semantic_core_consensus(stage1_result, output_path)
            
            # Stage 3: Prompt2 execution (Report generation)
            stage3_result = self._execute_stage3_prompt2(stage2_result, output_path)
            
            # Stage 4: Prompt3 execution  
            stage4_result = self._execute_stage4_prompt3(stage3_result, stage2_result, output_path)
            
            # Stage 5: JSON2 OPTIMIZED_CONSENSUS
            stage5_result = self._execute_stage5_optimized_consensus(stage4_result, output_path)
            
            # Compile final results
            total_time = time.time() - pipeline_start_time
            
            # Convert PipelineStageResult objects to dictionaries for JSON serialization
            def stage_result_to_dict(stage_result):
                return {
                    'stage_name': stage_result.stage_name,
                    'content': stage_result.content,
                    'confidence': stage_result.confidence,
                    'processing_time': stage_result.processing_time,
                    'method_used': stage_result.method_used,
                    'source_files': stage_result.source_files,
                    'metadata': stage_result.metadata
                }
            
            final_result = {
                'pipeline_version': 'V3',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_processing_time': total_time,
                'stages': {
                    'stage1_prompt1': stage_result_to_dict(stage1_result),
                    'stage2_semantic_core_consensus': stage_result_to_dict(stage2_result),
                    'stage3_prompt2': stage_result_to_dict(stage3_result),
                    'stage4_prompt3': stage_result_to_dict(stage4_result),
                    'stage5_optimized_consensus': stage_result_to_dict(stage5_result)
                },
                'final_unified_schema': stage5_result.content,
                'overall_confidence': (stage2_result.confidence + stage5_result.confidence) / 2,
                'consensus_methods_used': {
                    'json1_consensus': stage2_result.method_used,
                    'json2_consensus': stage5_result.method_used
                }
            }
            
            # Save final results
            result_file = output_path / "v3_pipeline_result.json"
            write_json(str(result_file), final_result)
            
            print(f"\nV3 Pipeline completed successfully!")
            print(f"Results saved to: {result_file}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Overall confidence: {final_result['overall_confidence']:.3f}")
            
            return final_result
            
        except Exception as e:
            print(f"\nPipeline execution failed: {e}")
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _execute_stage1_prompt1(self, input_data: str, output_path: Path) -> PipelineStageResult:
        """Stage 1: Execute Prompt1 (Field Extraction) across multiple models"""
        print("\n" + "=" * 60)
        print("STAGE 1: Prompt1 Execution (Field Extraction)")
        print("=" * 60)
        
        stage_start_time = time.time()
        
        # Preprocess input data
        if Path(input_data).exists():
            log_content = Path(input_data).read_text()
        else:
            log_content = input_data
        
        preprocessed_data = self.log_preprocessor.process(log_content)
        
        # Execute Prompt1 across all models
        prompt1_results = self.few_shot_orchestrator.run_prompt1(
            processed_data=preprocessed_data
        )
        
        # Save individual results
        prompt1_dir = output_path / "stage1_prompt1"
        prompt1_dir.mkdir(exist_ok=True)
        
        json_files = []
        for provider, result in prompt1_results.items():
            result_file = prompt1_dir / f"{provider}_prompt1.json"
            write_json(str(result_file), result)
            json_files.append(str(result_file))
        
        processing_time = time.time() - stage_start_time
        
        return PipelineStageResult(
            stage_name="stage1_prompt1",
            content=prompt1_results,
            confidence=0.8,
            processing_time=processing_time,
            method_used="multi_model_prompt1",
            source_files=json_files,
            metadata={
                'models_used': list(prompt1_results.keys()),
                'preprocessed_log_length': len(preprocessed_data.cleaned_data)
            }
        )
    
    def _execute_stage2_semantic_core_consensus(self, stage1_result: PipelineStageResult, output_path: Path) -> PipelineStageResult:
        """Stage 2: Apply SEMANTIC_CORE consensus to JSON1 files"""
        print("\n" + "=" * 60)
        print("STAGE 2: JSON1 SEMANTIC_CORE Consensus")
        print("=" * 60)
        
        stage_start_time = time.time()
        
        # Use SemanticBFTConsensusProcessor for SEMANTIC_CORE consensus
        consensus_result = self.semantic_core_processor.process_json_files(
            json_files=stage1_result.source_files,
            consensus_strength='comprehensive'
        )
        
        # Save consensus result
        consensus_dir = output_path / "stage2_semantic_core"
        consensus_dir.mkdir(exist_ok=True)
        
        consensus_file = consensus_dir / "json1_consensus.json"
        write_json(str(consensus_file), consensus_result.unified_json)
        
        # Create json1_consensus_prompt2.json with proper format for Prompt2
        prompt2_format = self._create_prompt2_format(consensus_result.unified_json, stage1_result.content)
        prompt2_file = consensus_dir / "json1_consensus_prompt2.json" 
        write_json(str(prompt2_file), prompt2_format)
        
        processing_time = time.time() - stage_start_time
        
        return PipelineStageResult(
            stage_name="stage2_semantic_core_consensus",
            content=prompt2_format,  # Use prompt2 format as the main content
            confidence=consensus_result.quality_metrics.get('overall_quality', 0.6),
            processing_time=processing_time,
            method_used="SEMANTIC_CORE",
            source_files=[str(consensus_file), str(prompt2_file)],
            metadata={
                'consensus_parts_count': consensus_result.consensus_parts_count,
                'original_file_count': consensus_result.original_file_count,
                'quality_metrics': consensus_result.quality_metrics
            }
        )
    
    def _execute_stage3_prompt2(self, stage2_result: PipelineStageResult, output_path: Path) -> PipelineStageResult:
        """Stage 3: Execute Prompt2 (Report generation) using JSON1_consensus"""
        print("\n" + "=" * 60)
        print("STAGE 3: Prompt2 Execution (Report Generation)")
        print("=" * 60)
        
        stage_start_time = time.time()
        
        # Execute Prompt2 using consensus from Stage 2
        # Convert consensus back to prompt1_results format for compatibility
        prompt1_results_format = {"consensus": {"parsed_json": stage2_result.content}}
        prompt2_results = self.few_shot_orchestrator.run_prompt2(
            prompt1_results=prompt1_results_format
        )
        
        # Save reports
        prompt2_dir = output_path / "stage3_prompt2"
        prompt2_dir.mkdir(exist_ok=True)
        
        report_files = []
        for provider, report_content in prompt2_results.items():
            report_file = prompt2_dir / f"{provider}_report.md"
            write_markdown(str(report_file), report_content)
            report_files.append(str(report_file))
        
        processing_time = time.time() - stage_start_time
        
        return PipelineStageResult(
            stage_name="stage3_prompt2",
            content=prompt2_results,
            confidence=0.8,
            processing_time=processing_time,
            method_used="multi_model_prompt2",
            source_files=report_files,
            metadata={
                'reports_generated': len(prompt2_results),
                'input_consensus_confidence': stage2_result.confidence
            }
        )
    
    def _execute_stage4_prompt3(self, stage3_result: PipelineStageResult, stage2_result: PipelineStageResult, output_path: Path) -> PipelineStageResult:
        """Stage 4: Execute Prompt3 (Security Pattern Detection) using reports"""
        print("\n" + "=" * 60)
        print("STAGE 4: Prompt3 Execution (Security Pattern Detection)")
        print("=" * 60)
        
        stage_start_time = time.time()
        
        # Execute Prompt3 using reports from Stage 3 and JSON1 consensus from Stage 2
        # Convert consensus back to prompt1_results format for compatibility
        prompt1_results_format = {"consensus": {"parsed_json": stage2_result.content}}
        prompt3_results = self.few_shot_orchestrator.run_prompt3(
            prompt1_results=prompt1_results_format,
            prompt2_results=stage3_result.content
        )
        
        # Save JSON2 results
        prompt3_dir = output_path / "stage4_prompt3"
        prompt3_dir.mkdir(exist_ok=True)
        
        json_files = []
        for provider, result in prompt3_results.items():
            result_file = prompt3_dir / f"{provider}_prompt3.json"
            write_json(str(result_file), result)
            json_files.append(str(result_file))
        
        processing_time = time.time() - stage_start_time
        
        return PipelineStageResult(
            stage_name="stage4_prompt3",
            content=prompt3_results,
            confidence=0.8,
            processing_time=processing_time,
            method_used="multi_model_prompt3",
            source_files=json_files,
            metadata={
                'models_used': list(prompt3_results.keys()),
                'input_reports_count': len(stage3_result.content)
            }
        )
    
    def _execute_stage5_optimized_consensus(self, stage4_result: PipelineStageResult, output_path: Path) -> PipelineStageResult:
        """Stage 5: Apply OPTIMIZED_CONSENSUS to JSON2 files using semantic_bft_consensus"""
        print("\n" + "=" * 60)
        print("STAGE 5: JSON2 OPTIMIZED_CONSENSUS")
        print("=" * 60)
        
        stage_start_time = time.time()
        
        print("Using SemanticBFTConsensusProcessor with ORIGINAL WORKING CONFIG")
        
        # Use SemanticBFTConsensusProcessor for OPTIMIZED_CONSENSUS
        consensus_result = self.optimized_consensus_processor.process_json_files(
            json_files=stage4_result.source_files,
            consensus_strength='comprehensive'
        )
        
        # Apply the _original_content wrapper fix
        final_content = self._fix_original_content_wrapper(consensus_result.unified_json)
        method_used = "OPTIMIZED_CONSENSUS_V2"
        confidence = consensus_result.quality_metrics.get('overall_quality', 0.6)
        quality_metrics = consensus_result.quality_metrics
        
        # Save final consensus result
        consensus_dir = output_path / "stage5_optimized_consensus"
        consensus_dir.mkdir(exist_ok=True)
        
        final_consensus_file = consensus_dir / "final_unified_schema.json"
        write_json(str(final_consensus_file), final_content)
        
        processing_time = time.time() - stage_start_time
        
        final_config = self.optimized_consensus_processor.config
        
        return PipelineStageResult(
            stage_name="stage5_optimized_consensus", 
            content=final_content,
            confidence=confidence,
            processing_time=processing_time,
            method_used=method_used,
            source_files=[str(final_consensus_file)],
            metadata={
                'quality_metrics': quality_metrics,
                'architectural_fixes_applied': True,
                'knowledge_preservation_approach': 'consensus_filtered',
                'original_file_count': len(stage4_result.source_files),
                'processing_method': method_used,
                'bft_enabled': final_config.get('use_bft_consensus', False),
                'muse_enabled': final_config.get('use_muse_adaptation', False),
                'ice_enabled': final_config.get('use_ice_refinement', False),
                'ted_enabled': final_config.get('use_semantic_ted', False),
                'clustering_enabled': final_config.get('use_semantic_clustering', False),
                'dempster_shafer_enabled': final_config.get('use_dempster_shafer', False),
                'clustering_method': final_config.get('clustering_method', 'simple'),
                'centralized_embeddings_enabled': final_config.get('enable_centralized_embeddings', False)
            }
        )


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V3 Multi-LLM JSON Consensus Pipeline')
    parser.add_argument('--input', required=True, help='Input log file or log content')
    parser.add_argument('--output', default='output', help='Output directory for results')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        pipeline = V3ConsensusPipeline(config_path=args.config)
        result = pipeline.execute_pipeline(args.input, args.output)
        print("\nPipeline execution completed successfully")
        
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()