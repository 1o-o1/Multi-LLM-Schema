"""
Multi-Agent Consensus Orchestrator
Central coordinator for all consensus algorithms and techniques
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .embedding_service import EmbeddingService
from .graph_clustering import GraphClustering
from .semantic_similarity import SemanticSimilarity
from .csp_framework_alignment import CSPFrameworkAlignment
from .bft_consensus import BFTConsensus
from .mcts_optimization import MCTSOptimization
from .dempster_shafer import DempsterShaferEngine
from .universal_consensus_engine import UniversalConsensusEngine, ConsensusStrength

logger = logging.getLogger(__name__)

class ConsensusOrchestrator:
    """Central orchestrator for multi-agent consensus processes"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize consensus orchestrator with all consensus engines
        
        Args:
            config: Configuration dictionary for consensus parameters
        """
        # Merge provided config with defaults to ensure all required keys are present
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Initialize all consensus engines
        self.embedding_service = EmbeddingService(
            model_name=self.config.get('embedding_model', 'gemini-embedding-001')
        )
        self.graph_clustering = GraphClustering()
        self.semantic_similarity = SemanticSimilarity()
        self.csp_alignment = CSPFrameworkAlignment()
        self.bft_consensus = BFTConsensus(
            fault_tolerance=self.config.get('bft_fault_tolerance', 0.33)
        )
        self.mcts_optimization = MCTSOptimization(
            exploration_constant=self.config.get('mcts_exploration', 1.41),
            max_iterations=self.config.get('mcts_iterations', 1000)
        )
        self.dempster_shafer = DempsterShaferEngine()
        
        # Initialize the Universal Consensus Engine
        self.universal_engine = UniversalConsensusEngine(config=self.config)
        
        # Consensus state
        self.consensus_results = {}
        self.algorithm_weights = self.config.get('algorithm_weights', {
            'embedding_similarity': 0.2,
            'graph_clustering': 0.15,
            'csp_alignment': 0.15,
            'bft_consensus': 0.2,
            'mcts_optimization': 0.15,
            'dempster_shafer': 0.15
        })
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embedding_model': 'gemini-embedding-001',
            'bft_fault_tolerance': 0.33,
            'mcts_exploration': 1.41,
            'mcts_iterations': 500,
            'similarity_threshold': 0.7,
            'consensus_threshold': 0.6,
            'max_parallel_algorithms': 4,
            'timeout_seconds': 300,
            'algorithm_weights': {
                'embedding_similarity': 0.2,
                'graph_clustering': 0.15,
                'csp_alignment': 0.15,
                'bft_consensus': 0.2,
                'mcts_optimization': 0.15,
                'dempster_shafer': 0.15
            }
        }
    
    def unified_consensus(self, 
                        provider_results: Dict[str, Dict],
                        target_key_path: str,
                        consensus_type: str = 'comprehensive',
                        algorithms: Optional[List[str]] = None,
                        custom_similarity_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        Main unified consensus method using Universal Consensus Engine
        
        Args:
            provider_results: Dictionary of provider results
            target_key_path: Dot-separated path to target data (e.g., 'observations.behavioral_patterns.malicious')
            consensus_type: Type of consensus ('fast', 'comprehensive', 'uncertainty_aware')
            algorithms: Specific algorithms to run (optional - handled by consensus strength)
            custom_similarity_func: Custom similarity function for domain-specific logic
            
        Returns:
            Unified consensus results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting universal consensus for target: {target_key_path}")
            
            # Validate inputs
            if not provider_results:
                return {'error': 'No provider results provided'}
            
            # Map consensus type to strength level
            strength_mapping = {
                'fast': ConsensusStrength.WEAK,
                'comprehensive': ConsensusStrength.STRONG, 
                'uncertainty_aware': ConsensusStrength.MAXIMUM
            }
            consensus_strength = strength_mapping.get(consensus_type, ConsensusStrength.STRONG)
            
            # Use Universal Consensus Engine
            universal_result = self.universal_engine.create_consensus(
                provider_results=provider_results,
                target_path=target_key_path,
                consensus_strength=consensus_strength,
                custom_similarity_func=custom_similarity_func
            )
            
            # Convert Universal result to orchestrator format for compatibility
            unified_result = {
                'target_key_path': target_key_path,
                'consensus_type': consensus_type,
                'consensus_strength': consensus_strength.value,
                'algorithms_used': list(universal_result.algorithm_results.keys()),
                'input_model_count': universal_result.input_model_count,
                'extracted_items_count': universal_result.extracted_items_count,
                'consensus_clusters': universal_result.consensus_clusters,
                'final_consensus': self._convert_consensus_items_to_dict(universal_result.final_consensus_items),
                'consensus_quality': universal_result.consensus_quality_metrics,
                'algorithm_results': universal_result.algorithm_results,
                'processing_metadata': universal_result.processing_metadata,
                'cluster_count': universal_result.cluster_count,
                'compression_ratio': universal_result.compression_ratio,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
            # Store results
            self.consensus_results[f"{target_key_path}_{int(time.time())}"] = unified_result
            
            logger.info(f"Universal consensus completed in {unified_result['processing_time']:.2f}s with quality score: {universal_result.consensus_quality.get('overall_quality', 0):.3f}")
            
            return unified_result
            
        except Exception as e:
            logger.error(f"Universal consensus failed: {e}")
            return {
                'error': str(e),
                'target_key_path': target_key_path,
                'processing_time': time.time() - start_time
            }
    
    def _analyze_target_structure(self, 
                                provider_results: Dict[str, Dict], 
                                target_key_path: str) -> Dict[str, Any]:
        """Analyze the structure of target data across providers"""
        structure_info = {
            'valid': True,
            'data_type': None,
            'item_count_range': (0, 0),
            'common_keys': set(),
            'data_patterns': {},
            'error': None
        }
        
        try:
            all_target_data = []
            item_counts = []
            all_keys = []
            
            for provider, result in provider_results.items():
                target_data = self.embedding_service._get_nested_value(result, target_key_path)
                
                if target_data is None:
                    continue
                    
                all_target_data.append(target_data)
                
                if isinstance(target_data, dict):
                    item_counts.append(len(target_data))
                    all_keys.extend(target_data.keys())
                elif isinstance(target_data, list):
                    item_counts.append(len(target_data))
                    # For lists, analyze first few items for structure
                    for item in target_data[:5]:
                        if isinstance(item, dict):
                            all_keys.extend(item.keys())
            
            if not all_target_data:
                structure_info['valid'] = False
                structure_info['error'] = f'No data found at path: {target_key_path}'
                return structure_info
            
            # Determine data type
            first_data = all_target_data[0]
            if isinstance(first_data, dict):
                structure_info['data_type'] = 'dict'
            elif isinstance(first_data, list):
                structure_info['data_type'] = 'list'
            else:
                structure_info['data_type'] = 'other'
            
            # Calculate item count range
            if item_counts:
                structure_info['item_count_range'] = (min(item_counts), max(item_counts))
            
            # Find common keys
            if all_keys:
                key_counts = defaultdict(int)
                for key in all_keys:
                    key_counts[key] += 1
                
                # Keys present in majority of items
                threshold = len(all_target_data) * 0.5
                structure_info['common_keys'] = set(
                    key for key, count in key_counts.items() if count >= threshold
                )
            
            logger.info(f"Target structure analysis: {structure_info['data_type']} with {len(structure_info['common_keys'])} common keys")
            return structure_info
            
        except Exception as e:
            structure_info['valid'] = False
            structure_info['error'] = str(e)
            return structure_info
    
    def _select_algorithms_for_type(self, 
                                  consensus_type: str, 
                                  target_structure: Dict[str, Any]) -> List[str]:
        """Select appropriate algorithms based on consensus type and data structure"""
        
        if consensus_type == 'fast':
            return ['embedding_similarity', 'semantic_similarity']
        elif consensus_type == 'uncertainty_aware':
            return ['dempster_shafer', 'bft_consensus', 'embedding_similarity']
        elif consensus_type == 'comprehensive':
            return ['embedding_similarity', 'semantic_similarity', 'bft_consensus', 'dempster_shafer', 'csp_alignment']
        else:
            # Default comprehensive
            return ['embedding_similarity', 'semantic_similarity', 'bft_consensus', 'dempster_shafer']
    
    def _run_consensus_algorithms(self, 
                                provider_results: Dict[str, Dict],
                                target_key_path: str,
                                algorithms: List[str],
                                target_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Run selected consensus algorithms in parallel"""
        
        algorithm_results = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.config.get('max_parallel_algorithms', 4)) as executor:
            
            # Submit algorithm tasks
            future_to_algorithm = {}
            
            for algorithm in algorithms:
                future = executor.submit(
                    self._run_single_algorithm, 
                    algorithm, provider_results, target_key_path, target_structure
                )
                future_to_algorithm[future] = algorithm
            
            # Collect results with timeout
            timeout = self.config.get('timeout_seconds', 300)
            
            for future in as_completed(future_to_algorithm, timeout=timeout):
                algorithm = future_to_algorithm[future]
                try:
                    result = future.result()
                    algorithm_results[algorithm] = result
                    logger.info(f"Algorithm {algorithm} completed successfully")
                except Exception as e:
                    logger.error(f"Algorithm {algorithm} failed: {e}")
                    algorithm_results[algorithm] = {'error': str(e)}
        
        return algorithm_results
    
    def _run_single_algorithm(self, 
                            algorithm: str,
                            provider_results: Dict[str, Dict],
                            target_key_path: str,
                            target_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single consensus algorithm"""
        
        try:
            if algorithm == 'embedding_similarity':
                return self._run_embedding_similarity(provider_results, target_key_path)
                
            elif algorithm == 'semantic_similarity':
                return self._run_semantic_similarity(provider_results, target_key_path)
                
            elif algorithm == 'graph_clustering':
                return self._run_graph_clustering(provider_results, target_key_path)
                
            elif algorithm == 'csp_alignment':
                return self._run_csp_alignment(provider_results, target_key_path)
                
            elif algorithm == 'bft_consensus':
                return self._run_bft_consensus(provider_results, target_key_path)
                
            elif algorithm == 'mcts_optimization':
                return self._run_mcts_optimization(provider_results, target_key_path)
                
            elif algorithm == 'dempster_shafer':
                return self._run_dempster_shafer(provider_results, target_key_path)
                
            else:
                return {'error': f'Unknown algorithm: {algorithm}'}
                
        except Exception as e:
            logger.error(f"Algorithm {algorithm} execution failed: {e}")
            return {'error': str(e)}
    
    def _run_embedding_similarity(self, 
                                provider_results: Dict[str, Dict], 
                                target_key_path: str) -> Dict[str, Any]:
        """Run embedding-based similarity clustering"""
        try:
            # Use generic hierarchical clustering
            clustered_items = self.embedding_service.hierarchical_clustering_generic(
                provider_results, target_key_path
            )
            
            # Convert clusters to consensus format
            consensus_items = {}
            for cluster_name, items in clustered_items.items():
                if len(items) > 1:  # Only include multi-provider clusters
                    # Create consensus item by merging
                    consensus_item = self._merge_items_by_similarity(items)
                    consensus_items[cluster_name] = consensus_item
            
            return {
                'algorithm': 'embedding_similarity',
                'consensus_items': consensus_items,
                'clusters': clustered_items,
                'cluster_count': len(clustered_items),
                'consensus_count': len(consensus_items)
            }
            
        except Exception as e:
            return {'error': f'Embedding similarity failed: {e}'}
    
    def _run_semantic_similarity(self, 
                               provider_results: Dict[str, Dict], 
                               target_key_path: str) -> Dict[str, Any]:
        """Run semantic similarity analysis"""
        try:
            # Cross-model comparison
            comparison_results = self.semantic_similarity.cross_model_result_comparison(provider_results)
            
            # Extract consensus from field consensus analysis
            field_consensus = comparison_results.get('field_consensus', {})
            
            consensus_items = {}
            for field_name, consensus_info in field_consensus.items():
                if consensus_info.get('presence_consensus', 0) >= self.config.get('consensus_threshold', 0.6):
                    consensus_items[field_name] = {
                        'consensus_score': consensus_info.get('average_agreement', 0),
                        'supporting_providers': consensus_info.get('providers_with_field', []),
                        'property_agreement': consensus_info.get('property_agreement', {})
                    }
            
            return {
                'algorithm': 'semantic_similarity',
                'consensus_items': consensus_items,
                'comparison_results': comparison_results,
                'consensus_count': len(consensus_items)
            }
            
        except Exception as e:
            return {'error': f'Semantic similarity failed: {e}'}
    
    def _run_graph_clustering(self, 
                            provider_results: Dict[str, Dict], 
                            target_key_path: str) -> Dict[str, Any]:
        """Run graph-based clustering"""
        try:
            # Build relationship graph
            graph = self.graph_clustering.build_field_relationship_graph(provider_results)
            
            # Perform community detection
            communities = self.graph_clustering.louvain_community_detection()
            
            # Calculate centrality measures
            centrality = self.graph_clustering.calculate_centrality_measures()
            
            # Convert communities to consensus format
            consensus_items = {}
            community_groups = defaultdict(list)
            
            for node, community_id in communities.items():
                community_groups[community_id].append(node)
            
            for community_id, nodes in community_groups.items():
                if len(nodes) > 1:
                    consensus_items[f"community_{community_id}"] = {
                        'member_nodes': nodes,
                        'community_size': len(nodes),
                        'centrality_scores': {node: centrality.get('pagerank', {}).get(node, 0) for node in nodes}
                    }
            
            return {
                'algorithm': 'graph_clustering',
                'consensus_items': consensus_items,
                'communities': communities,
                'centrality_measures': centrality,
                'graph_metrics': self.graph_clustering.calculate_graph_metrics()
            }
            
        except Exception as e:
            return {'error': f'Graph clustering failed: {e}'}
    
    def _run_csp_alignment(self, 
                         provider_results: Dict[str, Dict], 
                         target_key_path: str) -> Dict[str, Any]:
        """Run CSP-based alignment"""
        try:
            alignment_result = self.csp_alignment.generate_alignment_solution(provider_results)
            
            return {
                'algorithm': 'csp_alignment',
                'consensus_items': alignment_result.get('alignment_groups', {}),
                'csp_solution': alignment_result.get('csp_solution', {}),
                'success': alignment_result.get('success', False),
                'n_variables': alignment_result.get('n_variables', 0),
                'n_constraints': alignment_result.get('n_constraints', 0)
            }
            
        except Exception as e:
            return {'error': f'CSP alignment failed: {e}'}
    
    def _run_bft_consensus(self, 
                         provider_results: Dict[str, Dict], 
                         target_key_path: str) -> Dict[str, Any]:
        """Run Byzantine Fault Tolerance consensus"""
        try:
            # Extract field proposals in BFT format
            field_proposals = {}
            for provider, result in provider_results.items():
                target_data = self.embedding_service._get_nested_value(result, target_key_path)
                if target_data:
                    field_proposals[provider] = target_data
            
            # Run multi-algorithm BFT consensus
            bft_result = self.bft_consensus.multi_algorithm_consensus(provider_results)
            
            return {
                'algorithm': 'bft_consensus',
                'consensus_items': bft_result.get('combined_consensus', {}),
                'individual_results': bft_result.get('individual_results', {}),
                'algorithms_used': bft_result.get('algorithms_used', []),
                'total_providers': bft_result.get('total_providers', 0)
            }
            
        except Exception as e:
            return {'error': f'BFT consensus failed: {e}'}
    
    def _run_mcts_optimization(self, 
                             provider_results: Dict[str, Dict], 
                             target_key_path: str) -> Dict[str, Any]:
        """Run MCTS optimization"""
        try:
            optimization_result = self.mcts_optimization.optimize_schema_consensus(provider_results)
            
            return {
                'algorithm': 'mcts_optimization',
                'consensus_items': optimization_result.get('optimized_schema', {}),
                'action_sequence': optimization_result.get('action_sequence', []),
                'final_reward': optimization_result.get('final_reward', 0),
                'search_statistics': optimization_result.get('search_statistics', {})
            }
            
        except Exception as e:
            return {'error': f'MCTS optimization failed: {e}'}
    
    def _run_dempster_shafer(self, 
                           provider_results: Dict[str, Dict], 
                           target_key_path: str) -> Dict[str, Any]:
        """Run Dempster-Shafer analysis"""
        try:
            # Run multi-field DS analysis
            ds_result = self.dempster_shafer.multi_field_consensus_analysis(provider_results)
            
            # Extract consensus items from DS analysis
            consensus_items = {}
            field_analyses = ds_result.get('field_analyses', {})
            
            for field_name, analysis in field_analyses.items():
                if 'error' not in analysis:
                    recommendations = analysis.get('consensus_recommendations', {})
                    if recommendations.get('consensus_field_properties'):
                        consensus_items[field_name] = recommendations['consensus_field_properties']
            
            return {
                'algorithm': 'dempster_shafer',
                'consensus_items': consensus_items,
                'field_analyses': field_analyses,
                'overall_uncertainty': ds_result.get('overall_uncertainty', {}),
                'consensus_summary': ds_result.get('consensus_summary', {})
            }
            
        except Exception as e:
            return {'error': f'Dempster-Shafer failed: {e}'}
    
    def _merge_items_by_similarity(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar items from different providers"""
        if not items:
            return {}
        
        if len(items) == 1:
            return items[0]['data']
        
        # Extract data from items
        item_data_list = [item['data'] for item in items]
        
        # Merge based on data type
        first_item = item_data_list[0]
        
        if isinstance(first_item, dict):
            return self._merge_dict_items(item_data_list, items)
        elif isinstance(first_item, str):
            return self._merge_string_items(item_data_list, items)
        else:
            # For other types, take most common or first
            return first_item
    
    def _merge_dict_items(self, item_data_list: List[Dict], items: List[Dict]) -> Dict[str, Any]:
        """Merge dictionary items"""
        merged = {}
        
        # Get all keys
        all_keys = set()
        for item_data in item_data_list:
            all_keys.update(item_data.keys())
        
        # Merge each key
        for key in all_keys:
            values = []
            for item_data in item_data_list:
                if key in item_data:
                    values.append(item_data[key])
            
            if values:
                if isinstance(values[0], str):
                    # Take longest or most common string
                    merged[key] = max(values, key=len)
                elif isinstance(values[0], (int, float)):
                    # Take average for numbers
                    merged[key] = np.mean(values)
                else:
                    # Take first value for other types
                    merged[key] = values[0]
        
        # Add metadata about merge
        merged['_consensus_metadata'] = {
            'source_providers': [item['provider'] for item in items],
            'merge_count': len(items),
            'merge_timestamp': time.time()
        }
        
        return merged
    
    def _merge_string_items(self, item_data_list: List[str], items: List[Dict]) -> str:
        """Merge string items"""
        # For strings, take the longest one as it's likely most descriptive
        longest_string = max(item_data_list, key=len)
        return longest_string
    
    def _meta_consensus_combination(self, 
                                  algorithm_results: Dict[str, Any],
                                  target_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple algorithms using meta-consensus"""
        
        # Collect all consensus items from all algorithms
        all_consensus_items = {}
        algorithm_votes = defaultdict(list)
        
        for algorithm, result in algorithm_results.items():
            if 'error' in result:
                continue
                
            consensus_items = result.get('consensus_items', {})
            
            for item_name, item_data in consensus_items.items():
                # Normalize item name
                normalized_name = self._normalize_item_name(item_name)
                
                if normalized_name not in all_consensus_items:
                    all_consensus_items[normalized_name] = {
                        'data': item_data,
                        'supporting_algorithms': [],
                        'algorithm_scores': {},
                        'consensus_confidence': 0.0
                    }
                
                # Add algorithm vote
                algorithm_votes[normalized_name].append(algorithm)
                all_consensus_items[normalized_name]['supporting_algorithms'].append(algorithm)
                
                # Extract confidence score if available
                confidence = self._extract_confidence_from_result(result, item_name)
                all_consensus_items[normalized_name]['algorithm_scores'][algorithm] = confidence
        
        # Calculate final consensus scores
        final_consensus = {}
        
        for item_name, item_info in all_consensus_items.items():
            # Calculate weighted consensus score
            total_weight = 0.0
            weighted_score = 0.0
            
            for algorithm in item_info['supporting_algorithms']:
                weight = self.algorithm_weights.get(algorithm, 0.1)
                score = item_info['algorithm_scores'].get(algorithm, 0.5)
                
                weighted_score += weight * score
                total_weight += weight
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.0
            
            # Only include items above threshold
            if final_score >= self.config.get('consensus_threshold', 0.6):
                final_consensus[item_name] = {
                    'data': item_info['data'],
                    'consensus_score': final_score,
                    'supporting_algorithms': item_info['supporting_algorithms'],
                    'algorithm_count': len(item_info['supporting_algorithms']),
                    'algorithm_scores': item_info['algorithm_scores']
                }
        
        logger.info(f"Meta-consensus selected {len(final_consensus)} items from {len(all_consensus_items)} candidates")
        
        return final_consensus
    
    def _normalize_item_name(self, item_name: str) -> str:
        """Normalize item names for comparison"""
        # Remove provider prefixes and normalize
        if '::' in item_name:
            item_name = item_name.split('::', 1)[1]
        
        return item_name.lower().strip()
    
    def _extract_confidence_from_result(self, result: Dict[str, Any], item_name: str) -> float:
        """Extract confidence score from algorithm result"""
        # Try different ways to extract confidence
        item_data = result.get('consensus_items', {}).get(item_name, {})
        
        # Look for various confidence indicators
        confidence_keys = ['consensus_score', 'confidence', 'belief', 'score', 'weight']
        
        for key in confidence_keys:
            if key in item_data and isinstance(item_data[key], (int, float)):
                return float(item_data[key])
        
        # Default confidence based on algorithm success
        if 'error' not in result:
            return 0.7
        else:
            return 0.3
    
    def _assess_consensus_quality(self, 
                                algorithm_results: Dict[str, Any],
                                final_consensus: Dict[str, Any],
                                target_structure: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of consensus results"""
        
        quality_metrics = {
            'overall_score': 0.0,
            'algorithm_agreement': 0.0,
            'coverage': 0.0,
            'confidence': 0.0,
            'uncertainty': 0.0,
            'consistency': 0.0
        }
        
        try:
            # Algorithm agreement - how many algorithms produced results
            successful_algorithms = len([r for r in algorithm_results.values() if 'error' not in r])
            total_algorithms = len(algorithm_results)
            
            if total_algorithms > 0:
                quality_metrics['algorithm_agreement'] = successful_algorithms / total_algorithms
            
            # Coverage - how many items achieved consensus
            total_possible_items = target_structure.get('item_count_range', (0, 0))[1]
            consensus_items = len(final_consensus)
            
            if total_possible_items > 0:
                quality_metrics['coverage'] = min(1.0, consensus_items / total_possible_items)
            
            # Confidence - average confidence of consensus items
            if final_consensus:
                confidences = [item.get('consensus_score', 0) for item in final_consensus.values()]
                quality_metrics['confidence'] = np.mean(confidences)
            
            # Uncertainty - extract from Dempster-Shafer if available
            ds_result = algorithm_results.get('dempster_shafer', {})
            if 'overall_uncertainty' in ds_result:
                overall_uncertainty = ds_result['overall_uncertainty']
                avg_uncertainty = overall_uncertainty.get('average_uncertainty', 0)
                # Convert uncertainty to quality (lower uncertainty = higher quality)
                quality_metrics['uncertainty'] = max(0.0, 1.0 - avg_uncertainty / 3.0)
            else:
                quality_metrics['uncertainty'] = 0.5  # Default
            
            # Consistency - how consistent are the results across algorithms
            consistency_scores = []
            for item_name, item_info in final_consensus.items():
                algorithm_scores = list(item_info.get('algorithm_scores', {}).values())
                if len(algorithm_scores) > 1:
                    # Standard deviation of scores (lower = more consistent)
                    consistency = 1.0 - min(1.0, np.std(algorithm_scores))
                    consistency_scores.append(consistency)
            
            if consistency_scores:
                quality_metrics['consistency'] = np.mean(consistency_scores)
            else:
                quality_metrics['consistency'] = 1.0  # Perfect consistency if only one algorithm
            
            # Overall score as weighted average
            weights = {'algorithm_agreement': 0.2, 'coverage': 0.2, 'confidence': 0.3, 'uncertainty': 0.15, 'consistency': 0.15}
            
            quality_metrics['overall_score'] = sum(
                quality_metrics[metric] * weight 
                for metric, weight in weights.items()
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return quality_metrics
    
    def batch_consensus(self, 
                       provider_results: Dict[str, Dict],
                       target_paths: List[str],
                       consensus_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Run consensus on multiple target paths in batch
        
        Args:
            provider_results: Dictionary of provider results
            target_paths: List of target key paths to process
            consensus_type: Type of consensus to run
            
        Returns:
            Batch consensus results
        """
        batch_results = {
            'target_paths': target_paths,
            'consensus_type': consensus_type,
            'results': {},
            'summary': {}
        }
        
        # Process each target path
        for target_path in target_paths:
            logger.info(f"Processing batch target: {target_path}")
            
            result = self.unified_consensus(
                provider_results, target_path, consensus_type
            )
            
            batch_results['results'][target_path] = result
        
        # Generate batch summary
        batch_results['summary'] = self._generate_batch_summary(batch_results['results'])
        
        logger.info(f"Batch consensus completed for {len(target_paths)} targets")
        
        return batch_results
    
    def _generate_batch_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""
        summary = {
            'total_targets': len(batch_results),
            'successful_targets': 0,
            'failed_targets': 0,
            'average_quality_score': 0.0,
            'total_consensus_items': 0,
            'average_processing_time': 0.0
        }
        
        quality_scores = []
        processing_times = []
        
        for target_path, result in batch_results.items():
            if 'error' in result:
                summary['failed_targets'] += 1
            else:
                summary['successful_targets'] += 1
                
                quality_score = result.get('consensus_quality', {}).get('overall_score', 0)
                quality_scores.append(quality_score)
                
                processing_time = result.get('processing_time', 0)
                processing_times.append(processing_time)
                
                consensus_count = len(result.get('final_consensus', {}))
                summary['total_consensus_items'] += consensus_count
        
        if quality_scores:
            summary['average_quality_score'] = np.mean(quality_scores)
        
        if processing_times:
            summary['average_processing_time'] = np.mean(processing_times)
        
        return summary
    
    def pattern_based_consensus(self, 
                               provider_results: Dict[str, Dict],
                               pattern_paths: List[str] = None,
                               consensus_type: str = 'comprehensive',
                               custom_similarity_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        Specialized consensus for any pattern types using Universal Consensus Engine
        
        Args:
            provider_results: Dictionary of provider results
            pattern_paths: List of JSON paths to different pattern types (e.g., ['observations.behavioral_patterns.malicious'])
            consensus_type: Type of consensus ('fast', 'comprehensive', 'uncertainty_aware')
            custom_similarity_func: Custom similarity function for patterns
            
        Returns:
            Pattern-specific consensus results
        """
        start_time = time.time()
        
        try:
            if pattern_paths is None:
                # Default pattern paths based on example files
                pattern_paths = [
                    'observations.behavioral_patterns.malicious',
                    'observations.temporal_patterns.malicious',
                    'detection_patterns',
                    'framework_mappings.mitre_attack',
                    'indicators_of_compromise',
                    'soc_playbook'
                ]
            
            logger.info(f"Starting pattern-based consensus for {len(pattern_paths)} pattern paths")
            
            # Use Universal Consensus Engine for batch processing
            batch_results = self.universal_engine.batch_consensus(
                provider_results=provider_results,
                target_paths=pattern_paths,
                consensus_strength=self._map_consensus_type_to_strength(consensus_type)
            )
            
            # Process results into pattern-specific format
            pattern_consensus = {}
            unified_patterns = {}
            total_consensus_patterns = 0
            
            for path, result in batch_results.items():
                if hasattr(result, 'final_consensus_items') and result.final_consensus_items:
                    # Convert consensus items to dictionary format
                    consensus_dict = {}
                    for item in result.final_consensus_items:
                        consensus_dict[item.item_id] = {
                            'content': item.content,
                            'confidence_score': item.confidence_score,
                            'source_models': item.source_models,
                            'metadata': item.metadata
                        }
                    
                    pattern_consensus[path] = {
                        'target_path': path,
                        'consensus_items': consensus_dict,
                        'cluster_count': result.cluster_count,
                        'compression_ratio': result.compression_ratio,
                        'quality_metrics': result.consensus_quality_metrics
                    }
                    
                    unified_patterns[path] = consensus_dict
                    total_consensus_patterns += len(consensus_dict)
            
            # Create domain-specific similarity function for patterns if not provided
            if custom_similarity_func is None:
                custom_similarity_func = self._create_pattern_similarity_function()
            
            # Import utilities for cross-pattern merging
            from utils import merge_similar_patterns
            
            # Merge similar patterns across different paths
            all_pattern_items = []
            for path, consensus_items in unified_patterns.items():
                for item_id, item_info in consensus_items.items():
                    pattern_item = item_info['content'].copy() if isinstance(item_info['content'], dict) else item_info['content']
                    if isinstance(pattern_item, dict):
                        pattern_item['_source_path'] = path
                        pattern_item['_item_id'] = item_id
                        pattern_item['_confidence_score'] = item_info['confidence_score']
                        pattern_item['_source_models'] = item_info['source_models']
                    all_pattern_items.append(pattern_item)
            
            # Merge similar patterns using enhanced similarity
            final_merged_patterns = merge_similar_patterns(all_pattern_items, similarity_threshold=0.75)
            
            # Calculate overall quality metrics
            overall_quality = self._calculate_universal_pattern_quality(pattern_consensus, pattern_paths)
            
            return {
                'consensus_type': 'universal_pattern_based',
                'pattern_paths': pattern_paths,
                'provider_count': len(provider_results),
                'pattern_consensus_by_path': pattern_consensus,
                'unified_patterns_by_path': unified_patterns,
                'final_merged_patterns': final_merged_patterns,
                'total_consensus_patterns': total_consensus_patterns,
                'final_merged_count': len(final_merged_patterns),
                'overall_quality': overall_quality,
                'batch_processing_results': batch_results,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Universal pattern-based consensus failed: {e}")
            return {
                'error': str(e),
                'consensus_type': 'universal_pattern_based',
                'pattern_paths': pattern_paths,
                'processing_time': time.time() - start_time
            }
    
    def _calculate_pattern_consensus_quality(self, 
                                           pattern_consensus: Dict[str, Any],
                                           pattern_candidates: Dict[str, List]) -> Dict[str, float]:
        """Calculate quality metrics for pattern consensus"""
        
        quality_metrics = {
            'overall_score': 0.0,
            'pattern_type_coverage': 0.0,
            'average_pattern_quality': 0.0,
            'consensus_efficiency': 0.0,
            'provider_agreement': 0.0
        }
        
        try:
            # Pattern type coverage
            successful_types = len([r for r in pattern_consensus.values() if 'error' not in r])
            total_types = len(pattern_consensus)
            
            if total_types > 0:
                quality_metrics['pattern_type_coverage'] = successful_types / total_types
            
            # Average pattern quality across types
            pattern_qualities = []
            for pattern_type, result in pattern_consensus.items():
                if 'error' not in result and 'consensus_quality' in result:
                    quality_score = result['consensus_quality'].get('overall_score', 0)
                    pattern_qualities.append(quality_score)
            
            if pattern_qualities:
                quality_metrics['average_pattern_quality'] = sum(pattern_qualities) / len(pattern_qualities)
            
            # Consensus efficiency (how much data was unified)
            total_candidates = sum(len(candidates) for candidates in pattern_candidates.values())
            total_consensus = sum(
                len(result.get('final_consensus', {})) 
                for result in pattern_consensus.values() 
                if 'error' not in result
            )
            
            if total_candidates > 0:
                quality_metrics['consensus_efficiency'] = total_consensus / total_candidates
            
            # Provider agreement (how many providers contributed to patterns)
            contributing_providers = set()
            for candidates in pattern_candidates.values():
                for candidate in candidates:
                    contributing_providers.update(candidate.get('providers', []))
            
            if contributing_providers:
                quality_metrics['provider_agreement'] = len(contributing_providers) / max(1, total_candidates / 10)
            
            # Overall score as weighted average
            weights = {
                'pattern_type_coverage': 0.3,
                'average_pattern_quality': 0.4,
                'consensus_efficiency': 0.2,
                'provider_agreement': 0.1
            }
            
            quality_metrics['overall_score'] = sum(
                quality_metrics[metric] * weight 
                for metric, weight in weights.items()
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Pattern quality calculation failed: {e}")
            return quality_metrics
    
    def multi_target_consensus(self, 
                             provider_results: Dict[str, Dict],
                             target_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run consensus on multiple targets with different configurations
        
        Args:
            provider_results: Provider results
            target_configs: List of target configurations
                Each config: {'path': 'target.path', 'type': 'comprehensive', 'algorithms': [...]}
            
        Returns:
            Multi-target consensus results
        """
        multi_results = {
            'target_configs': target_configs,
            'provider_count': len(provider_results),
            'results': {},
            'summary': {},
            'processing_time': 0,
            'timestamp': time.time()
        }
        
        start_time = time.time()
        
        for i, config in enumerate(target_configs):
            target_path = config.get('path')
            consensus_type = config.get('type', 'comprehensive')
            algorithms = config.get('algorithms')
            
            logger.info(f"Processing multi-target consensus {i+1}/{len(target_configs)}: {target_path}")
            
            result = self.unified_consensus(
                provider_results, target_path, consensus_type, algorithms
            )
            
            multi_results['results'][target_path] = result
        
        # Generate summary
        multi_results['processing_time'] = time.time() - start_time
        multi_results['summary'] = self._generate_multi_target_summary(multi_results['results'])
        
        logger.info(f"Multi-target consensus completed for {len(target_configs)} targets")
        return multi_results
    
    def _generate_multi_target_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for multi-target consensus"""
        
        summary = {
            'total_targets': len(results),
            'successful_targets': 0,
            'failed_targets': 0,
            'total_consensus_items': 0,
            'average_quality_score': 0.0,
            'quality_by_target': {},
            'processing_times': {}
        }
        
        quality_scores = []
        
        for target_path, result in results.items():
            summary['processing_times'][target_path] = result.get('processing_time', 0)
            
            if 'error' in result:
                summary['failed_targets'] += 1
            else:
                summary['successful_targets'] += 1
                
                consensus_items = len(result.get('final_consensus', {}))
                summary['total_consensus_items'] += consensus_items
                
                quality_score = result.get('consensus_quality', {}).get('overall_score', 0)
                quality_scores.append(quality_score)
                summary['quality_by_target'][target_path] = quality_score
        
        if quality_scores:
            summary['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        return summary
    
    def _convert_consensus_items_to_dict(self, consensus_items: List) -> Dict[str, Any]:
        """Convert list of ConsensusItem objects to dictionary format"""
        result_dict = {}
        
        for item in consensus_items:
            result_dict[item.item_id] = {
                'content': item.content,
                'confidence_score': item.confidence_score,
                'source_models': item.source_models,
                'similarity_scores': item.similarity_scores,
                'metadata': item.metadata
            }
        
        return result_dict
    
    def _map_consensus_type_to_strength(self, consensus_type: str) -> 'ConsensusStrength':
        """Map consensus type string to ConsensusStrength enum"""
        mapping = {
            'fast': ConsensusStrength.WEAK,
            'comprehensive': ConsensusStrength.STRONG,
            'uncertainty_aware': ConsensusStrength.MAXIMUM
        }
        return mapping.get(consensus_type, ConsensusStrength.STRONG)
    
    def _create_pattern_similarity_function(self) -> callable:
        """Create domain-specific similarity function for security patterns"""
        
        def pattern_similarity(pattern1: Any, pattern2: Any) -> float:
            """Calculate similarity between security patterns"""
            if not isinstance(pattern1, dict) or not isinstance(pattern2, dict):
                return 0.0
            
            similarity = 0.0
            
            # Pattern name/title similarity (high weight)
            name_fields = ['pattern_name', 'name', 'title', 'detection_name']
            for field in name_fields:
                if field in pattern1 and field in pattern2:
                    name1 = str(pattern1[field]).lower()
                    name2 = str(pattern2[field]).lower()
                    if name1 == name2:
                        similarity += 0.4
                    elif any(word in name2 for word in name1.split()) or any(word in name1 for word in name2.split()):
                        similarity += 0.2
                    break
            
            # Attack technique similarity
            if 'mapped_ttp' in pattern1 and 'mapped_ttp' in pattern2:
                if pattern1['mapped_ttp'] == pattern2['mapped_ttp']:
                    similarity += 0.3
            
            # Description similarity
            desc_fields = ['description', 'Instruction', 'instruction', 'details']
            for field in desc_fields:
                if field in pattern1 and field in pattern2:
                    desc1_words = set(str(pattern1[field]).lower().split())
                    desc2_words = set(str(pattern2[field]).lower().split())
                    if desc1_words and desc2_words:
                        desc_similarity = len(desc1_words & desc2_words) / len(desc1_words | desc2_words)
                        similarity += desc_similarity * 0.2
                    break
            
            # Field overlap similarity
            if 'identifiable_fields' in pattern1 and 'identifiable_fields' in pattern2:
                fields1 = set(pattern1['identifiable_fields']) if isinstance(pattern1['identifiable_fields'], list) else set()
                fields2 = set(pattern2['identifiable_fields']) if isinstance(pattern2['identifiable_fields'], list) else set()
                if fields1 and fields2:
                    field_overlap = len(fields1 & fields2) / len(fields1 | fields2)
                    similarity += field_overlap * 0.1
            
            return min(1.0, similarity)
        
        return pattern_similarity
    
    def _calculate_universal_pattern_quality(self, 
                                           pattern_consensus: Dict[str, Any],
                                           pattern_paths: List[str]) -> Dict[str, float]:
        """Calculate quality metrics for universal pattern consensus"""
        
        quality_metrics = {
            'overall_score': 0.0,
            'path_coverage': 0.0,
            'average_confidence': 0.0,
            'compression_efficiency': 0.0,
            'model_diversity': 0.0
        }
        
        try:
            # Path coverage - how many paths produced consensus
            successful_paths = len([pc for pc in pattern_consensus.values() if 'consensus_items' in pc])
            if pattern_paths:
                quality_metrics['path_coverage'] = successful_paths / len(pattern_paths)
            
            # Average confidence across all consensus items
            all_confidences = []
            all_source_models = set()
            
            for path_result in pattern_consensus.values():
                if 'consensus_items' in path_result:
                    for item_info in path_result['consensus_items'].values():
                        if 'confidence_score' in item_info:
                            all_confidences.append(item_info['confidence_score'])
                        if 'source_models' in item_info:
                            all_source_models.update(item_info['source_models'])
            
            if all_confidences:
                quality_metrics['average_confidence'] = sum(all_confidences) / len(all_confidences)
            
            # Model diversity
            if all_source_models:
                quality_metrics['model_diversity'] = min(1.0, len(all_source_models) / 5)  # Assume max 5 models
            
            # Compression efficiency - placeholder calculation
            quality_metrics['compression_efficiency'] = 0.7  # Default reasonable value
            
            # Overall score as weighted average
            weights = {
                'path_coverage': 0.3,
                'average_confidence': 0.3,
                'compression_efficiency': 0.2,
                'model_diversity': 0.2
            }
            
            quality_metrics['overall_score'] = sum(
                quality_metrics[metric] * weight
                for metric, weight in weights.items()
            )
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Universal pattern quality calculation failed: {e}")
            return quality_metrics
    
    def register_llm_arbitrator(self, provider_name: str, arbitrator_instance: Any):
        """Register LLM provider for conflict arbitration"""
        self.universal_engine.register_llm_arbitrator(provider_name, arbitrator_instance)
        logger.info(f"Registered LLM arbitrator in orchestrator: {provider_name}")
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get statistics about all consensus operations"""
        return {
            'total_consensus_operations': len(self.consensus_results),
            'recent_results': list(self.consensus_results.keys())[-5:],
            'algorithm_weights': self.algorithm_weights,
            'config': self.config
        }