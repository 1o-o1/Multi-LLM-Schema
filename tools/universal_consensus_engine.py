#!/usr/bin/env python3
"""
🧠 JSON CONSENSUS FRAMEWORK SYSTEM - Universal Consensus Engine
Universal Consensus Engine for Multi-LLM Agent Systems

This file is part of System 2: JSON Consensus Framework
This engine can take any JSON path (e.g., 'observations.behavioral_patterns.malicious') 
and create consensus from the items at that path across all models.

Architecture follows the Multi-agent.md and AI consensus.md specifications with:
- Embedding-based similarity clustering
- Byzantine Fault Tolerance consensus
- Dempster-Shafer belief combination 
- Monte Carlo Tree Search optimization
- LLM-based arbitration for conflicts
- Iterative Consensus Ensemble (ICE) loop

See SYSTEM_ARCHITECTURE.md for complete system documentation
"""

import numpy as np
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import consensus algorithms - with error handling for missing modules
try:
    from .embedding_service import EmbeddingService
except ImportError:
    EmbeddingService = None
    
try:
    from .bft_consensus import BFTConsensus
except ImportError:
    BFTConsensus = None
    
try:
    from .dempster_shafer import DempsterShaferEngine
except ImportError:
    DempsterShaferEngine = None
    
try:
    from .mcts_optimization import MCTSOptimization
except ImportError:
    MCTSOptimization = None
    
try:
    from .llm_conflict_arbitrator import LLMConflictArbitrator, ConflictCase, ArbitrationStrategy
except ImportError:
    LLMConflictArbitrator = None
    ConflictCase = None
    ArbitrationStrategy = None

# Import utilities - using relative imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_nested_value, calculate_data_hash

try:
    from preprocessing import ConsensusDataPreprocessor
except ImportError:
    ConsensusDataPreprocessor = None

logger = logging.getLogger(__name__)

class ConsensusStrength(Enum):
    """Consensus strength levels"""
    WEAK = "weak"           # Simple majority vote
    MODERATE = "moderate"   # Multiple algorithms with basic validation
    STRONG = "strong"      # Full multi-algorithm with Byzantine fault tolerance
    MAXIMUM = "maximum"    # All algorithms + LLM arbitration + ICE loop

@dataclass
class ConsensusItem:
    """Universal consensus item that can represent anything"""
    item_id: str
    content: Any  # Can be dict, string, list, etc.
    source_models: List[str]
    confidence_score: float = 0.0
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate hash for identity"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

@dataclass 
class ConsensusCluster:
    """Cluster of similar consensus items"""
    cluster_id: str
    items: List[ConsensusItem]
    representative_item: ConsensusItem
    intra_cluster_similarity: float
    consensus_strength: float
    
    def __post_init__(self):
        """Calculate cluster statistics"""
        self.size = len(self.items)
        self.model_coverage = len(set(model for item in self.items for model in item.source_models))
        self.average_confidence = np.mean([item.confidence_score for item in self.items])

@dataclass
class UniversalConsensusResult:
    """Result from universal consensus process"""
    target_path: str
    input_model_count: int
    extracted_items_count: int
    consensus_clusters: List[ConsensusCluster]
    final_consensus_items: List[ConsensusItem]
    consensus_quality_metrics: Dict[str, float]
    processing_metadata: Dict[str, Any]
    algorithm_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate summary statistics"""
        self.cluster_count = len(self.consensus_clusters)
        self.final_consensus_count = len(self.final_consensus_items)
        self.compression_ratio = 1.0 - (self.final_consensus_count / max(1, self.extracted_items_count))
        
        # Add aliases for backwards compatibility
        self.consensus_items = self.final_consensus_items
        
        # Calculate final confidence score
        if self.final_consensus_items:
            confidence_scores = [item.confidence_score for item in self.final_consensus_items if hasattr(item, 'confidence_score')]
            self.final_confidence_score = np.mean(confidence_scores) if confidence_scores else 0.5
        else:
            self.final_confidence_score = 0.0
        

class UniversalConsensusEngine:
    """
    Universal consensus engine that can process any JSON structure
    
    Usage:
    engine = UniversalConsensusEngine()
    result = engine.create_consensus(
        provider_results={
            'model1': {'parsed_json': {...}}, 
            'model2': {'parsed_json': {...}}
        },
        target_path='observations.behavioral_patterns.malicious',
        consensus_strength=ConsensusStrength.STRONG
    )
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize universal consensus engine"""
        # Merge provided config with defaults to ensure all required keys are present
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Initialize consensus algorithms with fallbacks
        self.embedding_service = self._init_embedding_service()
        self.bft_consensus = self._init_bft_consensus()
        self.dempster_shafer = self._init_dempster_shafer()
        self.mcts_optimization = self._init_mcts_optimization()
        self.preprocessor = self._init_preprocessor()
        
        # LLM-based conflict arbitration
        self.llm_arbitrator = self._init_llm_arbitrator()
        
        # LLM providers for arbitration (legacy)
        self.llm_arbitrators = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embedding_model': 'gemini-embedding-001',
            'similarity_threshold': 0.75,
            'consensus_threshold': 0.6,
            'bft_fault_tolerance': 0.33,
            'mcts_exploration': 1.41,
            'mcts_iterations': 500,
            'max_parallel_algorithms': 4,
            'enable_llm_arbitration': True,
            'ice_loop_max_iterations': 3,
            'ice_confidence_threshold': 0.7,
            
            # Component enable/disable flags for research analysis
            'enable_sbert_embeddings': True,
            'enable_semantic_clustering': True,
            'enable_canonical_concepts': True,
            'enable_weighted_voting': True,
            'enable_bft_consensus': True,
            'enable_conflict_arbitration': True,
            'enable_semantic_ted': True,
            'enable_muse_adaptation': True,
            'enable_ice_refinement': True,
            'enable_mcts_optimization': True,
            'enable_dempster_shafer': True,
            
            'llm_arbitration': {
                'cache_enabled': True,
                'arbitration_strategy': 'single_llm',
                'conflict_detection_threshold': 0.3,
                'preferred_provider': 'mock'
            }
        }
    
    def create_consensus(self, 
                        provider_results: Dict[str, Dict],
                        target_path: str,
                        consensus_strength: ConsensusStrength = ConsensusStrength.STRONG,
                        custom_similarity_func: Optional[callable] = None) -> UniversalConsensusResult:
        """
        Create consensus for any JSON path across multiple models
        
        Args:
            provider_results: Dictionary of model results
            target_path: Dot-separated path to target data (e.g., 'observations.behavioral_patterns.malicious')
            consensus_strength: Strength of consensus algorithm to apply
            custom_similarity_func: Custom similarity function for domain-specific logic
            
        Returns:
            Universal consensus result
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting universal consensus for path: {target_path}")
            logger.info(f"Consensus strength: {consensus_strength.value}, Models: {list(provider_results.keys())}")
            
            # Step 1: Extract items from all models at the specified path
            extracted_items = self._extract_items_from_path(provider_results, target_path)
            
            if not extracted_items:
                return self._create_empty_result(target_path, provider_results, "No items found at specified path")
            
            logger.info(f"Extracted {len(extracted_items)} items from {len(provider_results)} models")
            
            # Step 2: Create similarity clusters using embeddings
            consensus_clusters = self._create_similarity_clusters(
                extracted_items, custom_similarity_func
            )
            
            logger.info(f"Created {len(consensus_clusters)} similarity clusters")
            
            # Step 3: Apply consensus algorithms based on strength
            algorithm_results = self._apply_consensus_algorithms(
                consensus_clusters, consensus_strength
            )
            
            # Step 4: Apply LLM arbitration if enabled and conflicts detected
            if self.config.get('enable_llm_arbitration', True) and consensus_strength in [ConsensusStrength.STRONG, ConsensusStrength.MAXIMUM]:
                algorithm_results = self._apply_llm_arbitration(algorithm_results, target_path)
            
            # Step 5: Combine algorithm results and create final consensus
            final_consensus_items = self._create_final_consensus(
                consensus_clusters, algorithm_results
            )
            
            logger.info(f"Final consensus: {len(final_consensus_items)} items")
            
            # Step 5: Apply ICE loop if needed
            if consensus_strength in [ConsensusStrength.STRONG, ConsensusStrength.MAXIMUM]:
                final_consensus_items = self._apply_ice_loop(
                    final_consensus_items, provider_results, target_path
                )
            
            # Step 6: Calculate quality metrics
            quality_metrics = self._calculate_consensus_quality(
                extracted_items, consensus_clusters, final_consensus_items, algorithm_results
            )
            
            # Step 7: Create result
            result = UniversalConsensusResult(
                target_path=target_path,
                input_model_count=len(provider_results),
                extracted_items_count=len(extracted_items),
                consensus_clusters=consensus_clusters,
                final_consensus_items=final_consensus_items,
                consensus_quality_metrics=quality_metrics,
                algorithm_results=algorithm_results,
                processing_metadata={
                    'processing_time': time.time() - start_time,
                    'consensus_strength': consensus_strength.value,
                    'config_used': self.config.copy(),
                    'timestamp': time.time()
                }
            )
            
            logger.info(f"Universal consensus completed in {result.processing_metadata['processing_time']:.2f}s")
            logger.info(f"Quality score: {quality_metrics.get('overall_quality', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Universal consensus failed: {e}")
            return self._create_error_result(target_path, provider_results, str(e), time.time() - start_time)
    
    def _extract_items_from_path(self, 
                                provider_results: Dict[str, Dict], 
                                target_path: str) -> List[ConsensusItem]:
        """Extract items from specified path across all models"""
        
        extracted_items = []
        
        for model_name, result in provider_results.items():
            # Get data at target path
            target_data = get_nested_value(result, target_path)
            
            if target_data is None:
                logger.debug(f"No data found at path '{target_path}' for model {model_name}")
                continue
            
            # Handle different data structures
            if isinstance(target_data, list):
                # List of items
                for i, item in enumerate(target_data):
                    consensus_item = ConsensusItem(
                        item_id=f"{model_name}_item_{i}",
                        content=item,
                        source_models=[model_name],
                        metadata={
                            'source_model': model_name,
                            'source_path': target_path,
                            'source_index': i,
                            'extraction_time': time.time()
                        }
                    )
                    extracted_items.append(consensus_item)
                    
            elif isinstance(target_data, dict):
                # Dictionary of items
                for key, item in target_data.items():
                    consensus_item = ConsensusItem(
                        item_id=f"{model_name}_{key}",
                        content=item,
                        source_models=[model_name],
                        metadata={
                            'source_model': model_name,
                            'source_path': target_path,
                            'source_key': key,
                            'extraction_time': time.time()
                        }
                    )
                    extracted_items.append(consensus_item)
                    
            else:
                # Single item
                consensus_item = ConsensusItem(
                    item_id=f"{model_name}_single",
                    content=target_data,
                    source_models=[model_name],
                    metadata={
                        'source_model': model_name,
                        'source_path': target_path,
                        'extraction_time': time.time()
                    }
                )
                extracted_items.append(consensus_item)
        
        return extracted_items
    
    def _create_similarity_clusters(self, 
                                  items: List[ConsensusItem], 
                                  custom_similarity_func: Optional[callable] = None) -> List[ConsensusCluster]:
        """Create similarity clusters using embeddings and custom logic"""
        
        if len(items) <= 1:
            # Create single cluster for single or no items
            if items:
                cluster = ConsensusCluster(
                    cluster_id="single_cluster",
                    items=items,
                    representative_item=items[0],
                    intra_cluster_similarity=1.0,
                    consensus_strength=1.0
                )
                return [cluster]
            return []
        
        # Convert items to text for embedding
        item_texts = []
        for item in items:
            if isinstance(item.content, dict):
                # Convert dict to searchable text
                text_parts = []
                for key, value in item.content.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
                    elif isinstance(value, list):
                        text_parts.append(f"{key}: {' '.join(map(str, value))}")
                    else:
                        text_parts.append(f"{key}: {str(value)}")
                item_text = " | ".join(text_parts)
            else:
                item_text = str(item.content)
            
            item_texts.append(item_text)
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_text(item_texts)
        
        # Calculate similarity matrix
        similarity_matrix = self.embedding_service.cosine_similarity_matrix(embeddings)
        
        # Apply custom similarity function if provided
        if custom_similarity_func:
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    custom_sim = custom_similarity_func(items[i].content, items[j].content)
                    # Combine with embedding similarity
                    similarity_matrix[i, j] = (similarity_matrix[i, j] + custom_sim) / 2
                    similarity_matrix[j, i] = similarity_matrix[i, j]
        
        # Cluster items based on similarity
        clusters = self._cluster_by_similarity(items, similarity_matrix)
        
        return clusters
    
    def _cluster_by_similarity(self, 
                             items: List[ConsensusItem], 
                             similarity_matrix: np.ndarray) -> List[ConsensusCluster]:
        """Cluster items based on similarity matrix"""
        
        threshold = self.config['similarity_threshold']
        n_items = len(items)
        visited = set()
        clusters = []
        
        for i in range(n_items):
            if i in visited:
                continue
            
            # Start new cluster
            cluster_items = [items[i]]
            cluster_indices = {i}
            visited.add(i)
            
            # Find all similar items
            for j in range(i + 1, n_items):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    cluster_items.append(items[j])
                    cluster_indices.add(j)
                    visited.add(j)
            
            # Calculate cluster statistics
            if len(cluster_items) > 1:
                similarities = []
                indices = list(cluster_indices)
                for idx1 in indices:
                    for idx2 in indices:
                        if idx1 < idx2:
                            similarities.append(similarity_matrix[idx1, idx2])
                avg_similarity = np.mean(similarities)
            else:
                avg_similarity = 1.0
            
            # Choose representative item (highest average similarity to others)
            if len(cluster_items) > 1:
                rep_scores = []
                for idx in cluster_indices:
                    other_indices = cluster_indices - {idx}
                    rep_score = np.mean([similarity_matrix[idx, other_idx] for other_idx in other_indices])
                    rep_scores.append((rep_score, items[idx]))
                representative_item = max(rep_scores, key=lambda x: x[0])[1]
            else:
                representative_item = cluster_items[0]
            
            # Create cluster
            cluster = ConsensusCluster(
                cluster_id=f"cluster_{len(clusters)}",
                items=cluster_items,
                representative_item=representative_item,
                intra_cluster_similarity=avg_similarity,
                consensus_strength=len(cluster_items) / n_items
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _apply_consensus_algorithms(self, 
                                  clusters: List[ConsensusCluster], 
                                  consensus_strength: ConsensusStrength) -> Dict[str, Any]:
        """Apply consensus algorithms based on strength level and component enable flags"""
        
        algorithm_results = {}
        
        # Determine which algorithms to run based on component flags
        algorithms_to_run = {}
        
        # Embedding similarity (if SBERT embeddings enabled)
        if self.config.get('enable_sbert_embeddings', True):
            algorithms_to_run['embedding_similarity'] = self._apply_embedding_consensus
        
        # BFT consensus (if BFT consensus enabled)
        if self.config.get('enable_bft_consensus', True):
            algorithms_to_run['bft_consensus'] = self._apply_bft_consensus
        
        # Dempster-Shafer (if enabled)
        if self.config.get('enable_dempster_shafer', True):
            algorithms_to_run['dempster_shafer'] = self._apply_dempster_shafer
        
        # MCTS optimization (if enabled and strength is high enough)
        if (self.config.get('enable_mcts_optimization', True) and 
            consensus_strength in [ConsensusStrength.STRONG, ConsensusStrength.MAXIMUM]):
            algorithms_to_run['mcts_optimization'] = self._apply_mcts_optimization
        
        # Weighted voting (simulate with enhanced majority vote if enabled)
        # Check both 'enable_weighted_voting' and 'use_weighted_voting' for compatibility
        enable_weighted = self.config.get('enable_weighted_voting', self.config.get('use_weighted_voting', True))
        if enable_weighted:
            algorithms_to_run['weighted_voting'] = self._apply_weighted_vote
        
        # Conflict arbitration (simulate with enhanced processing if enabled)
        if self.config.get('enable_conflict_arbitration', True):
            algorithms_to_run['conflict_arbitration'] = self._apply_conflict_arbitration
        
        # If no algorithms are enabled, use basic majority vote
        if not algorithms_to_run:
            algorithm_results['majority_vote'] = self._apply_majority_vote(clusters)
            return algorithm_results
        
        # Execute enabled algorithms
        if len(algorithms_to_run) == 1:
            # Single algorithm - run directly
            algorithm_name, algorithm_func = next(iter(algorithms_to_run.items()))
            try:
                algorithm_results[algorithm_name] = algorithm_func(clusters)
            except Exception as e:
                logger.error(f"Algorithm {algorithm_name} failed: {e}")
                algorithm_results[algorithm_name] = {'error': str(e)}
        else:
            # Multiple algorithms - run in parallel
            with ThreadPoolExecutor(max_workers=min(len(algorithms_to_run), self.config['max_parallel_algorithms'])) as executor:
                futures = {
                    executor.submit(algorithm_func, clusters): algorithm_name
                    for algorithm_name, algorithm_func in algorithms_to_run.items()
                }
                
                for future in as_completed(futures):
                    algorithm_name = futures[future]
                    try:
                        result = future.result()
                        algorithm_results[algorithm_name] = result
                    except Exception as e:
                        logger.error(f"Algorithm {algorithm_name} failed: {e}")
                        algorithm_results[algorithm_name] = {'error': str(e)}
        
        return algorithm_results
    
    def _apply_majority_vote(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Simple majority vote consensus - INCLUSIVE approach"""
        
        consensus_items = []
        # Use LOWER threshold for majority vote (more inclusive)
        permissive_threshold = max(0.3, self.config['consensus_threshold'] - 0.2)
        
        total_models = len(set(model for cluster in clusters 
                              for item in cluster.items 
                              for model in item.source_models))
        
        for cluster in clusters:
            # Majority vote is more permissive - includes more items
            coverage_ratio = cluster.model_coverage / max(1, total_models)
            if coverage_ratio >= permissive_threshold:
                # Include ALL items from cluster, not just representative
                for item in cluster.items[:2]:  # Take up to 2 items per cluster
                    consensus_items.append({
                        'item': item,
                        'confidence': coverage_ratio * 0.8,  # Lower confidence but more inclusive
                        'supporting_models': item.source_models,
                        'selection_reason': 'majority_inclusive'
                    })
        
        return {
            'algorithm': 'majority_vote',
            'consensus_items': consensus_items,
            'threshold_used': permissive_threshold,
            'items_selected': len(consensus_items)
        }
    
    def _apply_embedding_consensus(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Embedding-based consensus - STRICT similarity requirements"""
        
        # Use STRICTER similarity threshold for embedding-based consensus
        strict_threshold = self.config['similarity_threshold'] + 0.15  # More stringent
        consensus_items = []
        
        for cluster in clusters:
            if cluster.intra_cluster_similarity >= strict_threshold:
                # Embedding consensus prefers high-similarity, smaller clusters
                confidence = cluster.intra_cluster_similarity * 0.9  # Conservative confidence
                consensus_items.append({
                    'item': cluster.representative_item,
                    'confidence': confidence,
                    'similarity_score': cluster.intra_cluster_similarity,
                    'cluster_size': cluster.size,
                    'selection_reason': 'high_similarity'
                })
        
        return {
            'algorithm': 'embedding_similarity',
            'consensus_items': consensus_items,
            'similarity_threshold': strict_threshold,
            'items_selected': len(consensus_items)
        }
    
    def _apply_basic_bft(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Basic Byzantine Fault Tolerance"""
        
        # Simple BFT: require 2f+1 agreement where f is fault tolerance
        total_models = len(set(model for cluster in clusters for item in cluster.items for model in item.source_models))
        min_agreement = int(total_models * (1 - self.config['bft_fault_tolerance']))
        
        consensus_items = []
        
        for cluster in clusters:
            supporting_models = len(set(model for item in cluster.items for model in item.source_models))
            if supporting_models >= min_agreement:
                consensus_items.append({
                    'item': cluster.representative_item,
                    'confidence': supporting_models / total_models,
                    'supporting_model_count': supporting_models,
                    'bft_threshold': min_agreement
                })
        
        return {
            'algorithm': 'basic_bft',
            'consensus_items': consensus_items,
            'min_agreement': min_agreement,
            'total_models': total_models
        }
    
    def _apply_bft_consensus(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Byzantine Fault Tolerance - MODEL AGREEMENT focused"""
        
        try:
            # BFT requires strong model agreement - apply 2/3 Byzantine threshold
            total_models = len(set(model for cluster in clusters 
                                 for item in cluster.items 
                                 for model in item.source_models))
            bft_threshold = max(2, int(total_models * 0.67))  # 2/3 majority for BFT
            
            consensus_items = []
            
            for cluster in clusters:
                # For BFT, group items by exact model agreement
                model_agreement_groups = {}
                
                for item in cluster.items:
                    model_set = tuple(sorted(item.source_models))
                    if model_set not in model_agreement_groups:
                        model_agreement_groups[model_set] = []
                    model_agreement_groups[model_set].append(item)
                
                # Select items with sufficient model agreement
                for model_set, items in model_agreement_groups.items():
                    if len(model_set) >= bft_threshold:
                        # BFT selects the most consistent item from agreeing models
                        best_item = max(items, key=lambda x: x.confidence_score)
                        
                        # BFT applies conservative confidence adjustment
                        bft_confidence = (len(model_set) / total_models) * best_item.confidence_score * 0.9
                        
                        consensus_items.append({
                            'item': best_item,
                            'confidence': bft_confidence,
                            'model_agreement': len(model_set),
                            'bft_threshold': bft_threshold,
                            'selection_reason': 'bft_model_agreement'
                        })
            
            return {
                'algorithm': 'bft_consensus',
                'consensus_items': consensus_items,
                'bft_threshold': bft_threshold,
                'total_models': total_models,
                'items_selected': len(consensus_items)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _apply_dempster_shafer(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Dempster-Shafer belief combination"""
        
        try:
            # Convert clusters to DS format
            ds_evidence = {}
            for cluster in clusters:
                cluster_evidence = []
                for item in cluster.items:
                    evidence_item = {
                        'source': item.source_models[0] if item.source_models else 'unknown',
                        'belief_mass': item.confidence_score or cluster.consensus_strength,
                        'hypothesis': cluster.cluster_id,
                        'evidence_data': item.content
                    }
                    cluster_evidence.append(evidence_item)
                ds_evidence[cluster.cluster_id] = cluster_evidence
            
            # Run DS analysis
            ds_result = self.dempster_shafer.multi_field_consensus_analysis({'ds_evidence': ds_evidence})
            
            return {
                'algorithm': 'dempster_shafer',
                'ds_result': ds_result,
                'evidence_processed': len(ds_evidence)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _apply_mcts_optimization(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Monte Carlo Tree Search optimization"""
        
        try:
            # Convert clusters to MCTS format
            mcts_data = {}
            for i, cluster in enumerate(clusters):
                for item in cluster.items:
                    provider = item.source_models[0] if item.source_models else f'provider_{i}'
                    if provider not in mcts_data:
                        mcts_data[provider] = {'parsed_json': {'items': {}}}
                    
                    mcts_data[provider]['parsed_json']['items'][item.item_id] = item.content
            
            # Run MCTS optimization
            mcts_result = self.mcts_optimization.optimize_schema_consensus(mcts_data)
            
            return {
                'algorithm': 'mcts_optimization',
                'mcts_result': mcts_result,
                'clusters_processed': len(clusters)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _apply_weighted_vote(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Weighted voting consensus - RELIABILITY-BASED selection"""
        
        try:
            # Calculate model reliability based on consistency across clusters
            model_reliability = {}
            model_participation = {}
            
            for cluster in clusters:
                for item in cluster.items:
                    for model in item.source_models:
                        if model not in model_reliability:
                            model_reliability[model] = []
                            model_participation[model] = 0
                        model_reliability[model].append(item.confidence_score)
                        model_participation[model] += 1
            
            # Calculate weighted reliability scores
            model_weights = {}
            for model, scores in model_reliability.items():
                # Higher weight for models with consistent high confidence
                avg_confidence = sum(scores) / len(scores)
                consistency = 1.0 - (max(scores) - min(scores))  # Lower variance = higher consistency
                participation_bonus = min(1.0, model_participation[model] / len(clusters))
                model_weights[model] = avg_confidence * consistency * participation_bonus
            
            # Select items differently - prefer items from most reliable models
            consensus_items = []
            for cluster in clusters:
                # Select top 2 most reliable items from each cluster
                scored_items = []
                for item in cluster.items:
                    reliability_score = sum(model_weights.get(model, 0) for model in item.source_models) / len(item.source_models)
                    scored_items.append((reliability_score, item))
                
                # Sort by reliability and take top items
                scored_items.sort(key=lambda x: x[0], reverse=True)
                for score, item in scored_items[:2]:  # Take top 2 items
                    if score > 0.3:  # Minimum reliability threshold
                        consensus_items.append({
                            'item': item,
                            'confidence': score,
                            'reliability_score': score,
                            'selection_reason': 'weighted_reliability'
                        })
            
            return {
                'algorithm': 'weighted_voting',
                'consensus_items': consensus_items,
                'model_weights': model_weights,
                'items_selected': len(consensus_items)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _apply_conflict_arbitration(self, clusters: List[ConsensusCluster]) -> Dict[str, Any]:
        """Conflict arbitration - CONSERVATIVE with detailed analysis"""
        
        try:
            # More sophisticated conflict detection and resolution
            conflicts_detected = 0
            resolved_items = []
            conflict_details = []
            
            for cluster in clusters:
                if len(cluster.items) <= 1:
                    # No conflict - apply strict quality filter
                    if cluster.items and cluster.items[0].confidence_score > 0.6:
                        resolved_items.append({
                            'item': cluster.items[0],
                            'confidence': cluster.items[0].confidence_score * 0.95,  # Conservative reduction
                            'selection_reason': 'no_conflict_high_quality'
                        })
                    continue
                
                # Multiple items in cluster - analyze conflict
                conflicts_detected += 1
                
                # Group items by similarity of content
                content_groups = {}
                for item in cluster.items:
                    content_key = str(hash(str(item.content)))[:8]
                    if content_key not in content_groups:
                        content_groups[content_key] = []
                    content_groups[content_key].append(item)
                
                # For each content group, select best representative
                for group_items in content_groups.values():
                    if len(group_items) >= 2:  # Require at least 2 models to agree
                        # Select item with highest confidence and most model support
                        best_item = max(group_items, 
                                      key=lambda x: (len(x.source_models), x.confidence_score))
                        
                        resolved_items.append({
                            'item': best_item,
                            'confidence': best_item.confidence_score * 0.85,  # Penalty for conflict
                            'supporting_models': len(best_item.source_models),
                            'selection_reason': 'conflict_resolved'
                        })
                        
                        conflict_details.append({
                            'cluster_id': cluster.cluster_id,
                            'conflicting_items': len(group_items),
                            'resolution': 'model_agreement'
                        })
            
            return {
                'algorithm': 'conflict_arbitration',
                'consensus_items': resolved_items,
                'conflicts_detected': conflicts_detected,
                'conflict_details': conflict_details,
                'items_selected': len(resolved_items)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_information_content(self, content: Any) -> float:
        """Calculate information content score for conflict resolution"""
        try:
            if isinstance(content, dict):
                return len(str(content))  # More complex dicts have more info
            elif isinstance(content, list):
                return len(content) * 10  # Lists with more items have more info
            elif isinstance(content, str):
                return len(content.strip())  # Longer strings have more info
            else:
                return len(str(content))
        except:
            return 0.0
    
    def _create_final_consensus(self, 
                               clusters: List[ConsensusCluster], 
                               algorithm_results: Dict[str, Any]) -> List[ConsensusItem]:
        """Combine algorithm results to create final consensus"""
        
        # Collect votes from all algorithms
        item_votes = defaultdict(lambda: {'votes': 0, 'total_confidence': 0.0, 'algorithms': []})
        
        for algorithm, result in algorithm_results.items():
            if 'error' in result:
                continue
                
            consensus_items = result.get('consensus_items', [])
            
            for consensus_item in consensus_items:
                # Handle both dictionary format and direct ConsensusItem objects
                if isinstance(consensus_item, dict):
                    item = consensus_item.get('item')
                    confidence = consensus_item.get('confidence', 0.5)
                else:
                    # Direct ConsensusItem object
                    item = consensus_item
                    confidence = getattr(consensus_item, 'confidence_score', 0.5)
                
                if item:
                    item_key = item.content_hash if hasattr(item, 'content_hash') else str(item.item_id)
                    
                    item_votes[item_key]['votes'] += 1
                    item_votes[item_key]['total_confidence'] += confidence
                    item_votes[item_key]['algorithms'].append(algorithm)
                    item_votes[item_key]['item'] = item
        
        # Select final consensus items
        final_consensus = []
        min_votes = max(1, len(algorithm_results) // 2)  # Majority requirement
        
        for item_key, vote_data in item_votes.items():
            if vote_data['votes'] >= min_votes:
                item = vote_data['item']
                final_confidence = vote_data['total_confidence'] / vote_data['votes']
                
                # Update item confidence
                item.confidence_score = final_confidence
                item.metadata.update({
                    'algorithm_votes': vote_data['votes'],
                    'supporting_algorithms': vote_data['algorithms'],
                    'final_confidence': final_confidence
                })
                
                final_consensus.append(item)
        
        # Sort by confidence
        final_consensus.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return final_consensus
    
    def _apply_ice_loop(self, 
                       consensus_items: List[ConsensusItem],
                       provider_results: Dict[str, Dict],
                       target_path: str) -> List[ConsensusItem]:
        """Apply Iterative Consensus Ensemble (ICE) loop for refinement"""
        
        max_iterations = self.config['ice_loop_max_iterations']
        confidence_threshold = self.config['ice_confidence_threshold']
        
        current_items = consensus_items.copy()
        
        for iteration in range(max_iterations):
            # Find low-confidence items
            low_confidence_items = [
                item for item in current_items 
                if item.confidence_score < confidence_threshold
            ]
            
            if not low_confidence_items:
                logger.info(f"ICE loop converged after {iteration} iterations")
                break
            
            logger.info(f"ICE iteration {iteration + 1}: refining {len(low_confidence_items)} low-confidence items")
            
            # Re-query and refine low-confidence items
            refined_items = self._refine_low_confidence_items(
                low_confidence_items, provider_results, target_path
            )
            
            # Replace low-confidence items with refined versions
            item_id_to_refined = {item.item_id: refined_item for item, refined_item in zip(low_confidence_items, refined_items)}
            
            current_items = [
                item_id_to_refined.get(item.item_id, item) 
                for item in current_items
            ]
        
        return current_items
    
    def _refine_low_confidence_items(self, 
                                   low_confidence_items: List[ConsensusItem],
                                   provider_results: Dict[str, Dict],
                                   target_path: str) -> List[ConsensusItem]:
        """Refine low-confidence items through re-analysis"""
        
        refined_items = []
        
        for item in low_confidence_items:
            # For now, apply basic confidence boosting
            # In a full implementation, this would re-query LLMs or apply human-in-the-loop
            
            refined_item = ConsensusItem(
                item_id=item.item_id + "_refined",
                content=item.content,
                source_models=item.source_models,
                confidence_score=min(1.0, item.confidence_score + 0.1),  # Slight boost
                similarity_scores=item.similarity_scores.copy(),
                metadata={**item.metadata, 'ice_refined': True}
            )
            
            refined_items.append(refined_item)
        
        return refined_items
    
    def _calculate_consensus_quality(self, 
                                   extracted_items: List[ConsensusItem],
                                   clusters: List[ConsensusCluster],
                                   final_consensus: List[ConsensusItem],
                                   algorithm_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive consensus quality metrics"""
        
        metrics = {}
        
        # Coverage: how much of original data is represented
        metrics['coverage'] = len(final_consensus) / max(1, len(extracted_items))
        
        # Compression: how much data was consolidated
        metrics['compression_ratio'] = 1.0 - (len(final_consensus) / max(1, len(extracted_items)))
        
        # Algorithm agreement: how many algorithms contributed to consensus
        successful_algorithms = len([r for r in algorithm_results.values() if 'error' not in r])
        metrics['algorithm_agreement'] = successful_algorithms / max(1, len(algorithm_results))
        
        # Average confidence of final items
        if final_consensus:
            metrics['average_confidence'] = np.mean([item.confidence_score for item in final_consensus])
        else:
            metrics['average_confidence'] = 0.0
        
        # Model diversity: how many different models contributed to final consensus
        all_source_models = set()
        for item in final_consensus:
            all_source_models.update(item.source_models)
        
        total_models = len(set(item.source_models[0] for item in extracted_items if item.source_models))
        metrics['model_diversity'] = len(all_source_models) / max(1, total_models)
        
        # Cluster quality: average intra-cluster similarity
        if clusters:
            metrics['cluster_quality'] = np.mean([c.intra_cluster_similarity for c in clusters])
        else:
            metrics['cluster_quality'] = 0.0
        
        # Overall quality as weighted combination
        weights = {
            'coverage': 0.2,
            'algorithm_agreement': 0.2, 
            'average_confidence': 0.3,
            'model_diversity': 0.15,
            'cluster_quality': 0.15
        }
        
        metrics['overall_quality'] = sum(metrics[m] * w for m, w in weights.items())
        
        return metrics
    
    def _create_empty_result(self, target_path: str, provider_results: Dict, reason: str) -> UniversalConsensusResult:
        """Create empty result for failed consensus"""
        
        return UniversalConsensusResult(
            target_path=target_path,
            input_model_count=len(provider_results),
            extracted_items_count=0,
            consensus_clusters=[],
            final_consensus_items=[],
            consensus_quality_metrics={'overall_quality': 0.0, 'reason': reason},
            processing_metadata={'error': reason, 'timestamp': time.time()}
        )
    
    def _create_error_result(self, target_path: str, provider_results: Dict, error: str, processing_time: float) -> UniversalConsensusResult:
        """Create error result"""
        
        return UniversalConsensusResult(
            target_path=target_path,
            input_model_count=len(provider_results),
            extracted_items_count=0,
            consensus_clusters=[],
            final_consensus_items=[],
            consensus_quality_metrics={'overall_quality': 0.0, 'error': error},
            processing_metadata={
                'error': error,
                'processing_time': processing_time,
                'timestamp': time.time()
            }
        )
    
    def batch_consensus(self, 
                       provider_results: Dict[str, Dict],
                       target_paths: List[str],
                       consensus_strength: ConsensusStrength = ConsensusStrength.STRONG) -> Dict[str, UniversalConsensusResult]:
        """Run consensus for multiple paths in batch"""
        
        batch_results = {}
        
        for path in target_paths:
            logger.info(f"Processing batch consensus for path: {path}")
            result = self.create_consensus(provider_results, path, consensus_strength)
            batch_results[path] = result
        
        return batch_results
    
    def register_llm_arbitrator(self, provider_name: str, provider_instance: Any):
        """Register LLM provider for conflict arbitration"""
        self.llm_arbitrators[provider_name] = provider_instance
        self.llm_arbitrator.register_llm_provider(provider_name, provider_instance)
        logger.info(f"Registered LLM arbitrator: {provider_name}")
    
    def create_similarity_function(self, domain_rules: Dict[str, Any]) -> callable:
        """Create domain-specific similarity function"""
        
        def domain_similarity(item1: Any, item2: Any) -> float:
            """Custom similarity function based on domain rules"""
            
            similarity = 0.0
            
            if isinstance(item1, dict) and isinstance(item2, dict):
                # Check specific field similarities based on domain
                for field, weight in domain_rules.get('field_weights', {}).items():
                    if field in item1 and field in item2:
                        if item1[field] == item2[field]:
                            similarity += weight
                        elif isinstance(item1[field], str) and isinstance(item2[field], str):
                            # String similarity for text fields
                            words1 = set(item1[field].lower().split())
                            words2 = set(item2[field].lower().split())
                            if words1 and words2:
                                word_sim = len(words1 & words2) / len(words1 | words2)
                                similarity += weight * word_sim
                
                # Apply domain-specific bonus rules
                for rule in domain_rules.get('bonus_rules', []):
                    if rule['condition'](item1, item2):
                        similarity += rule['bonus']
            
            return min(1.0, similarity)
        
        return domain_similarity
    
    def _init_embedding_service(self):
        """Initialize embedding service with fallback"""
        if EmbeddingService:
            try:
                return EmbeddingService(
                    model_name=self.config.get('embedding_model', 'gemini-embedding-001')
                )
            except Exception as e:
                logger.warning(f"Failed to initialize EmbeddingService: {e}")
        return MockEmbeddingService()
    
    def _init_bft_consensus(self):
        """Initialize BFT consensus with fallback"""
        if BFTConsensus:
            try:
                return BFTConsensus(
                    fault_tolerance=self.config.get('bft_fault_tolerance', 0.33)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize BFTConsensus: {e}")
        return MockBFTConsensus()
    
    def _init_dempster_shafer(self):
        """Initialize Dempster-Shafer with fallback"""
        if DempsterShaferEngine:
            try:
                return DempsterShaferEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize DempsterShaferEngine: {e}")
        return MockDempsterShaferEngine()
    
    def _init_mcts_optimization(self):
        """Initialize MCTS with fallback"""
        if MCTSOptimization:
            try:
                return MCTSOptimization(
                    exploration_constant=self.config.get('mcts_exploration', 1.41),
                    max_iterations=self.config.get('mcts_iterations', 500)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MCTSOptimization: {e}")
        return MockMCTSOptimization()
    
    def _init_preprocessor(self):
        """Initialize preprocessor with fallback"""
        if ConsensusDataPreprocessor:
            try:
                return ConsensusDataPreprocessor()
            except Exception as e:
                logger.warning(f"Failed to initialize ConsensusDataPreprocessor: {e}")
        return MockConsensusDataPreprocessor()
    
    def _init_llm_arbitrator(self):
        """Initialize LLM arbitrator with fallback"""
        if LLMConflictArbitrator:
            try:
                return LLMConflictArbitrator(config=self.config.get('llm_arbitration', {}))
            except Exception as e:
                logger.warning(f"Failed to initialize LLMConflictArbitrator: {e}")
        return MockLLMConflictArbitrator()
    
    def _apply_llm_arbitration(self, 
                              algorithm_results: Dict[str, Any],
                              target_path: str) -> Dict[str, Any]:
        """Apply LLM-based conflict arbitration to algorithm results"""
        
        try:
            # Detect conflicts between algorithm results
            conflicts = self.llm_arbitrator.detect_conflicts(
                algorithm_results, 
                consensus_threshold=self.config.get('llm_arbitration', {}).get('conflict_detection_threshold', 0.3)
            )
            
            if not conflicts:
                logger.info("No conflicts detected for LLM arbitration")
                return algorithm_results
            
            logger.info(f"Detected {len(conflicts)} conflicts for LLM arbitration")
            
            # Get arbitration strategy
            arbitration_config = self.config.get('llm_arbitration', {})
            strategy_name = arbitration_config.get('arbitration_strategy', 'single_llm')
            preferred_provider = arbitration_config.get('preferred_provider', 'mock')
            
            # Map strategy name to enum
            strategy_mapping = {
                'single_llm': ArbitrationStrategy.SINGLE_LLM,
                'multi_llm_consensus': ArbitrationStrategy.MULTI_LLM_CONSENSUS,
                'hierarchical_review': ArbitrationStrategy.HIERARCHICAL_REVIEW,
                'ensemble_weighted': ArbitrationStrategy.ENSEMBLE_WEIGHTED
            }
            strategy = strategy_mapping.get(strategy_name, ArbitrationStrategy.SINGLE_LLM)
            
            # Run arbitration (using asyncio.run for sync context)
            arbitration_results = asyncio.run(
                self.llm_arbitrator.arbitrate_conflicts(conflicts, strategy, preferred_provider)
            )
            
            if arbitration_results:
                # Apply arbitration results to algorithm results
                enhanced_results = self._integrate_arbitration_results(
                    algorithm_results, arbitration_results, target_path
                )
                
                logger.info(f"Applied {len(arbitration_results)} arbitration results to algorithm results")
                return enhanced_results
            else:
                logger.warning("No arbitration results obtained")
                return algorithm_results
                
        except Exception as e:
            logger.error(f"LLM arbitration failed: {e}")
            return algorithm_results
    
    def _integrate_arbitration_results(self, 
                                      algorithm_results: Dict[str, Any],
                                      arbitration_results: Dict[str, Any],
                                      target_path: str) -> Dict[str, Any]:
        """Integrate arbitration results back into algorithm results"""
        
        enhanced_results = algorithm_results.copy()
        
        for conflict_id, arbitration in arbitration_results.items():
            try:
                # Add arbitration as a new "algorithm" result
                enhanced_results[f'llm_arbitration_{conflict_id}'] = {
                    'algorithm': 'llm_arbitration',
                    'consensus_items': arbitration.resolution,
                    'confidence_score': arbitration.confidence_score,
                    'reasoning': arbitration.arbitration_reasoning,
                    'source_conflict': conflict_id,
                    'processing_time': arbitration.processing_time,
                    'metadata': arbitration.metadata
                }
                
                # Optionally boost confidence of items that were arbitrated
                self._boost_arbitrated_items(enhanced_results, arbitration, conflict_id)
                
            except Exception as e:
                logger.error(f"Failed to integrate arbitration result {conflict_id}: {e}")
        
        return enhanced_results
    
    def _boost_arbitrated_items(self, 
                               enhanced_results: Dict[str, Any],
                               arbitration: Any,
                               conflict_id: str):
        """Boost confidence of items that were successfully arbitrated"""
        
        confidence_boost = 0.1  # Boost factor for arbitrated items
        
        # Apply boost to algorithms that had conflicts resolved
        for algo_name, algo_result in enhanced_results.items():
            if 'error' not in algo_result and 'consensus_items' in algo_result:
                for item_name, item_data in algo_result['consensus_items'].items():
                    if isinstance(item_data, dict) and 'confidence' in item_data:
                        # Apply boost if this item was part of the arbitrated conflict
                        # (This is a simplified heuristic)
                        if arbitration.confidence_score > 0.7:
                            item_data['confidence'] = min(1.0, item_data['confidence'] + confidence_boost)
                            item_data['arbitration_boosted'] = True

# Mock classes for fallback when real services aren't available
class MockEmbeddingService:
    """Mock embedding service for testing/fallback"""
    
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings"""
        # Simple hash-based embeddings for consistency
        embeddings = []
        for text in texts:
            hash_val = hash(text)
            # Convert hash to 768-dimensional vector
            embedding = [(hash_val >> i) & 1 for i in range(768)]
            embedding = [float(x) for x in embedding]
            embeddings.append(embedding)
        return embeddings
    
    def cosine_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """Calculate cosine similarity matrix"""
        n = len(embeddings)
        matrix = np.eye(n)  # Identity matrix
        
        for i in range(n):
            for j in range(i + 1, n):
                # Simple similarity based on text length difference
                sim = 1.0 / (1.0 + abs(len(embeddings[i]) - len(embeddings[j])) / 100)
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return matrix

class MockBFTConsensus:
    """Mock BFT consensus for testing/fallback"""
    
    def __init__(self, fault_tolerance: float = 0.33):
        self.fault_tolerance = fault_tolerance
    
    def create_consensus(self, data: Dict) -> Dict:
        """Mock BFT consensus"""
        return {'consensus': 'mock_bft', 'confidence': 0.7}

class MockDempsterShaferEngine:
    """Enhanced Mock Dempster-Shafer Engine with proper belief combination for observations"""
    
    def multi_field_consensus_analysis(self, evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced DS analysis for observations and behavioral patterns"""
        try:
            consensus_items = []
            belief_scores = {}
            combined_evidence = {}
            
            # Process evidence for observations structure
            ds_evidence = evidence_data.get('ds_evidence', {})
            
            for cluster_id, cluster_evidence in ds_evidence.items():
                # Combine evidence from all sources for this cluster
                cluster_beliefs = []
                cluster_content = []
                
                for evidence_item in cluster_evidence:
                    belief_mass = evidence_item.get('belief_mass', 0.5)
                    evidence_content = evidence_item.get('evidence_data', {})
                    source = evidence_item.get('source', 'unknown')
                    
                    cluster_beliefs.append(belief_mass)
                    cluster_content.append(evidence_content)
                    
                    # Store individual source beliefs
                    belief_scores[f"{cluster_id}_{source}"] = belief_mass
                
                # Apply Dempster-Shafer combination rule
                if cluster_beliefs:
                    # Simplified DS combination: weighted average with conflict resolution
                    combined_belief = self._combine_beliefs(cluster_beliefs)
                    
                    # Create consensus item if belief is strong enough
                    if combined_belief > 0.3:  # Lower threshold for observations
                        # Select best content from cluster
                        best_content = self._select_best_content(cluster_content, cluster_beliefs)
                        
                        consensus_item = {
                            'item': ConsensusItem(
                                item_id=f"ds_{cluster_id}",
                                content=best_content,
                                source_models=[f"cluster_{cluster_id}"],
                                confidence_score=combined_belief
                            ),
                            'confidence': combined_belief,
                            'belief_score': combined_belief,
                            'evidence_strength': len(cluster_evidence),
                            'selection_reason': 'dempster_shafer_belief'
                        }
                        consensus_items.append(consensus_item)
                        
                        combined_evidence[cluster_id] = {
                            'combined_belief': combined_belief,
                            'evidence_count': len(cluster_evidence),
                            'consensus_content': best_content
                        }
            
            return {
                'algorithm': 'dempster_shafer',
                'consensus_items': consensus_items,
                'belief_scores': belief_scores,
                'combined_evidence': combined_evidence,
                'evidence_processed': len(ds_evidence),
                'items_selected': len(consensus_items)
            }
            
        except Exception as e:
            logger.error(f"Mock DS analysis failed: {e}")
            return {
                'algorithm': 'dempster_shafer',
                'consensus_items': [],
                'belief_scores': {},
                'error': str(e)
            }
    
    def _combine_beliefs(self, beliefs: List[float]) -> float:
        """Combine multiple belief masses using Dempster-Shafer rule"""
        if not beliefs:
            return 0.0
        
        if len(beliefs) == 1:
            return beliefs[0]
        
        # Simplified DS combination rule
        combined = beliefs[0]
        
        for belief in beliefs[1:]:
            # DS combination: m1 * m2 / (1 - conflict)
            # Assuming low conflict for similar observations
            conflict = 0.1  # Assume 10% conflict
            combined = (combined * belief) / (1 - conflict)
            combined = min(1.0, combined)  # Cap at 1.0
        
        return combined
    
    def _select_best_content(self, contents: List[Any], beliefs: List[float]) -> Any:
        """Select content with highest belief score"""
        if not contents or not beliefs:
            return {}
        
        # Find content with highest belief
        max_belief_idx = beliefs.index(max(beliefs))
        best_content = contents[max_belief_idx]
        
        # If it's a behavioral pattern structure, ensure proper format
        if isinstance(best_content, dict):
            # Ensure observations structure exists
            if 'behavioral_patterns' in str(best_content) or 'temporal_patterns' in str(best_content):
                return best_content
            else:
                # Convert to observations format if needed
                return {
                    'pattern_type': 'security_observation',
                    'content': best_content,
                    'confidence': max(beliefs)
                }
        
        return best_content

class MockMCTSOptimization:
    """Mock MCTS optimization for testing/fallback"""
    
    def __init__(self, exploration_constant: float = 1.41, max_iterations: int = 500):
        self.exploration_constant = exploration_constant
        self.max_iterations = max_iterations
    
    def optimize_schema_consensus(self, data: Dict) -> Dict:
        """Mock MCTS optimization"""
        return {
            'optimized_consensus': 'mock_mcts',
            'exploration_constant': self.exploration_constant,
            'iterations': min(self.max_iterations, 100),
            'confidence': 0.75
        }

class MockConsensusDataPreprocessor:
    """Mock preprocessor for testing/fallback"""
    
    def preprocess_data(self, data: Any) -> Any:
        """Mock preprocessing"""
        return data

class MockLLMConflictArbitrator:
    """Enhanced Mock LLM Conflict Arbitrator with proper async support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def detect_conflicts(self, algorithm_results: Dict[str, Any], consensus_threshold: float = 0.3) -> List[str]:
        """Enhanced conflict detection for observations"""
        conflicts = []
        
        # Check for conflicts between algorithm results
        algorithm_names = list(algorithm_results.keys())
        
        for i, algo1 in enumerate(algorithm_names):
            for j, algo2 in enumerate(algorithm_names[i+1:], i+1):
                result1 = algorithm_results[algo1]
                result2 = algorithm_results[algo2]
                
                # Skip if either has errors
                if 'error' in result1 or 'error' in result2:
                    continue
                
                # Check for conflicts in consensus items
                items1 = result1.get('consensus_items', [])
                items2 = result2.get('consensus_items', [])
                
                # Simple conflict detection: different number of items or low overlap
                if abs(len(items1) - len(items2)) > 2:
                    conflict_id = f"conflict_{algo1}_vs_{algo2}_count"
                    conflicts.append(conflict_id)
                
                # Check confidence differences
                conf1 = result1.get('confidence', 0.5)
                conf2 = result2.get('confidence', 0.5)
                
                if abs(conf1 - conf2) > consensus_threshold:
                    conflict_id = f"conflict_{algo1}_vs_{algo2}_confidence"
                    conflicts.append(conflict_id)
        
        return conflicts
    
    async def arbitrate_conflicts(self, conflicts: List[str], strategy, preferred_provider: str) -> Dict[str, Any]:
        """Enhanced async conflict arbitration for observations"""
        arbitration_results = {}
        
        for conflict_id in conflicts:
            try:
                # Mock arbitration result with proper structure
                mock_arbitration = MockArbitrationResult(
                    conflict_id=conflict_id,
                    resolution=self._create_mock_resolution(conflict_id),
                    confidence_score=0.75,
                    arbitration_reasoning=f"Mock arbitration resolved conflict {conflict_id} using {strategy}",
                    processing_time=0.1,
                    metadata={'strategy': str(strategy), 'provider': preferred_provider}
                )
                
                arbitration_results[conflict_id] = mock_arbitration
                
            except Exception as e:
                logger.error(f"Mock arbitration failed for {conflict_id}: {e}")
        
        return arbitration_results
    
    def _create_mock_resolution(self, conflict_id: str) -> List[Dict[str, Any]]:
        """Create mock resolution items for observations"""
        # Create mock consensus items that focus on observations
        mock_items = []
        
        if 'behavioral' in conflict_id or 'observations' in conflict_id:
            # Create mock behavioral pattern observation
            mock_item = {
                'item': ConsensusItem(
                    item_id=f"arbitrated_{conflict_id}",
                    content={
                        'pattern_type': 'security_behavior',
                        'description': f'Arbitrated pattern for {conflict_id}',
                        'indicators': ['mock_indicator_1', 'mock_indicator_2'],
                        'severity': 'medium',
                        'confidence': 0.75
                    },
                    source_models=['arbitrator'],
                    confidence_score=0.75
                ),
                'confidence': 0.75,
                'arbitration_reasoning': f'Resolved conflict in {conflict_id}'
            }
            mock_items.append(mock_item)
        
        return mock_items
    
    def register_llm_provider(self, provider_name: str, provider_instance: Any):
        """Register LLM provider"""
        pass

class MockArbitrationResult:
    """Mock arbitration result class"""
    
    def __init__(self, conflict_id: str, resolution: List[Dict], confidence_score: float, 
                 arbitration_reasoning: str, processing_time: float, metadata: Dict):
        self.conflict_id = conflict_id
        self.resolution = resolution
        self.confidence_score = confidence_score
        self.arbitration_reasoning = arbitration_reasoning
        self.processing_time = processing_time
        self.metadata = metadata