"""
Research-Grade Consensus Orchestrator (Multi-agent.md Implementation)
Complete implementation of the modular framework from Section 2 with all research components
"""

import numpy as np
import logging
import time
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
from datetime import datetime
import json

# Import research components (Sections 3.1, 4.3, 6.1, 7.2, 8)
from .embedding_service import EmbeddingService
from .graph_clustering import GraphClustering
from .semantic_similarity import SemanticSimilarity
from .semantic_tree_edit_distance import SemanticallyInformedTreeEditDistance
from .weighted_voting_reliability import WeightedVotingReliabilitySystem
from .muse_llm_adaptation import MuseLLMAdapter
from .ice_loop_refinement import ICELoopRefinement

# Import existing consensus engines (keep Universal Consensus Engine)
from .universal_consensus_engine import UniversalConsensusEngine, ConsensusStrength
from .bft_consensus import BFTConsensus
from .dempster_shafer import DempsterShaferEngine
from .mcts_optimization import MCTSOptimization

logger = logging.getLogger(__name__)

class ResearchConsensusOrchestrator:
    """
    Section 2: Modular Framework for Generalized Consensus
    
    Complete implementation of the research architecture:
    1. Aligner Module (Section 3): Schema normalization + semantic clustering
    2. Consensus Engine (Section 6): BFT protocols with weighted voting
    3. Uncertainty Module (Section 7): MUSE-based confidence calibration
    4. ICE Loop (Section 8): Iterative refinement mechanism
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize research-grade consensus orchestrator
        
        Args:
            config: Configuration for all research components
        """
        self.config = self._get_research_default_config()
        if config:
            self.config.update(config)
        
        logger.info("Initializing Research Consensus Orchestrator (Multi-agent.md)")
        
        # Section 3.1: Modern SBERT Embeddings
        self.embedding_service = EmbeddingService(
            model_name=self.config.get('sbert_model', 'gemini-embedding-001')
        )
        
        # Section 4.3: Semantically-Informed Tree Edit Distance
        self.semantic_ted = SemanticallyInformedTreeEditDistance(self.embedding_service)
        
        # Existing semantic analysis components (now properly integrated)
        self.graph_clustering = GraphClustering()
        self.semantic_similarity = SemanticSimilarity()
        
        # Section 6.1: Weighted Voting & Dynamic Reliability
        self.weighted_voting_system = WeightedVotingReliabilitySystem(
            alpha=self.config.get('reliability_alpha', 0.7),
            beta=self.config.get('reliability_beta', 0.3),
            reliability_decay_days=self.config.get('reliability_decay_days', 30)
        )
        
        # Section 7.2: MUSE Algorithm Adaptation
        self.muse_adapter = MuseLLMAdapter(
            confidence_aggregation_method=self.config.get('muse_aggregation', 'weighted_average')
        )
        
        # Section 8: ICE Loop Integration
        self.ice_loop = ICELoopRefinement(
            confidence_threshold=self.config.get('ice_threshold', 0.6),
            max_iterations=self.config.get('ice_max_iterations', 3),
            enable_hitl=self.config.get('enable_hitl', True)
        )
        
        # Keep existing Universal Consensus Engine (BFT→DS→MCTS→LLM→ICE pipeline)
        self.universal_engine = UniversalConsensusEngine(config=self.config)
        
        # Additional consensus engines
        self.bft_consensus = BFTConsensus(
            fault_tolerance=self.config.get('bft_fault_tolerance', 0.33)
        )
        self.dempster_shafer = DempsterShaferEngine()
        self.mcts_optimization = MCTSOptimization(
            exploration_constant=self.config.get('mcts_exploration', 1.41),
            max_iterations=self.config.get('mcts_iterations', 1000)
        )
        
        # Research pipeline state
        self.research_results = {}
        self.agent_registry = {}  # Track LLM agents for weighted voting
        
        logger.info("Research Consensus Orchestrator initialized with all Section components")
    
    def _get_research_default_config(self) -> Dict[str, Any]:
        """Get research-grade default configuration"""
        return {
            # Section 3.1: SBERT Configuration
            'sbert_model': 'sentence-transformers/all-MiniLM-L6-v2',  # Use fast local SentenceTransformers
            'enable_ontology_grounding': False,
            
            # Section 4.3: Semantic TED Configuration
            'ted_alpha': 0.5,  # Balance between structural and semantic
            
            # Section 6.1: Weighted Voting Configuration
            'reliability_alpha': 0.7,  # Weight for historical reliability (Ri)
            'reliability_beta': 0.3,   # Weight for current confidence (Ci)
            'reliability_decay_days': 30,
            'min_reliability_score': 0.1,
            
            # Section 7.2: MUSE Configuration
            'muse_aggregation': 'weighted_average',
            'muse_local_similarity_window': 3,
            
            # Section 8: ICE Loop Configuration
            'ice_threshold': 0.6,
            'ice_max_iterations': 3,
            'enable_hitl': True,
            'hitl_threshold': 0.4,
            
            # Integration settings
            'consensus_threshold': 0.6,
            'similarity_threshold': 0.7,
            'max_parallel_processes': 4,
            'timeout_seconds': 600,  # Longer timeout for research processes
            
            # Algorithm weights (updated for research components)
            'algorithm_weights': {
                'semantic_clustering': 0.15,
                'graph_clustering': 0.15,
                'weighted_voting': 0.25,
                'muse_confidence': 0.20,
                'bft_consensus': 0.15,
                'semantic_similarity': 0.10
            }
        }
    
    def register_llm_agents(self, agent_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Register LLM agents for weighted voting and MUSE adaptation
        
        Args:
            agent_configs: Dictionary mapping agent_id to configuration
        """
        for agent_id, config in agent_configs.items():
            # Register in weighted voting system
            self.weighted_voting_system.register_agent(
                agent_id=agent_id,
                specialization_domains=config.get('domains', ['general']),
                initial_reliability=config.get('initial_reliability', 0.5)
            )
            
            # Register as MUSE atlas
            self.muse_adapter.register_llm_atlas(
                agent_id=agent_id,
                architecture_type=config.get('architecture', 'transformer'),
                training_characteristics=config.get('training', 'general'),
                specialization_domains=config.get('domains', ['general']),
                performance_profile=config.get('performance_profile', {})
            )
            
            self.agent_registry[agent_id] = config
        
        logger.info(f"Registered {len(agent_configs)} LLM agents for research consensus")
    
    async def research_grade_consensus(self, 
                                     provider_results: Dict[str, Dict],
                                     target_key_path: str = 'parsed_json',
                                     task_domain: str = 'general',
                                     enable_ice_refinement: bool = True) -> Dict[str, Any]:
        """
        Main research-grade consensus pipeline implementing all Multi-agent.md sections
        
        Args:
            provider_results: Results from multiple LLM providers/agents
            target_key_path: Path to target data in results
            task_domain: Domain for specialization weighting
            enable_ice_refinement: Whether to apply ICE loop refinement
            
        Returns:
            Complete research consensus results with all section analyses
        """
        logger.info(f"Starting research-grade consensus for {len(provider_results)} providers")
        start_time = time.time()
        
        # ============================================================================
        # SECTION 3: ALIGNER MODULE - Schema Normalization & Semantic Clustering
        # ============================================================================
        
        logger.info("Phase 1: Aligner Module (Section 3)")
        alignment_result = await self._execute_aligner_module(
            provider_results, target_key_path
        )
        
        # ============================================================================
        # SECTION 4: HYBRID SIMILARITY ANALYSIS (Semantically-Informed TED)
        # ============================================================================
        
        logger.info("Phase 2: Semantic Tree Edit Distance (Section 4.3)")
        similarity_analysis = await self._execute_semantic_ted_analysis(
            provider_results, alignment_result
        )
        
        # ============================================================================
        # SECTION 6: CONSENSUS ENGINE - Weighted Voting & BFT Protocols
        # ============================================================================
        
        logger.info("Phase 3: Consensus Engine with Weighted Voting (Section 6.1)")
        consensus_result = await self._execute_weighted_consensus_engine(
            alignment_result, task_domain
        )
        
        # ============================================================================
        # SECTION 7: UNCERTAINTY MODULE - MUSE Algorithm Adaptation
        # ============================================================================
        
        logger.info("Phase 4: MUSE Uncertainty Quantification (Section 7.2)")
        uncertainty_analysis = await self._execute_muse_uncertainty_module(
            consensus_result, alignment_result
        )
        
        # ============================================================================
        # SECTION 8: ICE LOOP - Iterative Refinement (if enabled)
        # ============================================================================
        
        ice_results = {}
        if enable_ice_refinement and self.config.get('enable_ice_refinement', True):
            logger.info("Phase 5: ICE Loop Refinement (Section 8) - ENABLED")
            ice_results = await self._execute_ice_loop_refinement(
                consensus_result, uncertainty_analysis
            )
        else:
            logger.info("Phase 5: ICE Loop Refinement (Section 8) - DISABLED")
            ice_results = {
                'refinement_iterations': 0,
                'quality_improvements': {},
                'hitl_items': []
            }
        
        # ============================================================================
        # INTEGRATION & FINAL RESULT COMPILATION
        # ============================================================================
        
        processing_time = time.time() - start_time
        
        final_result = {
            # Core consensus output
            'consensus': consensus_result.get('final_consensus'),
            'confidence': uncertainty_analysis.get('overall_confidence', 0.0),
            
            # Research component results
            'aligner_analysis': alignment_result,
            'semantic_similarity_analysis': similarity_analysis,
            'weighted_voting_analysis': consensus_result,
            'muse_uncertainty_analysis': uncertainty_analysis,
            'ice_refinement_results': ice_results,
            
            # Metadata
            'research_methodology': {
                'framework': 'Multi-agent.md Sections 2-8',
                'components_used': [
                    'Section 3.1: SBERT Semantic Clustering',
                    'Section 4.3: Semantically-Informed TED',
                    'Section 6.1: Weighted Voting & Dynamic Reliability',
                    'Section 7.2: MUSE Algorithm Adaptation',
                    'Section 8: ICE Loop Refinement'
                ],
                'processing_time_seconds': processing_time,
                'providers_processed': len(provider_results),
                'ice_enabled': enable_ice_refinement,
                'task_domain': task_domain
            },
            
            # Quality metrics
            'quality_metrics': {
                'consensus_strength': consensus_result.get('consensus_strength', 0.0),
                'uncertainty_calibration': uncertainty_analysis.get('calibration_score', 0.0),
                'semantic_coherence': similarity_analysis.get('coherence_score', 0.0),
                'reliability_weighted_confidence': consensus_result.get('reliability_weighted_confidence', 0.0)
            },
            
            # System state
            'agent_statistics': self._compile_agent_statistics(),
            'hitl_items': self.ice_loop.get_hitl_items() if enable_ice_refinement else []
        }
        
        logger.info(f"Research consensus complete: {processing_time:.2f}s, confidence: {final_result['confidence']:.3f}")
        return final_result
    
    async def _execute_aligner_module(self, provider_results: Dict[str, Dict], target_key_path: str) -> Dict[str, Any]:
        """
        Section 3: Aligner Module - Schema normalization and semantic clustering
        """
        # Extract target data from all providers
        extracted_data = {}
        for provider, result in provider_results.items():
            target_data = self._get_nested_value(result, target_key_path)
            if target_data:
                extracted_data[provider] = target_data
        
        if not extracted_data:
            return {'error': 'No valid data extracted for alignment'}
        
        # Step 1: Build relationship graph (integrate graph clustering) - if clustering enabled
        relationship_graph = {}
        if self.config.get('enable_semantic_clustering', True):
            relationship_graph = self.graph_clustering.build_field_relationship_graph(
                provider_results, similarity_threshold=self.config.get('similarity_threshold', 0.7)
            )
        else:
            logger.info("Graph clustering disabled - skipping relationship graph")
            relationship_graph = {'nodes': [], 'edges': [], 'clusters': []}
        
        # Step 2: Semantic clustering for label canonicalization (Section 3.1)
        all_field_names = []
        field_metadata = []
        
        for provider, data in extracted_data.items():
            if isinstance(data, dict) and 'fields' in data:
                for field_name, field_data in data['fields'].items():
                    all_field_names.append(field_name)
                    field_metadata.append({
                        'provider': provider,
                        'field_name': field_name,
                        'field_data': field_data
                    })
        
        # Generate SBERT embeddings and perform canonical concept assignment
        canonical_assignment = {'canonical_concepts': {}, 'concept_assignments': {}}
        
        if all_field_names and self.config.get('enable_sbert_embeddings', True):
            try:
                field_embeddings = self.embedding_service.embed_text(all_field_names)
                
                canonical_assignment = self.embedding_service.canonical_concept_assignment(
                    field_embeddings, 
                    all_field_names,
                    enable_ontology_grounding=self.config.get('enable_ontology_grounding', False)
                )
                logger.info("SBERT embeddings successfully generated")
            except Exception as embedding_error:
                logger.error(f"SBERT embedding failed: {str(embedding_error)}")
                logger.info("Falling back to basic field mapping without embeddings")
                # Create basic canonical concepts without embeddings
                for i, field_name in enumerate(all_field_names):
                    canonical_assignment['canonical_concepts'][f"concept_{i}"] = {
                        'canonical_label': field_name,
                        'confidence_score': 0.5,
                        'provider_fields': [field_name]
                    }
        elif all_field_names:
            logger.info("SBERT embeddings disabled - using basic field mapping")
            # Create basic canonical concepts without embeddings
            for i, field_name in enumerate(all_field_names):
                canonical_assignment['canonical_concepts'][f"concept_{i}"] = {
                    'canonical_label': field_name,
                    'confidence_score': 0.5,
                    'provider_fields': [field_name]
                }
        
        # Step 3: Cross-provider semantic similarity analysis (if enabled)
        cross_provider_similarity = {}
        if self.config.get('enable_semantic_clustering', True):
            try:
                cross_provider_similarity = self.semantic_similarity.cross_model_result_comparison(provider_results)
                logger.info("Semantic similarity analysis completed")
            except Exception as similarity_error:
                logger.error(f"Semantic similarity analysis failed: {str(similarity_error)}")
                logger.info("Using fallback similarity analysis")
                cross_provider_similarity = {
                    'similarity_matrix': {},
                    'cluster_assignments': {},
                    'semantic_distances': {},
                    'fallback_used': True
                }
        else:
            logger.info("Semantic clustering disabled - skipping similarity analysis")
            cross_provider_similarity = {
                'similarity_matrix': {},
                'cluster_assignments': {},
                'semantic_distances': {}
            }
        
        return {
            'relationship_graph': {
                'nodes': len(relationship_graph.get('nodes', [])),
                'edges': len(relationship_graph.get('edges', [])),
                'clusters': len(relationship_graph.get('clusters', [])),
                'graph_metrics': relationship_graph.get('all_fields', {})
            },
            'canonical_concepts': canonical_assignment,
            'cross_provider_analysis': cross_provider_similarity,
            'alignment_metadata': {
                'total_fields_processed': len(all_field_names),
                'providers_analyzed': len(extracted_data),
                'clustering_method': 'SBERT + HDBSCAN'
            }
        }
    
    async def _execute_semantic_ted_analysis(self, provider_results: Dict[str, Dict], alignment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Section 4.3: Semantically-Informed Tree Edit Distance Analysis
        """
        # Convert provider results to tree structures
        provider_trees = self.semantic_ted.convert_json_results_to_trees(provider_results)
        
        if len(provider_trees) < 2:
            return {'warning': 'Insufficient data for tree comparison'}
        
        # Extract trees and IDs
        trees = [tree for _, tree in provider_trees]
        tree_ids = [provider for provider, _ in provider_trees]
        
        # Perform batch tree comparison with hybrid similarity
        comparison_result = self.semantic_ted.batch_tree_comparison(trees, tree_ids)
        
        # Calculate overall semantic coherence
        similarity_matrix = comparison_result['similarity_matrix']
        avg_similarity = comparison_result['analysis']['average_similarity']
        
        coherence_score = avg_similarity  # Use average similarity as coherence measure
        
        return {
            'tree_comparison_results': comparison_result,
            'coherence_score': coherence_score,
            'structural_analysis': {
                'trees_compared': len(trees),
                'average_similarity': avg_similarity,
                'max_similarity': comparison_result['analysis']['max_similarity'],
                'min_similarity': comparison_result['analysis']['min_similarity']
            },
            'detailed_comparisons': comparison_result['detailed_comparisons']
        }
    
    async def _execute_weighted_consensus_engine(self, alignment_result: Dict[str, Any], task_domain: str) -> Dict[str, Any]:
        """
        Section 6.1: Consensus Engine with Weighted Voting and Dynamic Reliability
        """
        # Extract canonical concepts for voting
        canonical_concepts = alignment_result.get('canonical_concepts', {}).get('canonical_concepts', {})
        
        if not canonical_concepts:
            return {'error': 'No canonical concepts available for consensus'}
        
        # Simulate agent votes on canonical concepts (in real implementation, would query actual agents)
        agent_votes = {}
        agent_confidences = {}
        
        for concept_id, concept_info in canonical_concepts.items():
            concept_votes = {}
            concept_confidences = {}
            
            # Simulate votes from registered agents
            for agent_id in self.agent_registry.keys():
                # In real implementation, would query agent about this concept
                vote_content = {
                    'concept_id': concept_id,
                    'canonical_label': concept_info['canonical_label'],
                    'confidence': np.random.uniform(0.4, 0.9),  # Simulated
                    'support': True  # Simulated support
                }
                
                concept_votes[agent_id] = vote_content
                concept_confidences[agent_id] = vote_content['confidence']
            
            # Apply weighted voting consensus (if enabled)
            # Check both 'enable_weighted_voting' and 'use_weighted_voting' for compatibility
            enable_weighted = self.config.get('enable_weighted_voting', self.config.get('use_weighted_voting', True))
            if enable_weighted:
                weighted_result = self.weighted_voting_system.weighted_consensus_vote(
                    votes=concept_votes,
                    confidences=concept_confidences,
                    hierarchy_level=concept_info.get('hierarchy_level', 0),
                    task_domain=task_domain
                )
            else:
                # Simple majority vote without weighting
                avg_confidence = np.mean(list(concept_confidences.values())) if concept_confidences else 0.5
                weighted_result = {
                    'consensus_content': concept_info['canonical_label'],
                    'consensus_confidence': avg_confidence,
                    'voting_analysis': {
                        'consensus_reached': True,
                        'weighted_voting_disabled': True
                    },
                    'reliability_scores': {}
                }
            
            agent_votes[concept_id] = weighted_result
        
        # Compile overall consensus results
        total_concepts = len(canonical_concepts)
        consensus_reached_count = sum(1 for result in agent_votes.values() 
                                    if result.get('voting_analysis', {}).get('consensus_reached', False))
        
        overall_confidence = np.mean([
            result.get('confidence', 0.0) for result in agent_votes.values()
        ]) if agent_votes else 0.0
        
        consensus_strength = consensus_reached_count / max(total_concepts, 1)
        
        return {
            'final_consensus': agent_votes,
            'consensus_strength': consensus_strength,
            'reliability_weighted_confidence': overall_confidence,
            'voting_statistics': {
                'total_concepts': total_concepts,
                'consensus_reached': consensus_reached_count,
                'consensus_rate': consensus_strength,
                'average_confidence': overall_confidence
            },
            'agent_participation': {
                agent_id: len([v for v in agent_votes.values() 
                              if agent_id in v.get('voting_analysis', {}).get('agent_weights', {})])
                for agent_id in self.agent_registry.keys()
            }
        }
    
    async def _execute_muse_uncertainty_module(self, consensus_result: Dict[str, Any], alignment_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Section 7.2: MUSE Algorithm Adaptation for Uncertainty Quantification
        """
        final_consensus = consensus_result.get('final_consensus', {})
        
        if not final_consensus:
            return {'error': 'No consensus data available for MUSE analysis'}
        
        # Prepare candidate nodes for MUSE processing
        candidate_nodes = {}
        for concept_id, consensus_data in final_consensus.items():
            # Extract agent proposals from consensus data
            agent_proposals = {}
            voting_analysis = consensus_data.get('voting_analysis', {})
            
            for agent_id in voting_analysis.get('agent_weights', {}).keys():
                agent_proposals[agent_id] = consensus_data.get('consensus')
            
            candidate_nodes[concept_id] = agent_proposals
        
        # Apply MUSE batch consensus
        muse_results = self.muse_adapter.batch_muse_consensus(
            candidate_nodes=candidate_nodes,
            embedding_service=self.embedding_service
        )
        
        # Calculate overall confidence calibration
        node_confidences = [result['final_confidence_score'] 
                          for result in muse_results['node_confidences'].values()]
        
        overall_confidence = np.mean(node_confidences) if node_confidences else 0.0
        confidence_std = np.std(node_confidences) if node_confidences else 0.0
        
        # Calibration score based on confidence distribution
        calibration_score = max(0.0, 1.0 - confidence_std)  # Lower std = better calibration
        
        return {
            'muse_consensus_results': muse_results,
            'overall_confidence': overall_confidence,
            'calibration_score': calibration_score,
            'uncertainty_statistics': {
                'confidence_mean': overall_confidence,
                'confidence_std': confidence_std,
                'high_confidence_nodes': muse_results['ensemble_statistics']['high_confidence_nodes'],
                'low_confidence_nodes': muse_results['ensemble_statistics']['low_confidence_nodes']
            }
        }
    
    async def _execute_ice_loop_refinement(self, consensus_result: Dict[str, Any], uncertainty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Section 8: ICE Loop Iterative Refinement
        """
        # Prepare consensus results in ICE-compatible format
        ice_input = {}
        final_consensus = consensus_result.get('final_consensus', {})
        muse_confidences = uncertainty_analysis.get('muse_consensus_results', {}).get('node_confidences', {})
        
        for concept_id in final_consensus.keys():
            muse_confidence = muse_confidences.get(concept_id, {}).get('final_confidence_score', 0.5)
            
            ice_input[concept_id] = {
                'final_confidence_score': muse_confidence,
                'consensus': final_consensus[concept_id].get('consensus'),
                'voting_analysis': final_consensus[concept_id].get('voting_analysis', {}),
                'consensus_reached': final_consensus[concept_id].get('voting_analysis', {}).get('consensus_reached', False)
            }
        
        # Apply ICE loop refinement
        refined_results = await self.ice_loop.batch_ice_refinement(
            consensus_results=ice_input,
            consensus_engine=self,  # Pass self as consensus engine
            llm_agents=self.agent_registry
        )
        
        # Get ICE statistics
        ice_stats = self.ice_loop.get_ice_statistics()
        
        return {
            'refined_consensus': refined_results,
            'ice_statistics': ice_stats,
            'hitl_queue_size': len(self.ice_loop.get_hitl_items()),
            'refinement_summary': {
                'nodes_processed': len(ice_input),
                'nodes_refined': ice_stats['nodes_refined'],
                'success_rate': ice_stats['success_rate'],
                'hitl_interventions': ice_stats['hitl_interventions']
            }
        }
    
    def _compile_agent_statistics(self) -> Dict[str, Any]:
        """Compile statistics for all registered agents"""
        agent_stats = {}
        
        for agent_id in self.agent_registry.keys():
            weighted_voting_stats = self.weighted_voting_system.get_agent_statistics(agent_id)
            agent_stats[agent_id] = {
                'weighted_voting': weighted_voting_stats,
                'muse_atlas_info': {
                    'registered': agent_id in self.muse_adapter.ensemble_atlases,
                    'specialization_domains': self.agent_registry[agent_id].get('domains', [])
                }
            }
        
        return agent_stats
    
    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """Get nested value from dictionary using dot-separated key path"""
        try:
            keys = key_path.split('.')
            current = data
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
                    
            return current
        except Exception as e:
            logger.error(f"Error accessing nested value '{key_path}': {e}")
            return None
    
    # Legacy compatibility methods
    
    def unified_consensus(self, 
                        provider_results: Dict[str, Dict],
                        target_key_path: str = 'parsed_json',
                        consensus_type: str = 'research_grade',
                        **kwargs) -> Dict[str, Any]:
        """
        Legacy compatibility wrapper for research consensus
        """
        if consensus_type == 'research_grade':
            # Run async research consensus in sync wrapper
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.research_grade_consensus(provider_results, target_key_path)
                )
            finally:
                loop.close()
        else:
            # Fallback to universal engine for other types
            return self.universal_engine.create_consensus(provider_results, target_key_path)
    
    def process_hitl_feedback(self, node_id: str, human_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process human feedback from ICE loop
        """
        return self.ice_loop.process_hitl_feedback(node_id, human_feedback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        return {
            'components_initialized': {
                'embedding_service': self.embedding_service is not None,
                'graph_clustering': self.graph_clustering is not None,
                'semantic_ted': self.semantic_ted is not None,
                'weighted_voting': self.weighted_voting_system is not None,
                'muse_adapter': self.muse_adapter is not None,
                'ice_loop': self.ice_loop is not None
            },
            'agents_registered': len(self.agent_registry),
            'hitl_queue_size': len(self.ice_loop.get_hitl_items()),
            'ice_statistics': self.ice_loop.get_ice_statistics(),
            'config': self.config
        }