"""
MUSE Algorithm Adaptation for LLM Aggregation (Section 7.2)
MUlti-atlas region Segmentation utilizing Ensembles adapted for semantic consensus
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class LocalConfidenceMetric:
    """Local confidence assessment for specific nodes/concepts"""
    node_id: str
    agent_id: str
    local_confidence: float
    semantic_similarity: float
    structural_consistency: float
    domain_expertise: float

@dataclass
class EnsembleAtlas:
    """Represents an LLM agent as an 'atlas' in MUSE terminology"""
    agent_id: str
    architecture_type: str
    training_characteristics: str
    specialization_domains: List[str]
    performance_profile: Dict[str, float]

class MuseLLMAdapter:
    """
    Section 7.2: Adapting the MUSE Algorithm for LLM Aggregation
    
    Original MUSE: Multi-atlas region segmentation with ensemble warping algorithms
    Our Adaptation: Multi-agent semantic consensus with ensemble LLM 'algorithms'
    """
    
    def __init__(self, 
                 local_similarity_window: int = 3,
                 confidence_aggregation_method: str = 'weighted_average',
                 minimum_atlas_agreement: float = 0.3):
        """
        Initialize MUSE-LLM adapter
        
        Args:
            local_similarity_window: Window size for local similarity assessment
            confidence_aggregation_method: Method for aggregating confidence scores
            minimum_atlas_agreement: Minimum agreement threshold for inclusion
        """
        self.local_similarity_window = local_similarity_window
        self.confidence_aggregation_method = confidence_aggregation_method
        self.minimum_atlas_agreement = minimum_atlas_agreement
        
        # Ensemble of LLM 'atlases' (agents)
        self.ensemble_atlases: Dict[str, EnsembleAtlas] = {}
        
        # Local confidence assessments
        self.local_confidence_cache: Dict[str, List[LocalConfidenceMetric]] = defaultdict(list)
        
        logger.info(f"Initialized MUSE-LLM Adapter with {confidence_aggregation_method} aggregation")
    
    def register_llm_atlas(self, 
                          agent_id: str,
                          architecture_type: str = 'transformer',
                          training_characteristics: str = 'general',
                          specialization_domains: List[str] = None,
                          performance_profile: Dict[str, float] = None) -> None:
        """
        Register an LLM agent as an 'atlas' in the ensemble
        
        Section 7.2: "Each LLM agent in our system is treated as a distinct 'warping algorithm'"
        
        Args:
            agent_id: Unique identifier for the LLM agent
            architecture_type: Type of architecture (transformer, retrieval-augmented, etc.)
            training_characteristics: Training data characteristics
            specialization_domains: Domains of expertise
            performance_profile: Historical performance metrics
        """
        if specialization_domains is None:
            specialization_domains = ['general']
            
        if performance_profile is None:
            performance_profile = {
                'accuracy': 0.7,
                'consistency': 0.6,
                'domain_expertise': 0.5,
                'calibration': 0.6
            }
        
        atlas = EnsembleAtlas(
            agent_id=agent_id,
            architecture_type=architecture_type,
            training_characteristics=training_characteristics,
            specialization_domains=specialization_domains,
            performance_profile=performance_profile
        )
        
        self.ensemble_atlases[agent_id] = atlas
        logger.info(f"Registered LLM atlas: {agent_id} with domains {specialization_domains}")
    
    def assess_local_confidence(self,
                               node_id: str,
                               agent_proposals: Dict[str, Any],
                               embedding_service,
                               context_nodes: List[str] = None) -> List[LocalConfidenceMetric]:
        """
        Section 7.2: "Locally Optimal Agent Selection"
        
        For each candidate node, assess which LLM agents were most "confident"
        in proposing that specific node or semantically equivalent ones
        
        Args:
            node_id: Identifier for the node being assessed
            agent_proposals: Dictionary of agent proposals for this node
            embedding_service: Service for semantic similarity calculation
            context_nodes: Surrounding nodes for local context assessment
            
        Returns:
            List of local confidence metrics for each agent
        """
        local_confidences = []
        
        # Get node content for semantic analysis
        node_contents = {}
        for agent_id, proposal in agent_proposals.items():
            if proposal is not None:
                node_contents[agent_id] = self._extract_node_content(proposal)
        
        # Generate embeddings for semantic similarity assessment
        if node_contents:
            all_contents = list(node_contents.values())
            content_embeddings = embedding_service.embed_text(all_contents)
            
            # Calculate pairwise semantic similarities
            similarity_matrix = embedding_service.cosine_similarity_matrix(content_embeddings)
        
        # Assess each agent's local confidence
        for i, (agent_id, proposal) in enumerate(agent_proposals.items()):
            if proposal is None:
                continue
                
            atlas = self.ensemble_atlases.get(agent_id)
            if atlas is None:
                self.register_llm_atlas(agent_id)  # Auto-register unknown agents
                atlas = self.ensemble_atlases[agent_id]
            
            # Extract confidence indicators
            local_confidence = self._extract_agent_confidence(agent_id, proposal)
            
            # Calculate semantic similarity (average with other agents)
            semantic_sim = 0.0
            if len(node_contents) > 1 and node_contents:
                other_similarities = [similarity_matrix[i, j] for j in range(len(node_contents)) if j != i]
                semantic_sim = np.mean(other_similarities) if other_similarities else 0.0
            
            # Assess structural consistency with context
            structural_consistency = self._assess_structural_consistency(
                agent_id, proposal, context_nodes
            )
            
            # Domain expertise bonus
            domain_expertise = self._calculate_domain_expertise_score(
                agent_id, node_id, atlas
            )
            
            confidence_metric = LocalConfidenceMetric(
                node_id=node_id,
                agent_id=agent_id,
                local_confidence=local_confidence,
                semantic_similarity=semantic_sim,
                structural_consistency=structural_consistency,
                domain_expertise=domain_expertise
            )
            
            local_confidences.append(confidence_metric)
        
        # Cache for future use
        self.local_confidence_cache[node_id] = local_confidences
        
        logger.debug(f"Assessed local confidence for node {node_id}: {len(local_confidences)} agents")
        return local_confidences
    
    def locally_optimal_atlas_selection(self,
                                      node_id: str,
                                      local_confidences: List[LocalConfidenceMetric],
                                      selection_criterion: str = 'composite') -> List[Tuple[str, float]]:
        """
        Section 7.2: Select locally optimal atlases (agents) for each node
        
        "We adapt MUSE's concept of 'locally optimal atlas selection.' For each candidate 
        node in the unified set, we assess which LLM agents were most 'confident'..."
        
        Args:
            node_id: Node identifier
            local_confidences: Local confidence assessments
            selection_criterion: Criterion for selection ('confidence', 'similarity', 'composite')
            
        Returns:
            List of (agent_id, selection_weight) tuples sorted by optimality
        """
        if not local_confidences:
            return []
        
        agent_scores = []
        
        for confidence_metric in local_confidences:
            if selection_criterion == 'confidence':
                score = confidence_metric.local_confidence
            elif selection_criterion == 'similarity':
                score = confidence_metric.semantic_similarity
            elif selection_criterion == 'composite':
                # Composite score combining all factors
                score = (
                    0.4 * confidence_metric.local_confidence +
                    0.3 * confidence_metric.semantic_similarity +
                    0.2 * confidence_metric.structural_consistency +
                    0.1 * confidence_metric.domain_expertise
                )
            else:
                score = confidence_metric.local_confidence  # Default
            
            agent_scores.append((confidence_metric.agent_id, score))
        
        # Sort by score (descending - higher is better)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Locally optimal atlases for {node_id}: {[f'{a}:{s:.3f}' for a, s in agent_scores[:3]]}")
        return agent_scores
    
    def subset_ensemble_confidence_scoring(self,
                                         node_id: str,
                                         agent_votes: Dict[str, Any],
                                         local_confidence_metrics: List[LocalConfidenceMetric]) -> Dict[str, Any]:
        """
        Section 7.2: "Subset-Ensemble Confidence Scoring"
        
        "The final confidence score for an included node is calculated as a weighted average
        of the votes it received, where the weights are determined by this MUSE-LLM process."
        
        Args:
            node_id: Node being scored
            agent_votes: Dictionary of agent votes/proposals
            local_confidence_metrics: Local confidence assessments
            
        Returns:
            Dictionary containing confidence score and detailed analysis
        """
        if not agent_votes or not local_confidence_metrics:
            return {
                'final_confidence_score': 0.0,
                'voting_agents': [],
                'confidence_breakdown': {},
                'ensemble_analysis': {}
            }
        
        # Create mapping of agents to their local confidence metrics
        agent_to_confidence = {metric.agent_id: metric for metric in local_confidence_metrics}
        
        # Calculate weighted votes
        weighted_votes = []
        total_weight = 0.0
        confidence_breakdown = {}
        
        for agent_id, vote in agent_votes.items():
            if vote is None:
                continue
                
            confidence_metric = agent_to_confidence.get(agent_id)
            if confidence_metric is None:
                continue
            
            # Calculate agent weight based on MUSE-LLM methodology
            agent_weight = self._calculate_muse_agent_weight(confidence_metric)
            
            weighted_votes.append((agent_id, vote, agent_weight))
            total_weight += agent_weight
            
            confidence_breakdown[agent_id] = {
                'local_confidence': confidence_metric.local_confidence,
                'semantic_similarity': confidence_metric.semantic_similarity,
                'structural_consistency': confidence_metric.structural_consistency,
                'domain_expertise': confidence_metric.domain_expertise,
                'final_weight': agent_weight,
                'vote_content': self._summarize_vote_content(vote)
            }
        
        # Calculate final confidence score
        if total_weight > 0:
            # Weighted average of agent confidence scores
            confidence_scores = [metric.local_confidence * self._calculate_muse_agent_weight(metric) 
                               for metric in local_confidence_metrics]
            final_confidence = sum(confidence_scores) / total_weight
        else:
            final_confidence = 0.0
        
        # Ensemble analysis
        ensemble_analysis = {
            'participating_agents': len(weighted_votes),
            'total_weight': total_weight,
            'average_local_confidence': np.mean([m.local_confidence for m in local_confidence_metrics]),
            'average_semantic_similarity': np.mean([m.semantic_similarity for m in local_confidence_metrics]),
            'consensus_strength': self._calculate_consensus_strength(local_confidence_metrics),
            'diversity_score': self._calculate_agent_diversity_score(weighted_votes)
        }
        
        result = {
            'final_confidence_score': final_confidence,
            'voting_agents': [agent_id for agent_id, _, _ in weighted_votes],
            'confidence_breakdown': confidence_breakdown,
            'ensemble_analysis': ensemble_analysis,
            'muse_methodology': True
        }
        
        logger.info(f"MUSE-LLM confidence for {node_id}: {final_confidence:.3f} ({len(weighted_votes)} agents)")
        return result
    
    def batch_muse_consensus(self,
                           candidate_nodes: Dict[str, Dict[str, Any]],
                           embedding_service,
                           context_graph: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Apply MUSE-LLM methodology to batch of candidate nodes
        
        Args:
            candidate_nodes: Dictionary mapping node_id to agent_proposals
            embedding_service: Embedding service for semantic analysis
            context_graph: Optional graph of node relationships
            
        Returns:
            Complete MUSE consensus results
        """
        consensus_results = {
            'node_confidences': {},
            'locally_optimal_selections': {},
            'ensemble_statistics': {},
            'methodology_metadata': {
                'algorithm': 'MUSE-LLM',
                'total_nodes': len(candidate_nodes),
                'total_atlases': len(self.ensemble_atlases),
                'local_similarity_window': self.local_similarity_window
            }
        }
        
        # Process each node with MUSE methodology
        for node_id, agent_proposals in candidate_nodes.items():
            # Get context nodes if graph is provided
            context_nodes = context_graph.get(node_id, []) if context_graph else []
            
            # Step 1: Assess local confidence
            local_confidences = self.assess_local_confidence(
                node_id, agent_proposals, embedding_service, context_nodes
            )
            
            # Step 2: Locally optimal atlas selection
            optimal_atlases = self.locally_optimal_atlas_selection(
                node_id, local_confidences
            )
            
            # Step 3: Subset-ensemble confidence scoring
            confidence_result = self.subset_ensemble_confidence_scoring(
                node_id, agent_proposals, local_confidences
            )
            
            # Store results
            consensus_results['node_confidences'][node_id] = confidence_result
            consensus_results['locally_optimal_selections'][node_id] = optimal_atlases
        
        # Calculate ensemble statistics
        all_confidences = [result['final_confidence_score'] 
                          for result in consensus_results['node_confidences'].values()]
        
        consensus_results['ensemble_statistics'] = {
            'average_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'confidence_std': np.std(all_confidences) if all_confidences else 0.0,
            'high_confidence_nodes': sum(1 for c in all_confidences if c > 0.7),
            'low_confidence_nodes': sum(1 for c in all_confidences if c < 0.3),
            'consensus_distribution': self._calculate_confidence_distribution(all_confidences)
        }
        
        logger.info(f"MUSE-LLM batch consensus: {len(candidate_nodes)} nodes, avg confidence: {consensus_results['ensemble_statistics']['average_confidence']:.3f}")
        return consensus_results
    
    def _extract_node_content(self, proposal: Any) -> str:
        """Extract textual content from node proposal"""
        if isinstance(proposal, str):
            return proposal
        elif isinstance(proposal, dict):
            # Extract key fields for content representation
            content_fields = ['name', 'description', 'content', 'value', 'label']
            content_parts = []
            
            for field in content_fields:
                if field in proposal and proposal[field]:
                    content_parts.append(f"{field}: {proposal[field]}")
            
            return " | ".join(content_parts) if content_parts else str(proposal)
        else:
            return str(proposal)
    
    def _extract_agent_confidence(self, agent_id: str, proposal: Any) -> float:
        """Extract confidence score from agent proposal"""
        if isinstance(proposal, dict):
            # Look for common confidence indicators
            confidence_keys = ['confidence', 'score', 'probability', 'certainty', 'weight']
            
            for key in confidence_keys:
                if key in proposal:
                    try:
                        return float(proposal[key])
                    except (ValueError, TypeError):
                        continue
        
        # Fallback: use atlas performance profile
        atlas = self.ensemble_atlases.get(agent_id)
        if atlas:
            return atlas.performance_profile.get('accuracy', 0.5)
        
        return 0.5  # Default neutral confidence
    
    def _assess_structural_consistency(self, agent_id: str, proposal: Any, context_nodes: List[str]) -> float:
        """Assess structural consistency of proposal with context"""
        # Simplified structural assessment
        # In full implementation, would analyze hierarchical relationships
        
        if not context_nodes:
            return 0.5  # Neutral if no context
        
        # Check if proposal has reasonable structural properties
        consistency_score = 0.5
        
        if isinstance(proposal, dict):
            # Has structured content
            consistency_score += 0.2
            
            # Has hierarchical indicators
            hierarchical_keys = ['parent', 'children', 'level', 'depth']
            if any(key in proposal for key in hierarchical_keys):
                consistency_score += 0.2
                
            # Has appropriate fields
            expected_keys = ['description', 'type', 'category']
            if any(key in proposal for key in expected_keys):
                consistency_score += 0.1
        
        return min(consistency_score, 1.0)
    
    def _calculate_domain_expertise_score(self, agent_id: str, node_id: str, atlas: EnsembleAtlas) -> float:
        """Calculate domain expertise score for agent-node pair"""
        # Extract domain indicators from node_id
        node_domain = self._infer_node_domain(node_id)
        
        # Check if agent has expertise in this domain
        if node_domain in atlas.specialization_domains or 'general' in atlas.specialization_domains:
            base_expertise = atlas.performance_profile.get('domain_expertise', 0.5)
            return min(base_expertise * 1.2, 1.0)  # 20% bonus for domain match
        
        return atlas.performance_profile.get('domain_expertise', 0.3)  # Reduced for out-of-domain
    
    def _infer_node_domain(self, node_id: str) -> str:
        """Infer domain from node identifier"""
        # Simple domain inference based on node_id keywords
        domain_keywords = {
            'security': ['security', 'threat', 'malware', 'attack', 'vulnerability'],
            'network': ['network', 'ip', 'dns', 'tcp', 'udp', 'protocol'],
            'system': ['system', 'process', 'registry', 'file', 'service'],
            'behavioral': ['behavior', 'pattern', 'anomaly', 'activity']
        }
        
        node_lower = node_id.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in node_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _calculate_muse_agent_weight(self, confidence_metric: LocalConfidenceMetric) -> float:
        """Calculate MUSE-style weight for agent"""
        # Weighted combination of MUSE factors
        weight = (
            0.4 * confidence_metric.local_confidence +
            0.3 * confidence_metric.semantic_similarity +
            0.2 * confidence_metric.structural_consistency +
            0.1 * confidence_metric.domain_expertise
        )
        
        return max(weight, 0.01)  # Minimum weight threshold
    
    def _summarize_vote_content(self, vote: Any) -> str:
        """Create summary of vote content"""
        content = self._extract_node_content(vote)
        return content[:100] + "..." if len(content) > 100 else content
    
    def _calculate_consensus_strength(self, confidence_metrics: List[LocalConfidenceMetric]) -> float:
        """Calculate strength of consensus among agents"""
        if not confidence_metrics:
            return 0.0
        
        confidences = [m.local_confidence for m in confidence_metrics]
        similarities = [m.semantic_similarity for m in confidence_metrics]
        
        # High consensus = high average confidence + low variance + high similarity
        avg_confidence = np.mean(confidences)
        confidence_variance = np.var(confidences)
        avg_similarity = np.mean(similarities)
        
        consensus_strength = 0.5 * avg_confidence + 0.3 * avg_similarity + 0.2 * (1 - confidence_variance)
        return max(min(consensus_strength, 1.0), 0.0)
    
    def _calculate_agent_diversity_score(self, weighted_votes: List[Tuple[str, Any, float]]) -> float:
        """Calculate diversity of agent ensemble"""
        if len(weighted_votes) <= 1:
            return 0.0
        
        # Diversity based on different architectures and training characteristics
        architectures = set()
        training_chars = set()
        
        for agent_id, _, _ in weighted_votes:
            atlas = self.ensemble_atlases.get(agent_id)
            if atlas:
                architectures.add(atlas.architecture_type)
                training_chars.add(atlas.training_characteristics)
        
        # Normalized diversity score
        max_diversity = len(weighted_votes)
        actual_diversity = len(architectures) + len(training_chars)
        
        return min(actual_diversity / (2 * max_diversity), 1.0)
    
    def _calculate_confidence_distribution(self, confidences: List[float]) -> Dict[str, int]:
        """Calculate distribution of confidence scores"""
        distribution = {
            'very_low': 0,    # 0.0 - 0.2
            'low': 0,         # 0.2 - 0.4
            'medium': 0,      # 0.4 - 0.6
            'high': 0,        # 0.6 - 0.8
            'very_high': 0    # 0.8 - 1.0
        }
        
        for conf in confidences:
            if conf < 0.2:
                distribution['very_low'] += 1
            elif conf < 0.4:
                distribution['low'] += 1
            elif conf < 0.6:
                distribution['medium'] += 1
            elif conf < 0.8:
                distribution['high'] += 1
            else:
                distribution['very_high'] += 1
        
        return distribution