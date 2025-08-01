#!/usr/bin/env python3
"""
BFT Consensus Quality Tracking Metrics (Section 9.2)

This module implements consensus-specific metrics:
- Agreement levels and fault detection accuracy
- Consensus time and reliability measurements
- Byzantine fault tolerance effectiveness validation
- Model reliability scoring and trust assessment
"""

import json
import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class ConsensusMetrics:
    """Container for consensus quality metrics"""
    # Agreement Levels
    consensus_strength: float = 0.0
    agreement_level: float = 0.0
    fault_detection_accuracy: float = 0.0
    
    # Consensus Time and Reliability
    consensus_time: float = 0.0
    convergence_rounds: int = 0
    reliability_score: float = 0.0
    
    # Byzantine Fault Tolerance
    bft_effectiveness: float = 0.0
    detected_byzantine_nodes: int = 0
    fault_tolerance_threshold: float = 0.0
    
    # Model Reliability
    model_trust_scores: Dict[str, float] = None
    overall_trust_score: float = 0.0
    
    # Consensus Protocol Performance
    pbft_performance: Dict[str, Any] = None
    hashgraph_performance: Dict[str, Any] = None
    algorand_performance: Dict[str, Any] = None
    
    # Additional metrics
    consensus_efficiency: float = 0.0
    message_complexity: int = 0
    scalability_score: float = 0.0
    
    def __post_init__(self):
        if self.model_trust_scores is None:
            self.model_trust_scores = {}
        if self.pbft_performance is None:
            self.pbft_performance = {}
        if self.hashgraph_performance is None:
            self.hashgraph_performance = {}
        if self.algorand_performance is None:
            self.algorand_performance = {}

class ConsensusQualityTracker:
    """BFT consensus quality tracking and assessment"""
    
    def __init__(self):
        """Initialize consensus quality tracker"""
        self.consensus_history = []
        self.performance_cache = {}
        
    def evaluate_consensus_quality(
        self,
        consensus_output: Dict[str, Any],
        processing_metadata: Dict[str, Any],
        original_inputs: List[Dict[str, Any]]
    ) -> ConsensusMetrics:
        """
        Comprehensive evaluation of consensus quality
        
        Args:
            consensus_output: Output from consensus system
            processing_metadata: Metadata about consensus process
            original_inputs: Original input files
            
        Returns:
            ConsensusMetrics with all computed metrics
        """
        logger.info("Starting consensus quality evaluation")
        
        metrics = ConsensusMetrics()
        
        try:
            # Extract consensus-specific data
            consensus_data = self._extract_consensus_data(consensus_output, processing_metadata)
            
            # 1. Agreement Levels Assessment
            metrics.consensus_strength = self._calculate_consensus_strength(consensus_data)
            metrics.agreement_level = self._calculate_agreement_level(consensus_data, original_inputs)
            metrics.fault_detection_accuracy = self._calculate_fault_detection_accuracy(consensus_data)
            
            # 2. Consensus Time and Reliability  
            metrics.consensus_time = processing_metadata.get('total_processing_time', 0.0)
            metrics.convergence_rounds = self._calculate_convergence_rounds(consensus_data)
            metrics.reliability_score = self._calculate_consensus_reliability(consensus_data)
            
            # 3. Byzantine Fault Tolerance Effectiveness
            metrics.bft_effectiveness = self._calculate_bft_effectiveness(consensus_data)
            metrics.detected_byzantine_nodes = self._count_byzantine_nodes(consensus_data)
            metrics.fault_tolerance_threshold = self._get_fault_tolerance_threshold(consensus_data)
            
            # 4. Model Reliability Scoring
            metrics.model_trust_scores = self._calculate_model_trust_scores(consensus_data, original_inputs)
            metrics.overall_trust_score = statistics.mean(metrics.model_trust_scores.values()) if metrics.model_trust_scores else 0.0
            
            # 5. Protocol Performance Analysis
            metrics.pbft_performance = self._analyze_pbft_performance(consensus_data)
            metrics.hashgraph_performance = self._analyze_hashgraph_performance(consensus_data)
            metrics.algorand_performance = self._analyze_algorand_performance(consensus_data)
            
            # 6. Additional Metrics
            metrics.consensus_efficiency = self._calculate_consensus_efficiency(metrics)
            metrics.message_complexity = self._calculate_message_complexity(consensus_data)
            metrics.scalability_score = self._calculate_scalability_score(metrics, len(original_inputs))
            
            logger.info(f"Consensus quality evaluation completed. Overall trust: {metrics.overall_trust_score:.3f}")
            
        except Exception as e:
            logger.error(f"Consensus quality evaluation failed: {e}")
        
        return metrics
    
    def _extract_consensus_data(
        self, 
        consensus_output: Dict[str, Any], 
        processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract consensus-specific data from output"""
        consensus_data = {
            'quality_metrics': consensus_output.get('quality_metrics', {}),
            'processing_metadata': processing_metadata,
            'part_contributions': consensus_output.get('part_contributions', {}),
            'consensus_parts_count': processing_metadata.get('consensus_parts_count', 0),
            'original_file_count': processing_metadata.get('original_file_count', 0)
        }
        
        # Look for research analysis data
        if hasattr(consensus_output, 'research_analysis'):
            consensus_data['research_analysis'] = consensus_output.research_analysis
        elif 'research_analysis' in consensus_output:
            consensus_data['research_analysis'] = consensus_output['research_analysis']
        
        return consensus_data
    
    def _calculate_consensus_strength(self, consensus_data: Dict[str, Any]) -> float:
        """Calculate overall consensus strength"""
        try:
            qm = consensus_data.get('quality_metrics', {})
            
            # Multiple indicators of consensus strength
            strength_indicators = []
            
            # Success rate indicator
            success_rate = qm.get('consensus_success_rate', 0.0)
            strength_indicators.append(success_rate)
            
            # Confidence indicator
            confidence = float(qm.get('average_consensus_confidence', 0.0))
            strength_indicators.append(confidence)
            
            # Source diversity indicator (higher diversity = stronger consensus)
            source_diversity = qm.get('source_diversity', 0.0)
            strength_indicators.append(source_diversity)
            
            # Data preservation indicator
            preservation = qm.get('data_preservation', 0.0)
            strength_indicators.append(preservation)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]  # Emphasize success rate and confidence
            strength = sum(w * s for w, s in zip(weights, strength_indicators))
            
            return min(1.0, max(0.0, strength)) #Why? can we use sigmoid here?
            
        except Exception as e:
            logger.error(f"Error calculating consensus strength: {e}")
            return 0.0
    
    def _calculate_agreement_level(
        self, 
        consensus_data: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> float:
        """Calculate agreement level between original inputs"""
        try:
            if len(original_inputs) < 2:
                return 1.0  # Perfect agreement with single input
            
            # Extract comparable elements from each input
            input_elements = []
            for input_data in original_inputs:
                elements = self._extract_comparable_elements(input_data)
                input_elements.append(elements)
            
            # Calculate pairwise agreement
            agreement_scores = []
            for i in range(len(input_elements)):
                for j in range(i + 1, len(input_elements)):
                    agreement = self._calculate_pairwise_agreement(
                        input_elements[i], input_elements[j]
                    )
                    agreement_scores.append(agreement)
            
            return statistics.mean(agreement_scores) if agreement_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating agreement level: {e}")
            return 0.0
    
    def _extract_comparable_elements(self, input_data: Dict[str, Any]) -> set:
        """Extract elements that can be compared across inputs"""
        elements = set()
        
        # Extract from parsed_json if available
        parsed_json = input_data.get('parsed_json', input_data)
        
        # Extract key structural elements
        if isinstance(parsed_json, dict):
            for key, value in parsed_json.items():
                if isinstance(value, dict):
                    # Add nested keys
                    for nested_key in value.keys():
                        elements.add(f"{key}.{nested_key}")
                elif isinstance(value, list):
                    # Add list structure info
                    elements.add(f"{key}[list:{len(value)}]")
                else:
                    # Add key-value pairs
                    elements.add(f"{key}:{str(value)[:50]}")  # Truncate long values
        
        return elements
    
    def _calculate_pairwise_agreement(self, elements1: set, elements2: set) -> float:
        """Calculate agreement between two sets of elements"""
        if not elements1 and not elements2:
            return 1.0
        
        intersection = len(elements1.intersection(elements2))
        union = len(elements1.union(elements2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_fault_detection_accuracy(self, consensus_data: Dict[str, Any]) -> float:
        """Calculate accuracy of fault detection mechanisms"""
        try:
            # Look for research analysis data about BFT
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'bft_analysis' in research_analysis:
                bft_data = research_analysis['bft_analysis']
                
                # Extract fault detection metrics
                detected_faults = bft_data.get('fault_nodes_detected', 0)
                total_nodes = bft_data.get('total_nodes', 1)
                byzantine_resilience = bft_data.get('byzantine_resilience', 0.0)
                
                # Calculate accuracy based on expected vs detected faults
                # In test scenarios, we expect low fault rates
                expected_fault_rate = 0.1  # 10% expected fault rate
                actual_fault_rate = detected_faults / total_nodes
                
                # Accuracy is how close we are to expected fault detection
                accuracy = 1.0 - abs(expected_fault_rate - actual_fault_rate)
                
                # Incorporate byzantine resilience score
                accuracy = (accuracy + byzantine_resilience) / 2.0
                
                return max(0.0, min(1.0, accuracy))
            
            # Fallback: use consensus success rate as proxy
            success_rate = consensus_data.get('quality_metrics', {}).get('consensus_success_rate', 0.0)
            return success_rate
            
        except Exception as e:
            logger.error(f"Error calculating fault detection accuracy: {e}")
            return 0.0
    
    def _calculate_convergence_rounds(self, consensus_data: Dict[str, Any]) -> int:
        """Calculate number of consensus rounds to convergence"""
        try:
            # Look for research analysis data
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'bft_analysis' in research_analysis:
                bft_data = research_analysis['bft_analysis']
                rounds = bft_data.get('consensus_rounds', 1)
                return max(1, rounds)
            
            # Fallback: estimate based on processing complexity
            parts_count = consensus_data.get('consensus_parts_count', 0)
            file_count = consensus_data.get('original_file_count', 1)
            
            # Simple heuristic: more parts and files = more rounds
            estimated_rounds = max(1, min(10, (parts_count // file_count) // 10))
            return estimated_rounds
            
        except Exception as e:
            logger.error(f"Error calculating convergence rounds: {e}")
            return 1
    
    def _calculate_consensus_reliability(self, consensus_data: Dict[str, Any]) -> float:
        """Calculate overall consensus reliability"""
        try:
            reliability_factors = []
            
            qm = consensus_data.get('quality_metrics', {})
            
            # Factor 1: Success rate
            success_rate = qm.get('consensus_success_rate', 0.0)
            reliability_factors.append(success_rate)
            
            # Factor 2: Confidence consistency
            confidence = float(qm.get('average_consensus_confidence', 0.0))
            reliability_factors.append(confidence)
            
            # Factor 3: Data preservation
            preservation = qm.get('data_preservation', 0.0)
            reliability_factors.append(preservation)
            
            # Factor 4: Source diversity (more sources = more reliable)
            diversity = qm.get('source_diversity', 0.0)
            reliability_factors.append(diversity)
            
            # Factor 5: Processing stability (consistent processing time)
            processing_time = consensus_data.get('processing_metadata', {}).get('total_processing_time', 0.0)
            if processing_time > 0:
                # Reliability decreases with very long or very short processing times
                time_reliability = self._calculate_time_reliability(processing_time)
                reliability_factors.append(time_reliability)
            
            return statistics.mean(reliability_factors) if reliability_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating consensus reliability: {e}")
            return 0.0
    
    def _calculate_time_reliability(self, processing_time: float) -> float:
        """Calculate reliability based on processing time"""
        # Optimal processing time range: 1-10 seconds
        if 1.0 <= processing_time <= 10.0:
            return 1.0
        elif processing_time < 1.0:
            # Too fast might indicate insufficient processing
            return 0.5 + (processing_time / 2.0)
        else:
            # Too slow indicates inefficiency
            return max(0.1, 1.0 - ((processing_time - 10.0) / 100.0))
    
    def _calculate_bft_effectiveness(self, consensus_data: Dict[str, Any]) -> float:
        """Calculate Byzantine Fault Tolerance effectiveness"""
        try:
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'bft_analysis' in research_analysis:
                bft_data = research_analysis['bft_analysis']
                
                # Direct BFT metrics
                byzantine_resilience = bft_data.get('byzantine_resilience', 0.0)
                fault_nodes = bft_data.get('fault_nodes_detected', 0)
                total_nodes = bft_data.get('total_nodes', 1)
                
                # Calculate effectiveness based on:
                # 1. Byzantine resilience score
                # 2. Appropriate fault detection (not too many false positives)
                # 3. System still functioning despite faults
                
                fault_rate = fault_nodes / total_nodes
                appropriate_detection = 1.0 - abs(fault_rate - 0.05)  # Expect ~5% fault rate
                
                effectiveness = (byzantine_resilience + appropriate_detection) / 2.0
                
                return max(0.0, min(1.0, effectiveness))
            
            # Fallback: use consensus quality as proxy for BFT effectiveness
            qm = consensus_data.get('quality_metrics', {})
            overall_quality = float(qm.get('overall_quality', 0.0))
            
            return overall_quality
            
        except Exception as e:
            logger.error(f"Error calculating BFT effectiveness: {e}")
            return 0.0
    
    def _count_byzantine_nodes(self, consensus_data: Dict[str, Any]) -> int:
        """Count detected Byzantine nodes"""
        try:
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'bft_analysis' in research_analysis:
                bft_data = research_analysis['bft_analysis']
                return bft_data.get('fault_nodes_detected', 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error counting Byzantine nodes: {e}")
            return 0
    
    def _get_fault_tolerance_threshold(self, consensus_data: Dict[str, Any]) -> float:
        """Get the fault tolerance threshold used"""
        try:
            # Look for BFT configuration
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'bft_analysis' in research_analysis:
                bft_data = research_analysis['bft_analysis']
                return bft_data.get('fault_tolerance_threshold', 0.33)
            
            # Default BFT threshold (< 1/3 nodes can be faulty)
            return 0.33
            
        except Exception as e:
            logger.error(f"Error getting fault tolerance threshold: {e}")
            return 0.33
    
    def _calculate_model_trust_scores(
        self, 
        consensus_data: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate trust scores for each model/provider"""
        trust_scores = {}
        
        try:
            # Extract provider information
            providers = set()
            for input_data in original_inputs:
                provider = input_data.get('provider', input_data.get('model', 'unknown'))
                providers.add(provider)
            
            # Look for contribution data
            part_contributions = consensus_data.get('part_contributions', {})
            
            for provider in providers:
                trust_score = self._calculate_individual_trust_score(
                    provider, part_contributions, consensus_data
                )
                trust_scores[provider] = trust_score
            
        except Exception as e:
            logger.error(f"Error calculating model trust scores: {e}")
        
        return trust_scores
    
    def _calculate_individual_trust_score(
        self, 
        provider: str, 
        part_contributions: Dict[str, List[str]], 
        consensus_data: Dict[str, Any]
    ) -> float:
        """Calculate trust score for individual provider"""
        try:
            trust_factors = []
            
            # Factor 1: Contribution frequency
            provider_contributions = sum(
                1 for contributors in part_contributions.values()
                if provider in contributors
            )
            total_parts = len(part_contributions)
            
            if total_parts > 0:
                contribution_ratio = provider_contributions / total_parts
                trust_factors.append(contribution_ratio)
            
            # Factor 2: Quality of contributions
            # (This is simplified - could be enhanced with more sophisticated analysis)
            qm = consensus_data.get('quality_metrics', {})
            overall_quality = float(qm.get('overall_quality', 0.5))
            trust_factors.append(overall_quality)
            
            # Factor 3: Consistency (based on consensus success rate)
            success_rate = qm.get('consensus_success_rate', 0.5)
            trust_factors.append(success_rate)
            
            # Factor 4: Reliability (no Byzantine behavior detected)
            detected_byzantine = self._count_byzantine_nodes(consensus_data)
            reliability_factor = 1.0 if detected_byzantine == 0 else 0.7
            trust_factors.append(reliability_factor)
            
            return statistics.mean(trust_factors) if trust_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating trust score for {provider}: {e}")
            return 0.5
    
    def _analyze_pbft_performance(self, consensus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PBFT protocol performance"""
        try:
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'bft_analysis' in research_analysis:
                bft_data = research_analysis['bft_analysis']
                
                return {
                    'rounds_completed': bft_data.get('consensus_rounds', 1),
                    'message_count': bft_data.get('pbft_messages', 0),
                    'view_changes': bft_data.get('view_changes', 0),
                    'performance_score': self._calculate_pbft_score(bft_data)
                }
            
            # Fallback performance analysis
            return {
                'rounds_completed': 1,
                'message_count': consensus_data.get('consensus_parts_count', 0) * 3,  # Estimate
                'view_changes': 0,
                'performance_score': 0.7  # Default moderate performance
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PBFT performance: {e}")
            return {'performance_score': 0.0}
    
    def _calculate_pbft_score(self, bft_data: Dict[str, Any]) -> float:
        """Calculate PBFT performance score"""
        try:
            score_factors = []
            
            # Factor 1: Low number of rounds (faster convergence)
            rounds = bft_data.get('consensus_rounds', 1)
            rounds_score = max(0.0, 1.0 - (rounds - 1) * 0.1)  # Penalty for extra rounds
            score_factors.append(rounds_score)
            
            # Factor 2: No view changes (stable leadership)
            view_changes = bft_data.get('view_changes', 0)
            view_score = 1.0 if view_changes == 0 else max(0.1, 1.0 - view_changes * 0.2)
            score_factors.append(view_score)
            
            # Factor 3: Byzantine resilience
            resilience = bft_data.get('byzantine_resilience', 0.0)
            score_factors.append(resilience)
            
            return statistics.mean(score_factors)
            
        except Exception as e:
            logger.error(f"Error calculating PBFT score: {e}")
            return 0.0
    
    def _analyze_hashgraph_performance(self, consensus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Hashgraph protocol performance"""
        try:
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'hashgraph_analysis' in research_analysis:
                hg_data = research_analysis['hashgraph_analysis']
                
                return {
                    'gossip_rounds': hg_data.get('gossip_rounds', 1),
                    'virtual_votes': hg_data.get('virtual_votes', 0),
                    'consensus_events': hg_data.get('consensus_events', 0),
                    'performance_score': self._calculate_hashgraph_score(hg_data)
                }
            
            # Fallback performance analysis
            return {
                'gossip_rounds': 2,  # Typical for small networks
                'virtual_votes': consensus_data.get('consensus_parts_count', 0),
                'consensus_events': consensus_data.get('consensus_parts_count', 0),
                'performance_score': 0.8  # Hashgraph typically performs well
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Hashgraph performance: {e}")
            return {'performance_score': 0.0}
    
    def _calculate_hashgraph_score(self, hg_data: Dict[str, Any]) -> float:
        """Calculate Hashgraph performance score"""
        try:
            score_factors = []
            
            # Factor 1: Efficient gossip (low rounds for convergence)
            gossip_rounds = hg_data.get('gossip_rounds', 1)
            gossip_score = max(0.0, 1.0 - (gossip_rounds - 1) * 0.05)
            score_factors.append(gossip_score)
            
            # Factor 2: High virtual voting efficiency
            virtual_votes = hg_data.get('virtual_votes', 0)
            consensus_events = hg_data.get('consensus_events', 1)
            vote_efficiency = min(1.0, virtual_votes / consensus_events) if consensus_events > 0 else 0.0
            score_factors.append(vote_efficiency)
            
            # Factor 3: Asynchronous Byzantine Fault Tolerance
            abft_score = hg_data.get('abft_effectiveness', 0.8)  # Typically high
            score_factors.append(abft_score)
            
            return statistics.mean(score_factors)
            
        except Exception as e:
            logger.error(f"Error calculating Hashgraph score: {e}")
            return 0.0
    
    def _analyze_algorand_performance(self, consensus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Algorand PoS protocol performance"""
        try:
            research_analysis = consensus_data.get('research_analysis', {})
            
            if 'algorand_analysis' in research_analysis:
                algo_data = research_analysis['algorand_analysis']
                
                return {
                    'committee_size': algo_data.get('committee_size', 5),
                    'vrf_rounds': algo_data.get('vrf_rounds', 1),
                    'stake_distribution': algo_data.get('stake_distribution', {}),
                    'performance_score': self._calculate_algorand_score(algo_data)
                }
            
            # Fallback performance analysis
            return {
                'committee_size': max(3, consensus_data.get('original_file_count', 3)),
                'vrf_rounds': 1,
                'stake_distribution': {'uniform': 1.0},
                'performance_score': 0.75  # Good default for PoS
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Algorand performance: {e}")
            return {'performance_score': 0.0}
    
    def _calculate_algorand_score(self, algo_data: Dict[str, Any]) -> float:
        """Calculate Algorand performance score"""
        try:
            score_factors = []
            
            # Factor 1: Appropriate committee size
            committee_size = algo_data.get('committee_size', 5)
            size_score = 1.0 if 3 <= committee_size <= 20 else 0.5
            score_factors.append(size_score)
            
            # Factor 2: VRF efficiency (low rounds)
            vrf_rounds = algo_data.get('vrf_rounds', 1)
            vrf_score = max(0.0, 1.0 - (vrf_rounds - 1) * 0.1)
            score_factors.append(vrf_score)
            
            # Factor 3: Stake distribution fairness
            stake_dist = algo_data.get('stake_distribution', {})
            if stake_dist:
                # Measure how evenly distributed the stake is
                stakes = list(stake_dist.values())
                stake_fairness = 1.0 - np.std(stakes) if len(stakes) > 1 else 1.0
                score_factors.append(max(0.0, min(1.0, stake_fairness)))
            
            return statistics.mean(score_factors)
            
        except Exception as e:
            logger.error(f"Error calculating Algorand score: {e}")
            return 0.0
    
    def _calculate_consensus_efficiency(self, metrics: ConsensusMetrics) -> float:
        """Calculate overall consensus efficiency"""
        try:
            efficiency_factors = []
            
            # Factor 1: Time efficiency (faster is better)
            if metrics.consensus_time > 0:
                time_efficiency = min(1.0, 10.0 / metrics.consensus_time)  # 10s as reference
                efficiency_factors.append(time_efficiency)
            
            # Factor 2: Round efficiency (fewer rounds is better)  
            if metrics.convergence_rounds > 0:
                round_efficiency = min(1.0, 3.0 / metrics.convergence_rounds)  # 3 rounds as reference
                efficiency_factors.append(round_efficiency)
            
            # Factor 3: Quality efficiency (quality per unit time)
            if metrics.consensus_time > 0 and metrics.consensus_strength > 0:
                quality_efficiency = metrics.consensus_strength / (metrics.consensus_time / 10.0)
                quality_efficiency = min(1.0, quality_efficiency)
                efficiency_factors.append(quality_efficiency)
            
            # Factor 4: Trust efficiency
            efficiency_factors.append(metrics.overall_trust_score)
            
            return statistics.mean(efficiency_factors) if efficiency_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating consensus efficiency: {e}")
            return 0.0
    
    def _calculate_message_complexity(self, consensus_data: Dict[str, Any]) -> int:
        """Calculate message complexity for consensus protocols"""
        try:
            # Estimate based on consensus parts and file count
            parts_count = consensus_data.get('consensus_parts_count', 0)
            file_count = consensus_data.get('original_file_count', 1)
            
            # PBFT: O(n^2) messages per consensus instance
            # Simplified estimation
            estimated_messages = parts_count * (file_count ** 2) * 3  # 3 phases in PBFT
            
            return max(0, estimated_messages)
            
        except Exception as e:
            logger.error(f"Error calculating message complexity: {e}")
            return 0
    
    def _calculate_scalability_score(self, metrics: ConsensusMetrics, node_count: int) -> float:
        """Calculate scalability score based on performance with node count"""
        try:
            # Scalability factors
            scalability_factors = []
            
            # Factor 1: Efficiency doesn't degrade too much with more nodes
            expected_efficiency = max(0.1, 1.0 - (node_count - 1) * 0.1)  # Expected degradation
            actual_efficiency = metrics.consensus_efficiency
            efficiency_ratio = actual_efficiency / expected_efficiency if expected_efficiency > 0 else 0.0
            scalability_factors.append(min(1.0, efficiency_ratio))
            
            # Factor 2: Consensus strength remains high
            scalability_factors.append(metrics.consensus_strength)
            
            # Factor 3: Trust scores remain stable
            scalability_factors.append(metrics.overall_trust_score)
            
            # Factor 4: BFT effectiveness (essential for scalability)
            scalability_factors.append(metrics.bft_effectiveness)
            
            return statistics.mean(scalability_factors) if scalability_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating scalability score: {e}")
            return 0.0

def create_consensus_report(metrics: ConsensusMetrics, output_file: Optional[str] = None) -> str:
    """Create a comprehensive consensus quality report"""
    
    report = f"""
# Consensus Quality Assessment Report

## Overall Consensus Performance

### Consensus Strength: {metrics.consensus_strength:.3f}
### Overall Trust Score: {metrics.overall_trust_score:.3f}
### Consensus Efficiency: {metrics.consensus_efficiency:.3f}

## Agreement and Fault Tolerance

### Agreement Level: {metrics.agreement_level:.3f}
### Fault Detection Accuracy: {metrics.fault_detection_accuracy:.3f}
### BFT Effectiveness: {metrics.bft_effectiveness:.3f}
### Detected Byzantine Nodes: {metrics.detected_byzantine_nodes}
### Fault Tolerance Threshold: {metrics.fault_tolerance_threshold:.3f}

## Performance Metrics

### Consensus Time: {metrics.consensus_time:.2f}s
### Convergence Rounds: {metrics.convergence_rounds}
### Message Complexity: {metrics.message_complexity:,}
### Scalability Score: {metrics.scalability_score:.3f}
### Reliability Score: {metrics.reliability_score:.3f}

## Protocol Performance Analysis

### PBFT Performance
- Score: {metrics.pbft_performance.get('performance_score', 'N/A')}
- Rounds: {metrics.pbft_performance.get('rounds_completed', 'N/A')}
- Messages: {metrics.pbft_performance.get('message_count', 'N/A')}
- View Changes: {metrics.pbft_performance.get('view_changes', 'N/A')}

### Hashgraph Performance  
- Score: {metrics.hashgraph_performance.get('performance_score', 'N/A')}
- Gossip Rounds: {metrics.hashgraph_performance.get('gossip_rounds', 'N/A')}
- Virtual Votes: {metrics.hashgraph_performance.get('virtual_votes', 'N/A')}
- Consensus Events: {metrics.hashgraph_performance.get('consensus_events', 'N/A')}

### Algorand Performance
- Score: {metrics.algorand_performance.get('performance_score', 'N/A')}
- Committee Size: {metrics.algorand_performance.get('committee_size', 'N/A')}
- VRF Rounds: {metrics.algorand_performance.get('vrf_rounds', 'N/A')}

## Model Trust Scores

{_format_trust_scores(metrics.model_trust_scores)}

## Assessment Summary

{_generate_consensus_assessment(metrics)}

## Recommendations

{_generate_consensus_recommendations(metrics)}
"""
    
    if output_file:
        # Remove Unicode characters for Windows compatibility
        safe_report = report.replace('✓', 'SUCCESS').replace('⚠', 'WARNING').replace('✗', 'FAILED')
        Path(output_file).write_text(safe_report, encoding='utf-8')
        logger.info(f"Consensus report saved to {output_file}")
    
    return report

def _format_trust_scores(trust_scores: Dict[str, float]) -> str:
    """Format trust scores for display"""
    if not trust_scores:
        return "No trust scores available"
    
    lines = []
    for provider, score in sorted(trust_scores.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {provider}: {score:.3f}")
    
    return "\n".join(lines)

def _generate_consensus_assessment(metrics: ConsensusMetrics) -> str:
    """Generate overall consensus assessment"""
    assessment_lines = []
    
    # Overall performance
    if metrics.consensus_strength >= 0.8:
        assessment_lines.append("✓ EXCELLENT consensus strength")
    elif metrics.consensus_strength >= 0.6:
        assessment_lines.append("✓ GOOD consensus strength")
    else:
        assessment_lines.append("⚠ LOW consensus strength")
    
    # Trust assessment
    if metrics.overall_trust_score >= 0.8:
        assessment_lines.append("✓ HIGH model trust scores")
    elif metrics.overall_trust_score >= 0.6:
        assessment_lines.append("✓ MODERATE model trust scores")
    else:
        assessment_lines.append("⚠ LOW model trust scores")
    
    # BFT effectiveness
    if metrics.bft_effectiveness >= 0.8:
        assessment_lines.append("✓ EXCELLENT Byzantine fault tolerance")
    elif metrics.bft_effectiveness >= 0.6:
        assessment_lines.append("✓ ADEQUATE Byzantine fault tolerance")
    else:
        assessment_lines.append("⚠ WEAK Byzantine fault tolerance")
    
    # Efficiency assessment
    if metrics.consensus_efficiency >= 0.8:
        assessment_lines.append("✓ HIGH consensus efficiency")
    elif metrics.consensus_efficiency >= 0.6:
        assessment_lines.append("✓ MODERATE consensus efficiency")
    else:
        assessment_lines.append("⚠ LOW consensus efficiency")
    
    return "\n".join(assessment_lines)

def _generate_consensus_recommendations(metrics: ConsensusMetrics) -> str:
    """Generate recommendations based on consensus metrics"""
    recommendations = []
    
    if metrics.consensus_strength < 0.6:
        recommendations.append("- Increase consensus thresholds to improve agreement")
    
    if metrics.bft_effectiveness < 0.7:
        recommendations.append("- Review BFT configuration and fault tolerance settings")
    
    if metrics.consensus_efficiency < 0.6:
        recommendations.append("- Optimize consensus protocols for better time/quality trade-off")
    
    if metrics.convergence_rounds > 5:
        recommendations.append("- Investigate causes of slow convergence")
    
    if metrics.overall_trust_score < 0.7:
        recommendations.append("- Review model reliability and contribution quality")
    
    if metrics.scalability_score < 0.6:
        recommendations.append("- Consider protocol optimizations for better scalability")
    
    if not recommendations:
        recommendations.append("- Continue monitoring consensus performance")
        recommendations.append("- Consider testing with more diverse or challenging inputs")
    
    return "\n".join(recommendations)