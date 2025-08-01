"""
Weighted Voting & Dynamic Reliability System (Section 6.1)
Implementation of reputation-based weighting for LLM agents
"""

import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class AgentPerformanceMetric:
    """Individual performance metric for an agent"""
    timestamp: datetime
    task_type: str
    precision: float
    recall: float
    f1_score: float
    confidence_accuracy: float
    ground_truth_available: bool

@dataclass
class AgentReliabilityProfile:
    """Comprehensive reliability profile for an LLM agent"""
    agent_id: str
    historical_performance: List[AgentPerformanceMetric]
    current_reliability_score: float
    confidence_calibration_score: float
    specialization_domains: List[str]
    last_updated: datetime
    task_count: int

class WeightedVotingReliabilitySystem:
    """
    Section 6.1: Weighted Voting and Dynamic Reliability
    
    Implements dynamic, reputation-based weighting system for LLM agents
    with adaptive reliability scoring and expert-weighted synthesis
    """
    
    def __init__(self, 
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 reliability_decay_days: int = 30,
                 min_reliability_score: float = 0.1,
                 confidence_weight_threshold: float = 0.5):
        """
        Initialize weighted voting system with Section 6.1 parameters
        
        Args:
            alpha: Weight for historical reliability score (Ri)
            beta: Weight for current confidence score (Ci) 
            reliability_decay_days: Days for reliability score decay
            min_reliability_score: Minimum allowed reliability score
            confidence_weight_threshold: Minimum confidence for participation
        """
        self.alpha = alpha  # α in Section 6.1 formula: Wi = αRi + βCi
        self.beta = beta    # β in Section 6.1 formula
        self.reliability_decay_days = reliability_decay_days
        self.min_reliability_score = min_reliability_score
        self.confidence_weight_threshold = confidence_weight_threshold
        
        # Agent profiles storage
        self.agent_profiles: Dict[str, AgentReliabilityProfile] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[AgentPerformanceMetric]] = defaultdict(list)
        
        # Voting thresholds (Section 6.4: Dynamic Support Thresholding)
        self.hierarchical_thresholds = {
            'level_0': 0.67,  # Top-level nodes: >2/3 weighted vote
            'level_1': 0.55,  # Second-level nodes
            'level_2': 0.45,  # Third-level nodes
            'level_3': 0.35,  # Fourth-level nodes
            'default': 0.25   # Deep nodes: >1/4 weighted vote
        }
        
        logger.info(f"Initialized Section 6.1 Weighted Voting: α={alpha}, β={beta}")
    
    def register_agent(self, 
                      agent_id: str, 
                      specialization_domains: List[str] = None,
                      initial_reliability: float = 0.5) -> None:
        """
        Register a new LLM agent in the system
        
        Args:
            agent_id: Unique identifier for the agent
            specialization_domains: List of domain specializations
            initial_reliability: Starting reliability score
        """
        if specialization_domains is None:
            specialization_domains = ['general']
            
        profile = AgentReliabilityProfile(
            agent_id=agent_id,
            historical_performance=[],
            current_reliability_score=initial_reliability,
            confidence_calibration_score=0.5,
            specialization_domains=specialization_domains,
            last_updated=datetime.now(),
            task_count=0
        )
        
        self.agent_profiles[agent_id] = profile
        logger.info(f"Registered agent {agent_id} with domains: {specialization_domains}")
    
    def update_agent_performance(self, 
                                agent_id: str, 
                                performance_metric: AgentPerformanceMetric) -> None:
        """
        Update agent performance with new metric
        
        Args:
            agent_id: Agent identifier
            performance_metric: New performance data
        """
        if agent_id not in self.agent_profiles:
            self.register_agent(agent_id)
            
        profile = self.agent_profiles[agent_id]
        profile.historical_performance.append(performance_metric)
        profile.task_count += 1
        profile.last_updated = datetime.now()
        
        # Recalculate reliability score
        profile.current_reliability_score = self._calculate_reliability_score(agent_id)
        profile.confidence_calibration_score = self._calculate_confidence_calibration(agent_id)
        
        logger.debug(f"Updated {agent_id} performance: R={profile.current_reliability_score:.3f}")
    
    def calculate_agent_voting_weight(self, 
                                    agent_id: str, 
                                    current_confidence: float,
                                    task_domain: str = 'general') -> float:
        """
        Calculate voting weight for agent using Section 6.1 formula: Wi = αRi + βCi
        
        Args:
            agent_id: Agent identifier
            current_confidence: Agent's confidence score for current task (Ci)
            task_domain: Domain of current task for specialization bonus
            
        Returns:
            Voting weight for the agent
        """
        if agent_id not in self.agent_profiles:
            self.register_agent(agent_id)
            
        profile = self.agent_profiles[agent_id]
        
        # Ri: Historical reliability score with temporal decay
        reliability_score = self._get_decayed_reliability_score(agent_id)
        
        # Ci: Calibrated confidence score 
        calibrated_confidence = self._calibrate_confidence_score(agent_id, current_confidence)
        
        # Section 6.1 Formula: Wi = αRi + βCi
        base_weight = self.alpha * reliability_score + self.beta * calibrated_confidence
        
        # Domain specialization bonus
        specialization_bonus = 1.0
        if task_domain in profile.specialization_domains:
            specialization_bonus = 1.2  # 20% bonus for domain expertise
        
        # Apply minimum threshold and specialization
        final_weight = max(base_weight * specialization_bonus, self.min_reliability_score)
        
        # Confidence threshold check
        if current_confidence < self.confidence_weight_threshold:
            final_weight *= 0.5  # Reduce weight for low-confidence contributions
        
        logger.debug(f"Agent {agent_id} weight: {final_weight:.3f} (R={reliability_score:.3f}, C={calibrated_confidence:.3f})")
        return final_weight
    
    def weighted_consensus_vote(self, 
                              votes: Dict[str, Any],
                              confidences: Dict[str, float],
                              hierarchy_level: int = 0,
                              task_domain: str = 'general') -> Dict[str, Any]:
        """
        Perform weighted consensus voting with dynamic reliability
        
        Args:
            votes: Dictionary mapping agent_id to vote/proposal
            confidences: Dictionary mapping agent_id to confidence scores
            hierarchy_level: Level in hierarchy (0=top) for dynamic thresholding
            task_domain: Task domain for specialization weighting
            
        Returns:
            Consensus result with voting analysis
        """
        if not votes:
            return {'consensus': None, 'confidence': 0.0, 'voting_analysis': {}}
        
        # Calculate voting weights for all agents
        agent_weights = {}
        total_weight = 0.0
        
        for agent_id in votes.keys():
            confidence = confidences.get(agent_id, 0.5)
            weight = self.calculate_agent_voting_weight(agent_id, confidence, task_domain)
            agent_weights[agent_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            agent_weights = {k: v/total_weight for k, v in agent_weights.items()}
        
        # Get dynamic threshold for this hierarchy level
        threshold_key = f'level_{hierarchy_level}' if hierarchy_level < 4 else 'default'
        required_threshold = self.hierarchical_thresholds.get(threshold_key, self.hierarchical_thresholds['default'])
        
        # Aggregate votes by content (for identical proposals)
        vote_groups = defaultdict(list)
        for agent_id, vote in votes.items():
            # Convert vote to comparable format
            vote_key = self._normalize_vote_for_comparison(vote)
            vote_groups[vote_key].append((agent_id, vote, agent_weights[agent_id]))
        
        # Find consensus by weighted support
        best_consensus = None
        best_weight = 0.0
        consensus_analysis = {}
        
        for vote_key, supporting_agents in vote_groups.items():
            # Calculate total weighted support
            total_support = sum(weight for _, _, weight in supporting_agents)
            agent_list = [agent_id for agent_id, _, _ in supporting_agents]
            
            consensus_analysis[vote_key] = {
                'supporting_agents': agent_list,
                'weighted_support': total_support,
                'agent_count': len(agent_list),
                'meets_threshold': total_support >= required_threshold,
                'representative_vote': supporting_agents[0][1]  # Use first agent's vote as representative
            }
            
            if total_support > best_weight:
                best_weight = total_support
                best_consensus = supporting_agents[0][1]  # Representative vote
        
        # Calculate overall confidence
        overall_confidence = best_weight if best_consensus is not None else 0.0
        consensus_reached = best_weight >= required_threshold
        
        voting_analysis = {
            'total_agents': len(votes),
            'total_weight': sum(agent_weights.values()),
            'required_threshold': required_threshold,
            'consensus_weight': best_weight,
            'consensus_reached': consensus_reached,
            'hierarchy_level': hierarchy_level,
            'agent_weights': agent_weights,
            'vote_distribution': consensus_analysis,
            'task_domain': task_domain
        }
        
        result = {
            'consensus': best_consensus if consensus_reached else None,
            'confidence': overall_confidence,
            'voting_analysis': voting_analysis,
            'requires_ice_loop': not consensus_reached and best_weight > 0.1  # Trigger ICE if close
        }
        
        logger.info(f"Weighted consensus: {consensus_reached} (weight: {best_weight:.3f}/{required_threshold:.3f})")
        return result
    
    def _calculate_reliability_score(self, agent_id: str) -> float:
        """Calculate historical reliability score (Ri) from performance metrics"""
        profile = self.agent_profiles[agent_id]
        
        if not profile.historical_performance:
            return 0.5  # Default neutral score
        
        # Aggregate performance metrics
        total_precision = 0.0
        total_recall = 0.0
        total_confidence_accuracy = 0.0
        valid_metrics = 0
        
        for metric in profile.historical_performance:
            if metric.ground_truth_available:
                total_precision += metric.precision
                total_recall += metric.recall
                total_confidence_accuracy += metric.confidence_accuracy
                valid_metrics += 1
        
        if valid_metrics == 0:
            return 0.5
        
        # Calculate aggregate reliability
        avg_precision = total_precision / valid_metrics
        avg_recall = total_recall / valid_metrics
        avg_confidence_accuracy = total_confidence_accuracy / valid_metrics
        
        # F1-based reliability with confidence calibration bonus
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
        reliability = 0.7 * f1_score + 0.3 * avg_confidence_accuracy
        
        return max(reliability, self.min_reliability_score)
    
    def _calculate_confidence_calibration(self, agent_id: str) -> float:
        """Calculate confidence calibration score for agent"""
        profile = self.agent_profiles[agent_id]
        
        if not profile.historical_performance:
            return 0.5
        
        # Calculate Expected Calibration Error (ECE)
        calibration_errors = []
        
        for metric in profile.historical_performance:
            if metric.ground_truth_available:
                # Simplified calibration: difference between confidence and actual accuracy
                actual_accuracy = (metric.precision + metric.recall) / 2
                confidence_error = abs(metric.confidence_accuracy - actual_accuracy)
                calibration_errors.append(confidence_error)
        
        if not calibration_errors:
            return 0.5
        
        # Good calibration = low error
        avg_calibration_error = np.mean(calibration_errors)
        calibration_score = max(1.0 - avg_calibration_error, 0.0)
        
        return calibration_score
    
    def _get_decayed_reliability_score(self, agent_id: str) -> float:
        """Get reliability score with temporal decay"""
        profile = self.agent_profiles[agent_id]
        base_reliability = profile.current_reliability_score
        
        # Apply temporal decay
        days_since_update = (datetime.now() - profile.last_updated).days
        if days_since_update > 0:
            decay_factor = np.exp(-days_since_update / self.reliability_decay_days)
            decayed_reliability = base_reliability * decay_factor + 0.5 * (1 - decay_factor)
            return max(decayed_reliability, self.min_reliability_score)
        
        return base_reliability
    
    def _calibrate_confidence_score(self, agent_id: str, raw_confidence: float) -> float:
        """Calibrate confidence score based on agent's historical calibration"""
        profile = self.agent_profiles[agent_id]
        calibration_factor = profile.confidence_calibration_score
        
        # Adjust confidence based on calibration history
        if calibration_factor > 0.7:
            # Well-calibrated agent - trust their confidence
            return raw_confidence
        elif calibration_factor < 0.3:
            # Poorly calibrated agent - moderate their confidence
            return 0.5 * raw_confidence + 0.5 * 0.5  # Pull towards neutral
        else:
            # Moderate calibration
            return 0.8 * raw_confidence + 0.2 * calibration_factor
    
    def _normalize_vote_for_comparison(self, vote: Any) -> str:
        """Normalize vote for comparison purposes"""
        if isinstance(vote, dict):
            # Sort keys for consistent comparison
            return json.dumps(vote, sort_keys=True, default=str)
        elif isinstance(vote, list):
            return json.dumps(sorted(vote, key=str), default=str)
        else:
            return str(vote)
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an agent"""
        if agent_id not in self.agent_profiles:
            return {}
        
        profile = self.agent_profiles[agent_id]
        
        # Calculate performance trends
        recent_performance = [m for m in profile.historical_performance 
                            if (datetime.now() - m.timestamp).days < 7]
        
        stats = {
            'agent_id': agent_id,
            'current_reliability': profile.current_reliability_score,
            'confidence_calibration': profile.confidence_calibration_score,
            'specialization_domains': profile.specialization_domains,
            'total_tasks': profile.task_count,
            'recent_tasks': len(recent_performance),
            'last_updated': profile.last_updated.isoformat(),
            'decayed_reliability': self._get_decayed_reliability_score(agent_id),
            'performance_summary': {}
        }
        
        if profile.historical_performance:
            performance_data = profile.historical_performance
            stats['performance_summary'] = {
                'avg_precision': np.mean([m.precision for m in performance_data]),
                'avg_recall': np.mean([m.recall for m in performance_data]),
                'avg_f1': np.mean([m.f1_score for m in performance_data]),
                'task_types': list(set(m.task_type for m in performance_data))
            }
        
        return stats
    
    def bulk_performance_update(self, performance_data: Dict[str, List[Dict]]) -> None:
        """Bulk update performance data for multiple agents"""
        for agent_id, metrics in performance_data.items():
            for metric_data in metrics:
                metric = AgentPerformanceMetric(**metric_data)
                self.update_agent_performance(agent_id, metric)
        
        logger.info(f"Bulk performance update completed for {len(performance_data)} agents")
    
    def export_reliability_profiles(self) -> Dict[str, Any]:
        """Export all agent reliability profiles for persistence"""
        export_data = {
            'system_config': {
                'alpha': self.alpha,
                'beta': self.beta,
                'reliability_decay_days': self.reliability_decay_days,
                'min_reliability_score': self.min_reliability_score,
                'confidence_weight_threshold': self.confidence_weight_threshold,
                'hierarchical_thresholds': self.hierarchical_thresholds
            },
            'agent_profiles': {},
            'export_timestamp': datetime.now().isoformat()
        }
        
        for agent_id, profile in self.agent_profiles.items():
            export_data['agent_profiles'][agent_id] = {
                'agent_id': profile.agent_id,
                'current_reliability_score': profile.current_reliability_score,
                'confidence_calibration_score': profile.confidence_calibration_score,
                'specialization_domains': profile.specialization_domains,
                'last_updated': profile.last_updated.isoformat(),
                'task_count': profile.task_count,
                'historical_performance': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'task_type': m.task_type,
                        'precision': m.precision,
                        'recall': m.recall,
                        'f1_score': m.f1_score,
                        'confidence_accuracy': m.confidence_accuracy,
                        'ground_truth_available': m.ground_truth_available
                    } for m in profile.historical_performance
                ]
            }
        
        return export_data