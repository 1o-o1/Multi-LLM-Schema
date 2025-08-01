#!/usr/bin/env python3
"""
Uncertainty and Confidence Metrics (Section 9.2)

This module implements uncertainty quantification metrics:
- Confidence calibration and reliability assessment
- Epistemic vs aleatoric uncertainty separation
- Model agreement and disagreement analysis
- Uncertainty propagation through consensus pipeline
"""

import json
import numpy as np
import logging
import statistics
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class UncertaintyMetrics:
    """Container for uncertainty and confidence metrics"""
    # Overall Confidence Assessment
    overall_confidence: float = 0.0
    confidence_reliability: float = 0.0
    calibration_error: float = 0.0
    
    # Uncertainty Types
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Data uncertainty
    total_uncertainty: float = 0.0
    
    # Model Agreement Analysis
    inter_model_agreement: float = 0.0
    model_consensus_strength: float = 0.0
    disagreement_areas: List[str] = field(default_factory=list)
    
    # Confidence Distribution
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0
    confidence_histogram: Dict[str, int] = field(default_factory=dict)
    
    # Calibration Metrics
    expected_calibration_error: float = 0.0
    overconfidence_ratio: float = 0.0
    underconfidence_ratio: float = 0.0
    
    # Uncertainty Propagation
    input_uncertainty: float = 0.0
    processing_uncertainty: float = 0.0
    output_uncertainty: float = 0.0
    uncertainty_amplification: float = 0.0
    
    # Decision Confidence
    high_confidence_decisions: int = 0
    medium_confidence_decisions: int = 0
    low_confidence_decisions: int = 0
    uncertain_decisions: int = 0
    
    # Trust and Reliability
    model_trust_variance: float = 0.0
    reliability_score: float = 0.0
    consistency_score: float = 0.0
    
    def __post_init__(self):
        if not self.confidence_histogram:
            self.confidence_histogram = {}

class UncertaintyAnalyzer:
    """Uncertainty quantification and confidence analysis system"""
    
    def __init__(self):
        """Initialize uncertainty analyzer"""
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
    def evaluate_uncertainty(
        self,
        consensus_output: Dict[str, Any],
        processing_metadata: Dict[str, Any],
        original_inputs: List[Dict[str, Any]]
    ) -> UncertaintyMetrics:
        """
        Comprehensive uncertainty evaluation
        
        Args:
            consensus_output: Output from consensus system
            processing_metadata: Metadata about consensus process
            original_inputs: Original input files
            
        Returns:
            UncertaintyMetrics with all computed metrics
        """
        logger.info("Starting uncertainty evaluation")
        
        metrics = UncertaintyMetrics()
        
        try:
            # Extract confidence data
            confidence_data = self._extract_confidence_data(consensus_output, processing_metadata)
            
            # 1. Overall Confidence Assessment
            metrics.overall_confidence = self._calculate_overall_confidence(confidence_data)
            metrics.confidence_reliability = self._assess_confidence_reliability(confidence_data)
            metrics.calibration_error = self._calculate_calibration_error(confidence_data)
            
            # 2. Uncertainty Type Analysis
            metrics.epistemic_uncertainty = self._calculate_epistemic_uncertainty(confidence_data, original_inputs)
            metrics.aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(confidence_data)
            metrics.total_uncertainty = self._calculate_total_uncertainty(metrics.epistemic_uncertainty, metrics.aleatoric_uncertainty)
            
            # 3. Model Agreement Analysis
            metrics.inter_model_agreement = self._analyze_inter_model_agreement(original_inputs)
            metrics.model_consensus_strength = self._calculate_model_consensus_strength(confidence_data)
            metrics.disagreement_areas = self._identify_disagreement_areas(original_inputs)
            
            # 4. Confidence Distribution Analysis
            confidence_values = self._extract_confidence_values(confidence_data)
            if confidence_values:
                metrics.confidence_mean = statistics.mean(confidence_values)
                metrics.confidence_std = statistics.stdev(confidence_values) if len(confidence_values) > 1 else 0.0
                metrics.confidence_min = min(confidence_values)
                metrics.confidence_max = max(confidence_values)
                metrics.confidence_histogram = self._create_confidence_histogram(confidence_values)
            
            # 5. Calibration Metrics
            metrics.expected_calibration_error = self._calculate_expected_calibration_error(confidence_data)
            metrics.overconfidence_ratio = self._calculate_overconfidence_ratio(confidence_data)
            metrics.underconfidence_ratio = self._calculate_underconfidence_ratio(confidence_data)
            
            # 6. Uncertainty Propagation
            metrics.input_uncertainty = self._measure_input_uncertainty(original_inputs)
            metrics.processing_uncertainty = self._measure_processing_uncertainty(processing_metadata)
            metrics.output_uncertainty = self._measure_output_uncertainty(consensus_output)
            metrics.uncertainty_amplification = self._calculate_uncertainty_amplification(metrics)
            
            # 7. Decision Confidence Categories
            decision_counts = self._categorize_decisions_by_confidence(confidence_values)
            metrics.high_confidence_decisions = decision_counts['high']
            metrics.medium_confidence_decisions = decision_counts['medium']
            metrics.low_confidence_decisions = decision_counts['low']
            metrics.uncertain_decisions = decision_counts['uncertain']
            
            # 8. Trust and Reliability
            metrics.model_trust_variance = self._calculate_model_trust_variance(original_inputs)
            metrics.reliability_score = self._calculate_reliability_score(metrics)
            metrics.consistency_score = self._calculate_consistency_score(confidence_data)
            
            logger.info(f"Uncertainty evaluation completed. Overall confidence: {metrics.overall_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Uncertainty evaluation failed: {e}")
        
        return metrics
    
    def _extract_confidence_data(
        self, 
        consensus_output: Dict[str, Any], 
        processing_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract confidence-related data from consensus output"""
        confidence_data = {
            'quality_metrics': consensus_output.get('quality_metrics', {}),
            'processing_metadata': processing_metadata,
            'consensus_confidence': [],
            'individual_confidences': []
        }
        
        # Extract consensus confidence
        qm = confidence_data['quality_metrics']
        avg_confidence = qm.get('average_consensus_confidence', 0.0)
        if avg_confidence:
            confidence_data['consensus_confidence'].append(float(avg_confidence))
        
        # Look for research analysis data
        if hasattr(consensus_output, 'research_analysis'):
            research_analysis = consensus_output.research_analysis
        elif 'research_analysis' in consensus_output:
            research_analysis = consensus_output['research_analysis']
        else:
            research_analysis = {}
        
        confidence_data['research_analysis'] = research_analysis
        
        # Extract MUSE uncertainty data if available
        if 'muse_uncertainty_analysis' in research_analysis:
            muse_data = research_analysis['muse_uncertainty_analysis']
            confidence_data['muse_confidence'] = muse_data.get('overall_confidence', 0.0)
            confidence_data['calibration_score'] = muse_data.get('calibration_score', 0.0)
        
        # Extract weighted voting confidence if available
        if 'weighted_voting_analysis' in research_analysis:
            voting_data = research_analysis['weighted_voting_analysis']
            confidence_data['voting_confidence'] = voting_data.get('reliability_weighted_confidence', 0.0)
        
        return confidence_data
    
    def _calculate_overall_confidence(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate overall system confidence"""
        try:
            confidence_sources = []
            
            # Consensus confidence
            qm = confidence_data.get('quality_metrics', {})
            consensus_conf = float(qm.get('average_consensus_confidence', 0.0))
            if consensus_conf > 0:
                confidence_sources.append(consensus_conf)
            
            # MUSE confidence
            muse_conf = confidence_data.get('muse_confidence', 0.0)
            if muse_conf > 0:
                confidence_sources.append(float(muse_conf))
            
            # Voting confidence
            voting_conf = confidence_data.get('voting_confidence', 0.0)
            if voting_conf > 0:
                confidence_sources.append(float(voting_conf))
            
            # Calibration-based confidence
            calibration = confidence_data.get('calibration_score', 0.0)
            if calibration > 0:
                confidence_sources.append(float(calibration))
            
            return statistics.mean(confidence_sources) if confidence_sources else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.0
    
    def _assess_confidence_reliability(self, confidence_data: Dict[str, Any]) -> float:
        """Assess the reliability of confidence estimates"""
        try:
            reliability_factors = []
            
            # Factor 1: Consistency across different confidence sources
            confidence_values = []
            qm = confidence_data.get('quality_metrics', {})
            
            consensus_conf = float(qm.get('average_consensus_confidence', 0.0))
            muse_conf = confidence_data.get('muse_confidence', 0.0)
            voting_conf = confidence_data.get('voting_confidence', 0.0)
            
            for conf in [consensus_conf, muse_conf, voting_conf]:
                if conf > 0:
                    confidence_values.append(conf)
            
            if len(confidence_values) > 1:
                # Reliability is higher when confidence estimates agree
                conf_std = statistics.stdev(confidence_values)
                consistency = 1.0 - min(1.0, conf_std)  # Lower std = higher consistency
                reliability_factors.append(consistency)
            
            # Factor 2: Calibration quality
            calibration = confidence_data.get('calibration_score', 0.0)
            if calibration > 0:
                reliability_factors.append(float(calibration))
            
            # Factor 3: Success rate as proxy for confidence reliability
            success_rate = qm.get('consensus_success_rate', 0.0)
            reliability_factors.append(success_rate)
            
            return statistics.mean(reliability_factors) if reliability_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing confidence reliability: {e}")
            return 0.0
    
    def _calculate_calibration_error(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate confidence calibration error"""
        try:
            # Use MUSE calibration score if available
            calibration_score = confidence_data.get('calibration_score', 0.0)
            if calibration_score > 0:
                # Convert calibration score to error (1 - score)
                return 1.0 - float(calibration_score)
            
            # Fallback: estimate calibration error from consistency
            qm = confidence_data.get('quality_metrics', {})
            confidence = float(qm.get('average_consensus_confidence', 0.5))
            success_rate = qm.get('consensus_success_rate', 0.5)
            
            # Calibration error is the difference between confidence and actual performance
            calibration_error = abs(confidence - success_rate)
            return calibration_error
            
        except Exception as e:
            logger.error(f"Error calculating calibration error: {e}")
            return 0.5  # Default moderate error
    
    def _calculate_epistemic_uncertainty(
        self, 
        confidence_data: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> float:
        """Calculate epistemic (model) uncertainty"""
        try:
            # Epistemic uncertainty comes from model disagreement
            if len(original_inputs) < 2:
                return 0.0  # No disagreement possible with single model
            
            # Measure disagreement between models
            model_outputs = []
            for input_data in original_inputs:
                # Extract comparable outputs
                output_repr = self._extract_model_output_representation(input_data)
                model_outputs.append(output_repr)
            
            # Calculate pairwise disagreement
            disagreements = []
            for i in range(len(model_outputs)):
                for j in range(i + 1, len(model_outputs)):
                    disagreement = self._calculate_output_disagreement(model_outputs[i], model_outputs[j])
                    disagreements.append(disagreement)
            
            epistemic_uncertainty = statistics.mean(disagreements) if disagreements else 0.0
            return min(1.0, epistemic_uncertainty)
            
        except Exception as e:
            logger.error(f"Error calculating epistemic uncertainty: {e}")
            return 0.0
    
    def _extract_model_output_representation(self, input_data: Dict[str, Any]) -> set:
        """Extract a representation of model output for comparison"""
        representation = set()
        
        # Extract from parsed_json if available
        parsed_json = input_data.get('parsed_json', input_data)
        
        if isinstance(parsed_json, dict):
            for key, value in parsed_json.items():
                if isinstance(value, (str, int, float)):
                    representation.add(f"{key}:{str(value)[:100]}")  # Truncate long values
                elif isinstance(value, dict):
                    for nested_key in value.keys():
                        representation.add(f"{key}.{nested_key}")
                elif isinstance(value, list):
                    representation.add(f"{key}[{len(value)}]")
        
        return representation
    
    def _calculate_output_disagreement(self, output1: set, output2: set) -> float:
        """Calculate disagreement between two model outputs"""
        if not output1 and not output2:
            return 0.0
        
        intersection = len(output1.intersection(output2))
        union = len(output1.union(output2))
        
        # Disagreement is 1 - Jaccard similarity
        jaccard_similarity = intersection / union if union > 0 else 0.0
        disagreement = 1.0 - jaccard_similarity
        
        return disagreement
    
    def _calculate_aleatoric_uncertainty(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate aleatoric (data) uncertainty"""
        try:
            # Aleatoric uncertainty comes from inherent data noise/ambiguity
            qm = confidence_data.get('quality_metrics', {})
            
            # Use data preservation as inverse indicator of aleatoric uncertainty
            # Lower preservation suggests more ambiguous/noisy data
            data_preservation = qm.get('data_preservation', 0.5)
            
            # Use source diversity as indicator of data uncertainty
            # Higher diversity can indicate more conflicting/uncertain data
            source_diversity = qm.get('source_diversity', 0.5)
            
            # Combine factors
            aleatoric_uncertainty = (1.0 - data_preservation) * 0.6 + source_diversity * 0.4
            
            return min(1.0, max(0.0, aleatoric_uncertainty))
            
        except Exception as e:
            logger.error(f"Error calculating aleatoric uncertainty: {e}")
            return 0.0
    
    def _calculate_total_uncertainty(self, epistemic: float, aleatoric: float) -> float:
        """Calculate total uncertainty from epistemic and aleatoric components"""
        # Total uncertainty is not simply additive - they interact
        # Use quadrature sum as approximation
        total = math.sqrt(epistemic**2 + aleatoric**2)
        return min(1.0, total)
    
    def _analyze_inter_model_agreement(self, original_inputs: List[Dict[str, Any]]) -> float:
        """Analyze agreement between different models"""
        if len(original_inputs) < 2:
            return 1.0  # Perfect agreement with single model
        
        try:
            # Extract model representations
            model_representations = []
            for input_data in original_inputs:
                repr_set = self._extract_model_output_representation(input_data)
                model_representations.append(repr_set)
            
            # Calculate pairwise agreements
            agreements = []
            for i in range(len(model_representations)):
                for j in range(i + 1, len(model_representations)):
                    agreement = self._calculate_pairwise_agreement(
                        model_representations[i], model_representations[j]
                    )
                    agreements.append(agreement)
            
            return statistics.mean(agreements) if agreements else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing inter-model agreement: {e}")
            return 0.0
    
    def _calculate_pairwise_agreement(self, set1: set, set2: set) -> float:
        """Calculate agreement between two sets (Jaccard similarity)"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_model_consensus_strength(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate the strength of model consensus"""
        try:
            qm = confidence_data.get('quality_metrics', {})
            
            # Use consensus success rate as primary indicator
            consensus_strength = qm.get('consensus_success_rate', 0.0)
            
            # Adjust based on confidence level
            confidence = float(qm.get('average_consensus_confidence', 0.0))
            
            # Strong consensus requires both high success rate and high confidence
            adjusted_strength = (consensus_strength + confidence) / 2.0
            
            return adjusted_strength
            
        except Exception as e:
            logger.error(f"Error calculating model consensus strength: {e}")
            return 0.0
    
    def _identify_disagreement_areas(self, original_inputs: List[Dict[str, Any]]) -> List[str]:
        """Identify areas where models disagree significantly"""
        disagreement_areas = []
        
        if len(original_inputs) < 2:
            return disagreement_areas
        
        try:
            # Extract structured data for comparison
            model_structures = []
            for input_data in original_inputs:
                structure = self._extract_structural_elements(input_data)
                model_structures.append(structure)
            
            # Find elements that appear in some but not all models
            all_elements = set()
            for structure in model_structures:
                all_elements.update(structure.keys())
            
            for element in all_elements:
                appearances = sum(1 for structure in model_structures if element in structure)
                disagreement_ratio = 1.0 - (appearances / len(model_structures))
                
                if disagreement_ratio > 0.3:  # Significant disagreement
                    disagreement_areas.append(f"{element} (disagreement: {disagreement_ratio:.2f})")
            
        except Exception as e:
            logger.error(f"Error identifying disagreement areas: {e}")
        
        return disagreement_areas[:10]  # Limit to top 10
    
    def _extract_structural_elements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural elements for disagreement analysis"""
        elements = {}
        
        parsed_json = input_data.get('parsed_json', input_data)
        
        if isinstance(parsed_json, dict):
            for key, value in parsed_json.items():
                if isinstance(value, dict):
                    elements[key] = f"dict[{len(value)}]"
                elif isinstance(value, list):
                    elements[key] = f"list[{len(value)}]"
                else:
                    elements[key] = type(value).__name__
        
        return elements
    
    def _extract_confidence_values(self, confidence_data: Dict[str, Any]) -> List[float]:
        """Extract all confidence values for distribution analysis"""
        confidence_values = []
        
        # Add various confidence sources
        qm = confidence_data.get('quality_metrics', {})
        
        # Consensus confidence
        consensus_conf = qm.get('average_consensus_confidence', 0.0)
        if consensus_conf > 0:
            confidence_values.append(float(consensus_conf))
        
        # Other confidence sources
        for key in ['muse_confidence', 'voting_confidence']:
            conf = confidence_data.get(key, 0.0)
            if conf > 0:
                confidence_values.append(float(conf))
        
        # Success rate as confidence proxy
        success_rate = qm.get('consensus_success_rate', 0.0)
        if success_rate > 0:
            confidence_values.append(success_rate)
        
        return confidence_values
    
    def _create_confidence_histogram(self, confidence_values: List[float]) -> Dict[str, int]:
        """Create histogram of confidence values"""
        histogram = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }
        
        for value in confidence_values:
            if value <= 0.2:
                histogram['0.0-0.2'] += 1
            elif value <= 0.4:
                histogram['0.2-0.4'] += 1
            elif value <= 0.6:
                histogram['0.4-0.6'] += 1
            elif value <= 0.8:
                histogram['0.6-0.8'] += 1
            else:
                histogram['0.8-1.0'] += 1
        
        return histogram
    
    def _calculate_expected_calibration_error(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        try:
            # Use available calibration data
            calibration_score = confidence_data.get('calibration_score', 0.0)
            if calibration_score > 0:
                return 1.0 - float(calibration_score)
            
            # Fallback ECE calculation
            qm = confidence_data.get('quality_metrics', {})
            confidence = float(qm.get('average_consensus_confidence', 0.5))
            accuracy = qm.get('consensus_success_rate', 0.5)
            
            # Simple ECE approximation
            ece = abs(confidence - accuracy)
            return ece
            
        except Exception as e:
            logger.error(f"Error calculating Expected Calibration Error: {e}")
            return 0.0
    
    def _calculate_overconfidence_ratio(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate ratio of overconfident predictions"""
        try:
            qm = confidence_data.get('quality_metrics', {})
            confidence = float(qm.get('average_consensus_confidence', 0.5))
            accuracy = qm.get('consensus_success_rate', 0.5)
            
            # Overconfidence occurs when confidence > accuracy
            overconfidence = max(0.0, confidence - accuracy)
            return overconfidence
            
        except Exception as e:
            logger.error(f"Error calculating overconfidence ratio: {e}")
            return 0.0
    
    def _calculate_underconfidence_ratio(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate ratio of underconfident predictions"""
        try:
            qm = confidence_data.get('quality_metrics', {})
            confidence = float(qm.get('average_consensus_confidence', 0.5))
            accuracy = qm.get('consensus_success_rate', 0.5)
            
            # Underconfidence occurs when accuracy > confidence
            underconfidence = max(0.0, accuracy - confidence)
            return underconfidence
            
        except Exception as e:
            logger.error(f"Error calculating underconfidence ratio: {e}")
            return 0.0
    
    def _measure_input_uncertainty(self, original_inputs: List[Dict[str, Any]]) -> float:
        """Measure uncertainty in input data"""
        if len(original_inputs) < 2:
            return 0.0
        
        try:
            # Measure variation in input structures
            input_structures = []
            for input_data in original_inputs:
                structure = self._extract_structural_elements(input_data)
                input_structures.append(set(structure.keys()))
            
            # Calculate structural disagreement
            all_keys = set()
            for structure in input_structures:
                all_keys.update(structure)
            
            disagreements = []
            for key in all_keys:
                appearances = sum(1 for structure in input_structures if key in structure)
                disagreement = 1.0 - (appearances / len(input_structures))
                disagreements.append(disagreement)
            
            input_uncertainty = statistics.mean(disagreements) if disagreements else 0.0
            return input_uncertainty
            
        except Exception as e:
            logger.error(f"Error measuring input uncertainty: {e}")
            return 0.0
    
    def _measure_processing_uncertainty(self, processing_metadata: Dict[str, Any]) -> float:
        """Measure uncertainty introduced during processing"""
        try:
            # Processing uncertainty from time variability and complexity
            processing_time = processing_metadata.get('total_processing_time', 0.0)
            
            # Higher processing time suggests more uncertainty/complexity
            # Normalize by expected processing time
            expected_time = 5.0  # 5 seconds as baseline
            time_uncertainty = min(1.0, processing_time / (expected_time * 3))  # Cap at 3x expected
            
            return time_uncertainty
            
        except Exception as e:
            logger.error(f"Error measuring processing uncertainty: {e}")
            return 0.0
    
    def _measure_output_uncertainty(self, consensus_output: Dict[str, Any]) -> float:
        """Measure uncertainty in output"""
        try:
            qm = consensus_output.get('quality_metrics', {})
            
            # Output uncertainty from low confidence and coverage
            confidence = float(qm.get('average_consensus_confidence', 0.5))
            coverage = qm.get('coverage', 0.5)
            
            # Lower confidence and coverage indicate higher output uncertainty
            output_uncertainty = 1.0 - ((confidence + coverage) / 2.0)
            
            return output_uncertainty
            
        except Exception as e:
            logger.error(f"Error measuring output uncertainty: {e}")
            return 0.0
    
    def _calculate_uncertainty_amplification(self, metrics: UncertaintyMetrics) -> float:
        """Calculate how much uncertainty is amplified through the pipeline"""
        try:
            if metrics.input_uncertainty <= 0:
                return 0.0
            
            # Amplification is output uncertainty / input uncertainty
            amplification = metrics.output_uncertainty / metrics.input_uncertainty
            
            # Values > 1 indicate amplification, < 1 indicate reduction
            return amplification
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty amplification: {e}")
            return 1.0  # No amplification by default
    
    def _categorize_decisions_by_confidence(self, confidence_values: List[float]) -> Dict[str, int]:
        """Categorize decisions by confidence level"""
        categories = {'high': 0, 'medium': 0, 'low': 0, 'uncertain': 0}
        
        for confidence in confidence_values:
            if confidence >= self.confidence_thresholds['high']:
                categories['high'] += 1
            elif confidence >= self.confidence_thresholds['medium']:
                categories['medium'] += 1
            elif confidence >= self.confidence_thresholds['low']:
                categories['low'] += 1
            else:
                categories['uncertain'] += 1
        
        return categories
    
    def _calculate_model_trust_variance(self, original_inputs: List[Dict[str, Any]]) -> float:
        """Calculate variance in model trustworthiness"""
        if len(original_inputs) < 2:
            return 0.0
        
        try:
            # Estimate trust scores based on output quality/consistency
            trust_scores = []
            for input_data in original_inputs:
                # Simple heuristic: trust based on data completeness
                parsed_json = input_data.get('parsed_json', input_data)
                completeness = len(str(parsed_json)) / 1000.0  # Rough completeness metric
                trust_score = min(1.0, completeness)
                trust_scores.append(trust_score)
            
            # Calculate variance
            if len(trust_scores) > 1:
                return statistics.variance(trust_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating model trust variance: {e}")
            return 0.0
    
    def _calculate_reliability_score(self, metrics: UncertaintyMetrics) -> float:
        """Calculate overall reliability score"""
        try:
            reliability_factors = []
            
            # High confidence indicates reliability
            reliability_factors.append(metrics.overall_confidence)
            
            # Low calibration error indicates reliability
            calibration_reliability = 1.0 - metrics.calibration_error
            reliability_factors.append(calibration_reliability)
            
            # High model agreement indicates reliability
            reliability_factors.append(metrics.inter_model_agreement)
            
            # Low total uncertainty indicates reliability
            uncertainty_reliability = 1.0 - metrics.total_uncertainty
            reliability_factors.append(uncertainty_reliability)
            
            return statistics.mean(reliability_factors) if reliability_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {e}")
            return 0.0
    
    def _calculate_consistency_score(self, confidence_data: Dict[str, Any]) -> float:
        """Calculate consistency score across different confidence measures"""
        try:
            confidence_values = []
            
            qm = confidence_data.get('quality_metrics', {})
            consensus_conf = float(qm.get('average_consensus_confidence', 0.0))
            muse_conf = confidence_data.get('muse_confidence', 0.0)
            voting_conf = confidence_data.get('voting_confidence', 0.0)
            
            for conf in [consensus_conf, muse_conf, voting_conf]:
                if conf > 0:
                    confidence_values.append(conf)
            
            if len(confidence_values) < 2:
                return 1.0  # Perfect consistency with single value
            
            # Consistency is inverse of standard deviation
            conf_std = statistics.stdev(confidence_values)
            consistency = 1.0 - min(1.0, conf_std)  # Lower std = higher consistency
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0

def create_uncertainty_report(metrics: UncertaintyMetrics, output_file: Optional[str] = None) -> str:
    """Create a comprehensive uncertainty analysis report"""
    
    report = f"""
# Uncertainty and Confidence Analysis Report

## Overall Confidence Assessment

### System Confidence
- **Overall Confidence**: {metrics.overall_confidence:.3f}
- **Confidence Reliability**: {metrics.confidence_reliability:.3f}
- **Calibration Error**: {metrics.calibration_error:.3f}

### Confidence Distribution
- **Mean**: {metrics.confidence_mean:.3f}
- **Standard Deviation**: {metrics.confidence_std:.3f}
- **Range**: {metrics.confidence_min:.3f} - {metrics.confidence_max:.3f}

## Uncertainty Analysis

### Uncertainty Types
- **Epistemic Uncertainty** (Model): {metrics.epistemic_uncertainty:.3f}
- **Aleatoric Uncertainty** (Data): {metrics.aleatoric_uncertainty:.3f}
- **Total Uncertainty**: {metrics.total_uncertainty:.3f}

### Uncertainty Propagation
- **Input Uncertainty**: {metrics.input_uncertainty:.3f}
- **Processing Uncertainty**: {metrics.processing_uncertainty:.3f}
- **Output Uncertainty**: {metrics.output_uncertainty:.3f}
- **Uncertainty Amplification**: {metrics.uncertainty_amplification:.3f}

## Model Agreement Analysis

### Inter-Model Consensus
- **Inter-Model Agreement**: {metrics.inter_model_agreement:.3f}
- **Model Consensus Strength**: {metrics.model_consensus_strength:.3f}
- **Model Trust Variance**: {metrics.model_trust_variance:.3f}

### Disagreement Areas
{_format_disagreement_areas(metrics.disagreement_areas)}

## Calibration Metrics

### Calibration Assessment
- **Expected Calibration Error**: {metrics.expected_calibration_error:.3f}
- **Overconfidence Ratio**: {metrics.overconfidence_ratio:.3f}
- **Underconfidence Ratio**: {metrics.underconfidence_ratio:.3f}

## Decision Confidence Distribution

### Confidence Categories
- **High Confidence Decisions**: {metrics.high_confidence_decisions}
- **Medium Confidence Decisions**: {metrics.medium_confidence_decisions}
- **Low Confidence Decisions**: {metrics.low_confidence_decisions}
- **Uncertain Decisions**: {metrics.uncertain_decisions}

### Confidence Histogram
{_format_confidence_histogram(metrics.confidence_histogram)}

## Trust and Reliability

### Reliability Metrics
- **Overall Reliability Score**: {metrics.reliability_score:.3f}
- **Consistency Score**: {metrics.consistency_score:.3f}

## Uncertainty Assessment

{_generate_uncertainty_assessment(metrics)}

## Recommendations

{_generate_uncertainty_recommendations(metrics)}
"""
    
    if output_file:
        # Remove Unicode characters for Windows compatibility
        safe_report = report.replace('✓', 'SUCCESS').replace('⚠', 'WARNING').replace('✗', 'FAILED')
        Path(output_file).write_text(safe_report, encoding='utf-8')
        logger.info(f"Uncertainty report saved to {output_file}")
    
    return report

def _format_disagreement_areas(disagreement_areas: List[str]) -> str:
    """Format disagreement areas for display"""
    if not disagreement_areas:
        return "No significant disagreement areas identified"
    
    lines = []
    for area in disagreement_areas[:5]:  # Show top 5
        lines.append(f"- {area}")
    
    return "\n".join(lines)

def _format_confidence_histogram(confidence_histogram: Dict[str, int]) -> str:
    """Format confidence histogram for display"""
    if not confidence_histogram:
        return "No confidence distribution data available"
    
    lines = []
    for range_label, count in confidence_histogram.items():
        lines.append(f"- {range_label}: {count}")
    
    return "\n".join(lines)

def _generate_uncertainty_assessment(metrics: UncertaintyMetrics) -> str:
    """Generate overall uncertainty assessment"""
    assessment_lines = []
    
    # Overall confidence assessment
    if metrics.overall_confidence >= 0.8:
        assessment_lines.append("✓ HIGH overall system confidence")
    elif metrics.overall_confidence >= 0.6:
        assessment_lines.append("✓ MODERATE system confidence")
    else:
        assessment_lines.append("⚠ LOW system confidence")
    
    # Uncertainty level assessment
    if metrics.total_uncertainty <= 0.2:
        assessment_lines.append("✓ LOW total uncertainty")
    elif metrics.total_uncertainty <= 0.4:
        assessment_lines.append("✓ MODERATE uncertainty levels")
    else:
        assessment_lines.append("⚠ HIGH uncertainty levels")
    
    # Calibration assessment
    if metrics.calibration_error <= 0.1:
        assessment_lines.append("✓ EXCELLENT confidence calibration")
    elif metrics.calibration_error <= 0.2:
        assessment_lines.append("✓ GOOD confidence calibration")
    else:
        assessment_lines.append("⚠ POOR confidence calibration")
    
    # Model agreement assessment
    if metrics.inter_model_agreement >= 0.8:
        assessment_lines.append("✓ STRONG inter-model agreement")
    elif metrics.inter_model_agreement >= 0.6:
        assessment_lines.append("✓ MODERATE inter-model agreement")
    else:
        assessment_lines.append("⚠ WEAK inter-model agreement")
    
    return "\n".join(assessment_lines)

def _generate_uncertainty_recommendations(metrics: UncertaintyMetrics) -> str:
    """Generate recommendations based on uncertainty analysis"""
    recommendations = []
    
    if metrics.overall_confidence < 0.6:
        recommendations.append("- Investigate sources of low confidence and improve model reliability")
    
    if metrics.calibration_error > 0.2:
        recommendations.append("- Improve confidence calibration through better uncertainty estimation")
    
    if metrics.total_uncertainty > 0.4:
        recommendations.append("- Address high uncertainty through data quality improvements or model refinement")
    
    if metrics.epistemic_uncertainty > 0.3:
        recommendations.append("- Reduce model uncertainty through better training or ensemble methods")
    
    if metrics.aleatoric_uncertainty > 0.3:
        recommendations.append("- Address data uncertainty through data cleaning or acquisition")
    
    if metrics.inter_model_agreement < 0.6:
        recommendations.append("- Investigate model disagreements and resolve conflicting predictions")
    
    if metrics.overconfidence_ratio > 0.2:
        recommendations.append("- Address overconfidence through better calibration techniques")
    
    if metrics.uncertainty_amplification > 1.5:
        recommendations.append("- Optimize processing pipeline to reduce uncertainty amplification")
    
    if not recommendations:
        recommendations.append("- Continue monitoring uncertainty metrics")
        recommendations.append("- Consider testing with more challenging datasets")
    
    return "\n".join(recommendations)