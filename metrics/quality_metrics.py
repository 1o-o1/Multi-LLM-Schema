#!/usr/bin/env python3
"""
Comprehensive Quality Assessment Metrics (Section 9.2)

This module implements the quality metrics defined in Multi-agent.md Section 9.2:
- Node Inclusion Accuracy (Precision, Recall, F1-Score)
- Structural and Semantic Consistency 
- Confidence Calibration (Expected Calibration Error)
- Multi-dimensional quality scoring and reporting

Compatible with current JSON output format from consensus system.
"""

import json
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict
import re
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Container for quality assessment results"""
    # Node Inclusion Accuracy
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Structural Consistency
    structural_validity: bool = False
    structural_error_count: int = 0
    
    # Semantic Consistency  
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    semantic_coherence: float = 0.0
    
    # Confidence Calibration
    expected_calibration_error: float = 0.0
    reliability_score: float = 0.0
    
    # Multi-dimensional Quality
    overall_quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    accuracy_score: float = 0.0
    
    # Additional metrics
    coverage_ratio: float = 0.0
    information_density: float = 0.0
    consensus_stability: float = 0.0
    
    # Metadata
    evaluation_timestamp: str = ""
    input_file_count: int = 0
    output_node_count: int = 0

class QualityAssessment:
    """Comprehensive quality assessment system for consensus outputs"""
    
    def __init__(self, use_llm_judge: bool = False, gemini_provider_path: Optional[str] = None):
        """
        Initialize quality assessment system
        
        Args:
            use_llm_judge: Whether to use LLM for subjective quality assessments
            gemini_provider_path: Path to Gemini provider for LLM judging
        """
        self.use_llm_judge = use_llm_judge
        self.gemini_provider = None
        
        if use_llm_judge and gemini_provider_path:
            try:
                import sys
                sys.path.append(str(Path(gemini_provider_path).parent))
                from gemini_provider import GeminiProvider
                self.gemini_provider = GeminiProvider()
                logger.info("Initialized Gemini provider for LLM judging")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini provider: {e}")
                self.use_llm_judge = False
    
    def evaluate_consensus_output(
        self, 
        consensus_output: Dict[str, Any],
        original_inputs: List[Dict[str, Any]],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """
        Comprehensive evaluation of consensus output
        
        Args:
            consensus_output: Output from consensus system
            original_inputs: Original input files used for consensus
            ground_truth: Optional ground truth for accuracy measurement
            
        Returns:
            QualityMetrics object with all computed metrics
        """
        logger.info("Starting comprehensive quality assessment")
        
        metrics = QualityMetrics()
        metrics.evaluation_timestamp = pd.Timestamp.now().isoformat()
        metrics.input_file_count = len(original_inputs)
        
        try:
            # Extract unified JSON from consensus output
            unified_json = self._extract_unified_json(consensus_output)
            if not unified_json:
                logger.error("No unified JSON found in consensus output")
                return metrics
            
            metrics.output_node_count = self._count_nodes(unified_json)
            
            # 1. Node Inclusion Accuracy (Section 9.2)
            if ground_truth:
                precision, recall, f1 = self._calculate_node_inclusion_accuracy(
                    unified_json, ground_truth
                )
                metrics.precision = precision
                metrics.recall = recall 
                metrics.f1_score = f1
                metrics.accuracy_score = f1  # Use F1 as primary accuracy metric
            else:
                # Without ground truth, use inter-rater agreement from original inputs
                metrics.precision, metrics.recall, metrics.f1_score = self._calculate_consensus_agreement(
                    unified_json, original_inputs
                )
                metrics.accuracy_score = metrics.f1_score
            
            # 2. Structural Validity (Section 9.2)
            metrics.structural_validity, metrics.structural_error_count = self._check_structural_validity(
                unified_json
            )
            
            # 3. Semantic Consistency (Section 9.2) 
            if ground_truth:
                metrics.rouge_1, metrics.rouge_2, metrics.rouge_l = self._calculate_rouge_scores(
                    unified_json, ground_truth
                )
            else:
                # Compare against original inputs for semantic consistency
                metrics.semantic_coherence = self._calculate_semantic_coherence(
                    unified_json, original_inputs
                )
            
            # 4. Confidence Calibration (Section 9.2)
            metrics.expected_calibration_error = self._calculate_expected_calibration_error(
                consensus_output
            )
            metrics.reliability_score = self._calculate_reliability_score(consensus_output)
            
            # 5. Multi-dimensional Quality Scoring
            metrics.completeness_score = self._calculate_completeness_score(
                unified_json, original_inputs
            )
            metrics.consistency_score = self._calculate_consistency_score(unified_json)
            metrics.coverage_ratio = self._calculate_coverage_ratio(
                unified_json, original_inputs
            )
            metrics.information_density = self._calculate_information_density(unified_json)
            metrics.consensus_stability = self._calculate_consensus_stability(consensus_output)
            
            # 6. Overall Quality Score (weighted combination)
            metrics.overall_quality_score = self._calculate_overall_quality(metrics)
            
            # 7. LLM Judge Assessment (if enabled)
            if self.use_llm_judge and self.gemini_provider:
                llm_assessment = self._llm_judge_assessment(
                    unified_json, original_inputs
                )
                # Incorporate LLM assessment into metrics
                metrics.semantic_coherence = max(metrics.semantic_coherence, llm_assessment.get('coherence', 0))
                
            logger.info(f"Quality assessment completed. Overall score: {metrics.overall_quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
        
        return metrics
    
    def _extract_unified_json(self, consensus_output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract unified JSON from consensus output"""
        if 'unified_json' in consensus_output:
            return consensus_output['unified_json']
        elif 'enhanced_security_consensus' in consensus_output:
            return consensus_output['enhanced_security_consensus']
        elif 'consensus_result' in consensus_output:
            return consensus_output['consensus_result']
        else:
            # Try to find the main content
            for key, value in consensus_output.items():
                if isinstance(value, dict) and len(value) > 3:
                    return value
            return None
    
    def _count_nodes(self, json_obj: Any) -> int:
        """Count total nodes in JSON structure"""
        if isinstance(json_obj, dict):
            return 1 + sum(self._count_nodes(v) for v in json_obj.values())
        elif isinstance(json_obj, list):
            return sum(self._count_nodes(item) for item in json_obj)
        else:
            return 1
    
    def _calculate_node_inclusion_accuracy(
        self, 
        predicted: Dict[str, Any], 
        ground_truth: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 for node inclusion accuracy"""
        try:
            # Extract all paths and values from both structures
            pred_nodes = self._extract_all_paths(predicted)
            gt_nodes = self._extract_all_paths(ground_truth)
            
            # Convert to sets for comparison
            pred_set = set(pred_nodes.keys())
            gt_set = set(gt_nodes.keys())
            
            # Calculate metrics
            true_positives = len(pred_set.intersection(gt_set))
            false_positives = len(pred_set - gt_set)
            false_negatives = len(gt_set - pred_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
            
        except Exception as e:
            logger.error(f"Error calculating node inclusion accuracy: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_consensus_agreement(
        self, 
        unified_json: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Calculate consensus agreement as proxy for accuracy when no ground truth"""
        try:
            # Extract paths from unified output
            unified_paths = set(self._extract_all_paths(unified_json).keys())
            
            # Extract paths from each original input
            input_paths = []
            for input_data in original_inputs:
                if 'parsed_json' in input_data:
                    paths = set(self._extract_all_paths(input_data['parsed_json']).keys())
                    input_paths.append(paths)
                else:
                    paths = set(self._extract_all_paths(input_data).keys())
                    input_paths.append(paths)
            
            if not input_paths:
                return 0.0, 0.0, 0.0
            
            # Calculate how much of unified output was supported by inputs
            total_input_paths = set()
            for paths in input_paths:
                total_input_paths.update(paths)
            
            supported_paths = unified_paths.intersection(total_input_paths)
            
            # Precision: How much of output was in original inputs
            precision = len(supported_paths) / len(unified_paths) if unified_paths else 0
            
            # Recall: How much of original inputs was preserved
            recall = len(supported_paths) / len(total_input_paths) if total_input_paths else 0
            
            # F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
            
        except Exception as e:
            logger.error(f"Error calculating consensus agreement: {e}")
            return 0.0, 0.0, 0.0
    
    def _extract_all_paths(self, json_obj: Any, prefix: str = "") -> Dict[str, Any]:
        """Extract all JSON paths and their values"""
        paths = {}
        
        if isinstance(json_obj, dict):
            for key, value in json_obj.items():
                current_path = f"{prefix}.{key}" if prefix else key
                paths[current_path] = value
                
                # Recurse into nested structures
                nested_paths = self._extract_all_paths(value, current_path)
                paths.update(nested_paths)
                
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                current_path = f"{prefix}[{i}]"
                paths[current_path] = item
                
                # Recurse into list items
                nested_paths = self._extract_all_paths(item, current_path)
                paths.update(nested_paths)
        
        return paths
    
    def _check_structural_validity(self, json_obj: Dict[str, Any]) -> Tuple[bool, int]:
        """Check for structural errors like orphaned nodes or cycles"""
        error_count = 0
        
        try:
            # Check for basic JSON validity (already passed if we got here)
            
            # Check for orphaned references
            all_paths = self._extract_all_paths(json_obj)
            
            # Look for reference-like patterns
            reference_pattern = re.compile(r'#/|@ref|ref:|reference')
            for path, value in all_paths.items():
                if isinstance(value, str) and reference_pattern.search(value):
                    # Check if reference target exists
                    # This is a simplified check - could be more sophisticated
                    target = value.replace('#/', '').replace('@ref', '').strip()
                    if target and target not in all_paths:
                        error_count += 1
            
            # Check for extremely deep nesting (potential cycles)
            max_depth = self._calculate_max_depth(json_obj)
            if max_depth > 20:  # Arbitrary threshold
                error_count += 1
            
            # Check for duplicate keys at same level (JSON spec violation)
            duplicate_keys = self._find_duplicate_keys(json_obj)
            error_count += len(duplicate_keys)
            
            is_valid = error_count == 0
            return is_valid, error_count
            
        except Exception as e:
            logger.error(f"Error checking structural validity: {e}")
            return False, 1
    
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_max_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _find_duplicate_keys(self, obj: Any) -> List[str]:
        """Find duplicate keys in JSON structure"""
        duplicates = []
        
        if isinstance(obj, dict):
            # This is tricky in Python since dict automatically handles duplicates
            # We'll check for case-insensitive duplicates as a proxy
            keys = list(obj.keys())
            lower_keys = [k.lower() for k in keys]
            for i, key in enumerate(lower_keys):
                if lower_keys.count(key) > 1 and keys[i] not in duplicates:
                    duplicates.append(keys[i])
            
            # Recurse into nested objects
            for value in obj.values():
                duplicates.extend(self._find_duplicate_keys(value))
                
        elif isinstance(obj, list):
            for item in obj:
                duplicates.extend(self._find_duplicate_keys(item))
        
        return duplicates
    
    def _calculate_rouge_scores(
        self, 
        predicted: Dict[str, Any], 
        reference: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """Calculate ROUGE-1, ROUGE-2, ROUGE-L scores"""
        try:
            # Extract text content from both structures
            pred_text = self._extract_text_content(predicted)
            ref_text = self._extract_text_content(reference)
            
            if not pred_text or not ref_text:
                return 0.0, 0.0, 0.0
            
            # Simple ROUGE implementation (could use py-rouge library for more accuracy)
            rouge_1 = self._calculate_rouge_n(pred_text, ref_text, 1)
            rouge_2 = self._calculate_rouge_n(pred_text, ref_text, 2)
            rouge_l = self._calculate_rouge_l(pred_text, ref_text)
            
            return rouge_1, rouge_2, rouge_l
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return 0.0, 0.0, 0.0
    
    def _extract_text_content(self, obj: Any) -> str:
        """Extract all text content from JSON structure"""
        text_parts = []
        
        if isinstance(obj, str):
            text_parts.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                text_parts.append(self._extract_text_content(value))
        elif isinstance(obj, list):
            for item in obj:
                text_parts.append(self._extract_text_content(item))
        elif obj is not None:
            text_parts.append(str(obj))
        
        return " ".join(filter(None, text_parts))
    
    def _calculate_rouge_n(self, predicted: str, reference: str, n: int) -> float:
        """Calculate ROUGE-N score"""
        pred_ngrams = self._get_ngrams(predicted.lower().split(), n)
        ref_ngrams = self._get_ngrams(reference.lower().split(), n)
        
        if not ref_ngrams:
            return 0.0
        
        overlap = len(pred_ngrams.intersection(ref_ngrams))
        return overlap / len(ref_ngrams)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> set:
        """Get n-grams from token list"""
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    def _calculate_rouge_l(self, predicted: str, reference: str) -> float:
        """Calculate ROUGE-L (Longest Common Subsequence) score"""
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if not ref_tokens:
            return 0.0
        
        return lcs_length / len(ref_tokens)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _calculate_semantic_coherence(
        self, 
        unified_json: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> float:
        """Calculate semantic coherence score"""
        try:
            # Extract key semantic fields for comparison
            unified_content = self._extract_semantic_content(unified_json)
            input_contents = [self._extract_semantic_content(inp) for inp in original_inputs]
            
            if not unified_content or not input_contents:
                return 0.0
            
            # Calculate semantic overlap with original inputs
            coherence_scores = []
            for input_content in input_contents:
                overlap = len(set(unified_content).intersection(set(input_content)))
                total = len(set(unified_content).union(set(input_content)))
                if total > 0:
                    coherence_scores.append(overlap / total)
            
            return statistics.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def _extract_semantic_content(self, obj: Any) -> List[str]:
        """Extract semantic content (key terms and concepts)"""
        content = []
        
        if isinstance(obj, str):
            # Extract meaningful words (simple tokenization)
            words = re.findall(r'\b\w+\b', obj.lower())
            content.extend([w for w in words if len(w) > 2])  # Filter short words
        elif isinstance(obj, dict):
            for key, value in obj.items():
                # Include key names as semantic content
                content.append(key.lower())
                content.extend(self._extract_semantic_content(value))
        elif isinstance(obj, list):
            for item in obj:
                content.extend(self._extract_semantic_content(item))
        
        return content
    
    def _calculate_expected_calibration_error(self, consensus_output: Dict[str, Any]) -> float:
        """Calculate Expected Calibration Error (ECE) for confidence scores"""
        try:
            # Extract confidence scores from consensus output
            confidence_scores = self._extract_confidence_scores(consensus_output)
            
            if not confidence_scores:
                return 0.0
            
            # For ECE calculation, we need predicted probabilities and actual outcomes
            # Since we don't have ground truth, we'll use consensus agreement as proxy for accuracy
            
            # Bin confidence scores
            num_bins = 10
            bin_boundaries = np.linspace(0, 1, num_bins + 1)
            
            ece = 0.0
            total_samples = len(confidence_scores)
            
            for i in range(num_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                # Find samples in this bin
                in_bin = [(conf, acc) for conf, acc in confidence_scores 
                         if bin_lower <= conf < bin_upper or (i == num_bins - 1 and conf == bin_upper)]
                
                if in_bin:
                    bin_confidence = np.mean([conf for conf, _ in in_bin])
                    bin_accuracy = np.mean([acc for _, acc in in_bin])
                    bin_weight = len(in_bin) / total_samples
                    
                    ece += bin_weight * abs(bin_confidence - bin_accuracy)
            
            return ece
            
        except Exception as e:
            logger.error(f"Error calculating Expected Calibration Error: {e}")
            return 0.0
    
    def _extract_confidence_scores(self, consensus_output: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Extract confidence scores and corresponding accuracy estimates"""
        confidence_scores = []
        
        try:
            # Look for confidence scores in various places
            if 'quality_metrics' in consensus_output:
                qm = consensus_output['quality_metrics']
                if 'average_consensus_confidence' in qm:
                    conf = float(qm['average_consensus_confidence'])
                    # Use coverage as proxy for accuracy
                    acc = qm.get('coverage', 0.5)
                    confidence_scores.append((conf, acc))
            
            # Look for individual node confidence scores
            unified_json = self._extract_unified_json(consensus_output)
            if unified_json:
                node_confidences = self._extract_node_confidences(unified_json)
                confidence_scores.extend(node_confidences)
            
        except Exception as e:
            logger.error(f"Error extracting confidence scores: {e}")
        
        return confidence_scores
    
    def _extract_node_confidences(self, obj: Any, parent_key: str = "") -> List[Tuple[float, float]]:
        """Extract node-level confidence scores"""
        confidences = []
        
        if isinstance(obj, dict):
            # Look for confidence-related fields
            confidence_fields = ['confidence', 'confidence_score', 'reliability', 'certainty']
            node_confidence = None
            
            for field in confidence_fields:
                if field in obj:
                    try:
                        node_confidence = float(obj[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if node_confidence is not None:
                # Use presence of supporting metadata as proxy for accuracy
                accuracy = 0.7  # Default moderate accuracy
                if '_consensus_metadata' in obj:
                    accuracy = 0.8
                if '_source_providers' in obj and len(obj.get('_source_providers', [])) > 1:
                    accuracy = 0.9
                
                confidences.append((node_confidence, accuracy))
            
            # Recurse into nested objects
            for key, value in obj.items():
                confidences.extend(self._extract_node_confidences(value, key))
                
        elif isinstance(obj, list):
            for item in obj:
                confidences.extend(self._extract_node_confidences(item, parent_key))
        
        return confidences
    
    def _calculate_reliability_score(self, consensus_output: Dict[str, Any]) -> float:
        """Calculate overall reliability score"""
        try:
            reliability_factors = []
            
            # Factor 1: Consensus success rate
            qm = consensus_output.get('quality_metrics', {})
            success_rate = qm.get('consensus_success_rate', 0.5)
            reliability_factors.append(success_rate)
            
            # Factor 2: Source diversity
            source_diversity = qm.get('source_diversity', 0.5)
            reliability_factors.append(source_diversity)
            
            # Factor 3: Data preservation
            data_preservation = qm.get('data_preservation', 0.5)
            reliability_factors.append(data_preservation)
            
            # Factor 4: Structural validity
            unified_json = self._extract_unified_json(consensus_output)
            if unified_json:
                is_valid, _ = self._check_structural_validity(unified_json)
                reliability_factors.append(1.0 if is_valid else 0.0)
            
            return statistics.mean(reliability_factors) if reliability_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {e}")
            return 0.0
    
    def _calculate_completeness_score(
        self, 
        unified_json: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> float:
        """Calculate completeness score (how much information was preserved)"""
        try:
            # Count information elements in original inputs
            total_original_elements = 0
            for input_data in original_inputs:
                parsed_json = input_data.get('parsed_json', input_data)
                total_original_elements += self._count_information_elements(parsed_json)
            
            # Count information elements in unified output
            unified_elements = self._count_information_elements(unified_json)
            
            # Calculate completeness ratio
            if total_original_elements == 0:
                return 0.0
            
            completeness = min(1.0, unified_elements / total_original_elements)
            return completeness
            
        except Exception as e:
            logger.error(f"Error calculating completeness score: {e}")
            return 0.0
    
    def _count_information_elements(self, obj: Any) -> int:
        """Count meaningful information elements"""
        count = 0
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not key.startswith('_'):  # Skip metadata fields
                    count += 1
                    count += self._count_information_elements(value)
        elif isinstance(obj, list):
            count += len(obj)
            for item in obj:
                count += self._count_information_elements(item)
        elif isinstance(obj, str) and len(obj.strip()) > 0:
            count += 1
        elif obj is not None:
            count += 1
        
        return count
    
    def _calculate_consistency_score(self, unified_json: Dict[str, Any]) -> float:
        """Calculate internal consistency score"""
        try:
            consistency_factors = []
            
            # Factor 1: Structural consistency (no contradictions)
            contradictions = self._find_contradictions(unified_json)
            consistency_factors.append(1.0 - min(1.0, len(contradictions) * 0.1))
            
            # Factor 2: Naming consistency
            naming_consistency = self._check_naming_consistency(unified_json)
            consistency_factors.append(naming_consistency)
            
            # Factor 3: Value type consistency
            type_consistency = self._check_type_consistency(unified_json)
            consistency_factors.append(type_consistency)
            
            return statistics.mean(consistency_factors) if consistency_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0.0
    
    def _find_contradictions(self, obj: Any) -> List[str]:
        """Find potential contradictions in the data"""
        contradictions = []
        
        # This is a simplified contradiction detection
        # Could be enhanced with domain-specific rules
        
        if isinstance(obj, dict):
            # Look for conflicting boolean values
            true_fields = [k for k, v in obj.items() if v is True]
            false_fields = [k for k, v in obj.items() if v is False]
            
            # Check for opposite meanings
            opposites = [
                ('enable', 'disable'), ('allow', 'deny'), ('secure', 'insecure'),
                ('valid', 'invalid'), ('active', 'inactive')
            ]
            
            for pos, neg in opposites:
                pos_found = any(pos in field.lower() for field in true_fields)
                neg_found = any(neg in field.lower() for field in true_fields)
                if pos_found and neg_found:
                    contradictions.append(f"Contradictory boolean values: {pos} and {neg}")
            
            # Recurse into nested objects
            for value in obj.values():
                contradictions.extend(self._find_contradictions(value))
                
        elif isinstance(obj, list):
            for item in obj:
                contradictions.extend(self._find_contradictions(item))
        
        return contradictions
    
    def _check_naming_consistency(self, obj: Any) -> float:
        """Check consistency of naming conventions"""
        try:
            all_keys = self._extract_all_keys(obj)
            
            if not all_keys:
                return 1.0
            
            # Check for consistent naming patterns
            snake_case = sum(1 for key in all_keys if '_' in key and key.islower())
            camel_case = sum(1 for key in all_keys if any(c.isupper() for c in key[1:]) and '_' not in key)
            
            total_keys = len(all_keys)
            snake_ratio = snake_case / total_keys
            camel_ratio = camel_case / total_keys
            
            # Consistency is higher when one pattern dominates
            consistency = max(snake_ratio, camel_ratio)
            return consistency
            
        except Exception as e:
            logger.error(f"Error checking naming consistency: {e}")
            return 0.0
    
    def _extract_all_keys(self, obj: Any) -> List[str]:
        """Extract all keys from nested JSON structure"""
        keys = []
        
        if isinstance(obj, dict):
            keys.extend(obj.keys())
            for value in obj.values():
                keys.extend(self._extract_all_keys(value))
        elif isinstance(obj, list):
            for item in obj:
                keys.extend(self._extract_all_keys(item))
        
        return keys
    
    def _check_type_consistency(self, obj: Any) -> float:
        """Check consistency of value types for similar keys"""
        try:
            key_types = defaultdict(list)
            self._collect_key_types(obj, key_types)
            
            consistency_scores = []
            for key, types in key_types.items():
                if len(types) > 1:
                    # Calculate type consistency for this key
                    type_counts = defaultdict(int)
                    for t in types:
                        type_counts[t] += 1
                    
                    most_common_count = max(type_counts.values())
                    consistency = most_common_count / len(types)
                    consistency_scores.append(consistency)
                else:
                    consistency_scores.append(1.0)
            
            return statistics.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception as e:
            logger.error(f"Error checking type consistency: {e}")
            return 0.0
    
    def _collect_key_types(self, obj: Any, key_types: Dict[str, List[str]], prefix: str = ""):
        """Collect types for each key in the structure"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                key_types[key].append(type(value).__name__)
                
                if isinstance(value, (dict, list)):
                    self._collect_key_types(value, key_types, full_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._collect_key_types(item, key_types, f"{prefix}[{i}]")
    
    def _calculate_coverage_ratio(
        self, 
        unified_json: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> float:
        """Calculate coverage ratio (breadth of information covered)"""
        try:
            # Extract topic/domain coverage
            unified_topics = self._extract_topics(unified_json)
            
            all_original_topics = set()
            for input_data in original_inputs:
                parsed_json = input_data.get('parsed_json', input_data)
                topics = self._extract_topics(parsed_json)
                all_original_topics.update(topics)
            
            if not all_original_topics:
                return 0.0
            
            covered_topics = unified_topics.intersection(all_original_topics)
            coverage_ratio = len(covered_topics) / len(all_original_topics)
            
            return coverage_ratio
            
        except Exception as e:
            logger.error(f"Error calculating coverage ratio: {e}")
            return 0.0
    
    def _extract_topics(self, obj: Any) -> set:
        """Extract topic/domain indicators from JSON structure"""
        topics = set()
        
        # Domain-specific topic keywords (can be extended)
        topic_keywords = {
            'security', 'threat', 'attack', 'vulnerability', 'malicious', 'anomalous',
            'authentication', 'authorization', 'encryption', 'network', 'system',
            'behavioral', 'temporal', 'pattern', 'detection', 'monitoring',
            'compliance', 'risk', 'incident', 'forensic', 'analysis'
        }
        
        if isinstance(obj, str):
            words = re.findall(r'\b\w+\b', obj.lower())
            topics.update(word for word in words if word in topic_keywords)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                # Key names can indicate topics
                words = re.findall(r'\b\w+\b', key.lower())
                topics.update(word for word in words if word in topic_keywords)
                
                # Recurse into values
                topics.update(self._extract_topics(value))
        elif isinstance(obj, list):
            for item in obj:
                topics.update(self._extract_topics(item))
        
        return topics
    
    def _calculate_information_density(self, unified_json: Dict[str, Any]) -> float:
        """Calculate information density (information per node)"""
        try:
            total_nodes = self._count_nodes(unified_json)
            text_content = self._extract_text_content(unified_json)
            
            if total_nodes == 0:
                return 0.0
            
            # Simple measure: average text length per node
            avg_text_per_node = len(text_content) / total_nodes
            
            # Normalize to 0-1 scale (assuming 100 chars per node is high density)
            density = min(1.0, avg_text_per_node / 100.0)
            
            return density
            
        except Exception as e:
            logger.error(f"Error calculating information density: {e}")
            return 0.0
    
    def _calculate_consensus_stability(self, consensus_output: Dict[str, Any]) -> float:
        """Calculate consensus stability score"""
        try:
            stability_factors = []
            
            # Factor 1: Quality metric stability (no extreme values)
            qm = consensus_output.get('quality_metrics', {})
            quality_values = [v for v in qm.values() if isinstance(v, (int, float))]
            
            if quality_values:
                # Penalize extreme values (very high or very low)
                normalized_values = [abs(v - 0.5) for v in quality_values if 0 <= v <= 1]
                stability = 1.0 - statistics.mean(normalized_values) if normalized_values else 0.5
                stability_factors.append(stability)
            
            # Factor 2: Processing metadata stability
            pm = consensus_output.get('processing_metadata', {})
            if 'consensus_parts_count' in pm and 'original_file_count' in pm:
                parts_per_file = pm['consensus_parts_count'] / max(1, pm['original_file_count'])
                # Stable if reasonable parts per file (not too many, not too few)
                if 5 <= parts_per_file <= 50:
                    stability_factors.append(1.0)
                else:
                    stability_factors.append(0.5)
            
            return statistics.mean(stability_factors) if stability_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating consensus stability: {e}")
            return 0.0
    
    def _calculate_overall_quality(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score"""
        try:
            # Define weights for different aspects
            weights = {
                'accuracy': 0.25,      # F1 score / consensus agreement
                'completeness': 0.20,  # Information preservation
                'consistency': 0.20,   # Internal consistency
                'reliability': 0.15,   # Confidence calibration
                'coverage': 0.10,      # Breadth of coverage
                'structural': 0.10     # Structural validity
            }
            
            # Calculate weighted score
            overall_score = (
                weights['accuracy'] * metrics.accuracy_score +
                weights['completeness'] * metrics.completeness_score +
                weights['consistency'] * metrics.consistency_score +
                weights['reliability'] * metrics.reliability_score +
                weights['coverage'] * metrics.coverage_ratio +
                weights['structural'] * (1.0 if metrics.structural_validity else 0.0)
            )
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 0.0
    
    def _llm_judge_assessment(
        self, 
        unified_json: Dict[str, Any], 
        original_inputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Use Gemini 2.5 Pro as LLM judge for subjective quality assessment"""
        try:
            if not self.gemini_provider:
                return {}
            
            # Prepare prompt for LLM judge
            unified_text = json.dumps(unified_json, indent=2)[:2000]  # Truncate for API limits
            original_text = json.dumps(original_inputs[0], indent=2)[:1000] if original_inputs else ""
            
            prompt = f"""
You are an expert evaluator assessing the quality of a consensus-based information fusion system.

Original Input Sample:
{original_text}

Unified Consensus Output:
{unified_text}

Please evaluate the consensus output on the following dimensions (score 0.0-1.0):

1. Semantic Coherence: How well does the output maintain logical consistency and meaningful relationships?
2. Content Quality: How well does the output preserve and synthesize important information?
3. Structural Organization: How well is the information organized and structured?
4. Completeness: How comprehensive is the coverage of important topics from the inputs?

Provide scores as a JSON object with keys: coherence, quality, structure, completeness.
Only respond with the JSON object, no additional text.
"""
            
            response = self.gemini_provider.generate_text(
                prompt=prompt,
                model="gemini-2.5-pro",
                search_enabled=True
            )
            
            # Parse LLM response
            try:
                scores = json.loads(response)
                return {
                    'coherence': float(scores.get('coherence', 0.5)),
                    'quality': float(scores.get('quality', 0.5)),
                    'structure': float(scores.get('structure', 0.5)),
                    'completeness': float(scores.get('completeness', 0.5))
                }
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse LLM judge response")
                return {}
                
        except Exception as e:
            logger.error(f"LLM judge assessment failed: {e}")
            return {}

def create_quality_report(metrics: QualityMetrics, output_file: Optional[str] = None) -> str:
    """Create a comprehensive quality assessment report"""
    
    report = f"""
# Consensus Quality Assessment Report

Generated: {metrics.evaluation_timestamp}
Input Files: {metrics.input_file_count}
Output Nodes: {metrics.output_node_count}

## Overall Quality Score: {metrics.overall_quality_score:.3f}

## Detailed Metrics

### Node Inclusion Accuracy
- Precision: {metrics.precision:.3f}
- Recall: {metrics.recall:.3f}
- F1 Score: {metrics.f1_score:.3f}

### Structural Quality
- Structural Validity: {'✓' if metrics.structural_validity else '✗'}
- Structural Errors: {metrics.structural_error_count}

### Semantic Quality
- ROUGE-1: {metrics.rouge_1:.3f}
- ROUGE-2: {metrics.rouge_2:.3f}
- ROUGE-L: {metrics.rouge_l:.3f}
- Semantic Coherence: {metrics.semantic_coherence:.3f}

### Confidence Calibration
- Expected Calibration Error: {metrics.expected_calibration_error:.3f}
- Reliability Score: {metrics.reliability_score:.3f}

### Multi-dimensional Quality
- Completeness Score: {metrics.completeness_score:.3f}
- Consistency Score: {metrics.consistency_score:.3f}
- Accuracy Score: {metrics.accuracy_score:.3f}
- Coverage Ratio: {metrics.coverage_ratio:.3f}
- Information Density: {metrics.information_density:.3f}
- Consensus Stability: {metrics.consensus_stability:.3f}

## Quality Assessment Summary

{'✓ EXCELLENT' if metrics.overall_quality_score >= 0.8 else '✓ GOOD' if metrics.overall_quality_score >= 0.6 else '⚠ ACCEPTABLE' if metrics.overall_quality_score >= 0.4 else '✗ POOR'} - Overall Quality Score: {metrics.overall_quality_score:.3f}

### Strengths:
{_identify_strengths(metrics)}

### Areas for Improvement:
{_identify_weaknesses(metrics)}

### Recommendations:
{_generate_recommendations(metrics)}
"""
    
    if output_file:
        # Remove Unicode characters for Windows compatibility
        safe_report = report.replace('✓', 'SUCCESS').replace('⚠', 'WARNING').replace('✗', 'FAILED')
        Path(output_file).write_text(safe_report, encoding='utf-8')
        logger.info(f"Quality report saved to {output_file}")
    
    return report

def _identify_strengths(metrics: QualityMetrics) -> str:
    """Identify strengths based on metrics"""
    strengths = []
    
    if metrics.structural_validity:
        strengths.append("- Strong structural integrity")
    if metrics.f1_score >= 0.7:
        strengths.append("- High accuracy in node inclusion")
    if metrics.completeness_score >= 0.8:
        strengths.append("- Excellent information preservation")
    if metrics.consistency_score >= 0.8:
        strengths.append("- High internal consistency")
    if metrics.reliability_score >= 0.7:
        strengths.append("- Reliable confidence calibration")
    
    return "\n".join(strengths) if strengths else "- System shows baseline performance"

def _identify_weaknesses(metrics: QualityMetrics) -> str:
    """Identify weaknesses based on metrics"""
    weaknesses = []
    
    if not metrics.structural_validity:
        weaknesses.append("- Structural validity issues detected")
    if metrics.f1_score < 0.5:
        weaknesses.append("- Low accuracy in node inclusion")
    if metrics.completeness_score < 0.6:
        weaknesses.append("- Information loss during consensus")
    if metrics.consistency_score < 0.6:
        weaknesses.append("- Internal consistency issues")
    if metrics.expected_calibration_error > 0.2:
        weaknesses.append("- Poor confidence calibration")
    
    return "\n".join(weaknesses) if weaknesses else "- No significant weaknesses identified"

def _generate_recommendations(metrics: QualityMetrics) -> str:
    """Generate recommendations based on metrics"""
    recommendations = []
    
    if metrics.completeness_score < 0.7:
        recommendations.append("- Adjust preservation_ratio to retain more information")
    if metrics.consistency_score < 0.7:
        recommendations.append("- Improve conflict resolution mechanisms")
    if metrics.coverage_ratio < 0.6:
        recommendations.append("- Enhance topic coverage in consensus algorithm")
    if metrics.expected_calibration_error > 0.15:
        recommendations.append("- Calibrate confidence scoring mechanisms")
    if not metrics.structural_validity:
        recommendations.append("- Fix structural validation issues")
    
    if not recommendations:
        recommendations.append("- Continue monitoring quality metrics")
        recommendations.append("- Consider testing with more diverse inputs")
    
    return "\n".join(recommendations)

# Fix import error
try:
    import pandas as pd
except ImportError:
    # Fallback timestamp
    from datetime import datetime
    class pd:
        class Timestamp:
            @staticmethod
            def now():
                return datetime.now()