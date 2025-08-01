"""
Dempster-Shafer Uncertainty Quantification for Multi-Agent Schema Consensus
Evidence theory for handling uncertainty and conflicting evidence in field consensus
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set, FrozenSet
from collections import defaultdict
from itertools import combinations, chain
import json
from fractions import Fraction
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)

@dataclass
class Evidence:
    """Evidence for Dempster-Shafer theory"""
    source: str
    hypothesis_set: FrozenSet[str]
    belief_mass: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DempsterShaferEngine:
    """Dempster-Shafer uncertainty quantification engine"""
    
    def __init__(self):
        """Initialize Dempster-Shafer engine"""
        self.frame_of_discernment = set()  # Θ (Theta) - set of all possible hypotheses
        self.evidence_base = []
        self.mass_functions = {}  # Basic probability assignments
        self.belief_functions = {}
        self.plausibility_functions = {}
        self.uncertainty_measures = {}
        
    def define_frame_of_discernment(self, provider_results: Dict[str, Dict]) -> Set[str]:
        """
        Define frame of discernment from provider field results
        
        Args:
            provider_results: Dictionary of provider results
            
        Returns:
            Set of all possible field hypotheses
        """
        hypotheses = set()
        
        for provider, result in provider_results.items():
            if 'parsed_json' not in result or 'fields' not in result['parsed_json']:
                continue
                
            fields = result['parsed_json']['fields']
            for field_name, field_data in fields.items():
                # Create hypotheses for field properties
                if 'type' in field_data:
                    hypotheses.add(f"type_{field_data['type'].lower()}")
                
                # Framework mapping hypotheses
                for framework in ['OCSF', 'ECS', 'OSSEM']:
                    mapping = field_data.get(framework)
                    if mapping and mapping != 'null':
                        hypotheses.add(f"{framework.lower()}_{mapping}")
                
                # Importance level hypotheses
                importance = field_data.get('importance', 5)
                if importance <= 3:
                    hypotheses.add('low_importance')
                elif importance <= 7:
                    hypotheses.add('medium_importance')
                else:
                    hypotheses.add('high_importance')
                
                # Field category hypotheses based on description
                description = field_data.get('description', '').lower()
                if any(keyword in description for keyword in ['user', 'account', 'identity']):
                    hypotheses.add('user_related')
                if any(keyword in description for keyword in ['process', 'execution', 'command']):
                    hypotheses.add('process_related')
                if any(keyword in description for keyword in ['network', 'ip', 'address']):
                    hypotheses.add('network_related')
                if any(keyword in description for keyword in ['time', 'timestamp', 'date']):
                    hypotheses.add('temporal_related')
        
        self.frame_of_discernment = hypotheses
        logger.info(f"Defined frame of discernment with {len(hypotheses)} hypotheses: {list(hypotheses)[:10]}...")
        
        return hypotheses
    
    def extract_evidence_from_providers(self, 
                                      provider_results: Dict[str, Dict],
                                      field_name: str) -> List[Evidence]:
        """
        Extract evidence for a specific field from all providers
        
        Args:
            provider_results: Dictionary of provider results
            field_name: Name of field to extract evidence for
            
        Returns:
            List of evidence from each provider
        """
        evidence_list = []
        
        for provider, result in provider_results.items():
            if 'parsed_json' not in result or 'fields' not in result['parsed_json']:
                continue
                
            fields = result['parsed_json']['fields']
            
            # Find matching field (exact or similar name)
            matching_field_data = None
            for fname, fdata in fields.items():
                if fname.lower() == field_name.lower():
                    matching_field_data = fdata
                    break
            
            if matching_field_data is None:
                # Look for similar field names
                for fname, fdata in fields.items():
                    if self._field_name_similarity(fname.lower(), field_name.lower()) > 0.7:
                        matching_field_data = fdata
                        break
            
            if matching_field_data:
                # Extract evidence from field data
                field_evidence = self._extract_field_evidence(
                    provider, matching_field_data, field_name
                )
                evidence_list.extend(field_evidence)
        
        logger.info(f"Extracted {len(evidence_list)} pieces of evidence for field '{field_name}'")
        return evidence_list
    
    def _field_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between field names"""
        name1_words = set(name1.replace('_', ' ').split())
        name2_words = set(name2.replace('_', ' ').split())
        
        if not name1_words or not name2_words:
            return 0.0
        
        intersection = name1_words.intersection(name2_words)
        union = name1_words.union(name2_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_field_evidence(self, 
                              provider: str, 
                              field_data: Dict[str, Any],
                              field_name: str) -> List[Evidence]:
        """Extract evidence from a single field definition"""
        evidence_list = []
        
        # Evidence for field type
        field_type = field_data.get('type', '').lower()
        if field_type:
            type_confidence = 0.9  # High confidence in type evidence
            type_evidence = Evidence(
                source=provider,
                hypothesis_set=frozenset([f"type_{field_type}"]),
                belief_mass=type_confidence,
                confidence=type_confidence,
                metadata={'field_name': field_name, 'evidence_type': 'type'}
            )
            evidence_list.append(type_evidence)
        
        # Evidence for framework mappings
        for framework in ['OCSF', 'ECS', 'OSSEM']:
            mapping = field_data.get(framework)
            if mapping and mapping != 'null':
                mapping_confidence = 0.8  # High confidence in framework mappings
                mapping_evidence = Evidence(
                    source=provider,
                    hypothesis_set=frozenset([f"{framework.lower()}_{mapping}"]),
                    belief_mass=mapping_confidence,
                    confidence=mapping_confidence,
                    metadata={'field_name': field_name, 'evidence_type': 'framework_mapping', 'framework': framework}
                )
                evidence_list.append(mapping_evidence)
        
        # Evidence for importance level
        importance = field_data.get('importance', 5)
        importance_category = 'medium_importance'
        if importance <= 3:
            importance_category = 'low_importance'
        elif importance >= 8:
            importance_category = 'high_importance'
        
        importance_confidence = min(0.7, importance / 10)  # Confidence based on importance value
        importance_evidence = Evidence(
            source=provider,
            hypothesis_set=frozenset([importance_category]),
            belief_mass=importance_confidence,
            confidence=importance_confidence,
            metadata={'field_name': field_name, 'evidence_type': 'importance', 'raw_importance': importance}
        )
        evidence_list.append(importance_evidence)
        
        # Evidence for field categories based on description
        description = field_data.get('description', '').lower()
        category_hypotheses = []
        
        if any(keyword in description for keyword in ['user', 'account', 'identity']):
            category_hypotheses.append('user_related')
        if any(keyword in description for keyword in ['process', 'execution', 'command']):
            category_hypotheses.append('process_related')
        if any(keyword in description for keyword in ['network', 'ip', 'address']):
            category_hypotheses.append('network_related')
        if any(keyword in description for keyword in ['time', 'timestamp', 'date']):
            category_hypotheses.append('temporal_related')
        
        if category_hypotheses:
            category_confidence = 0.6  # Medium confidence for category inference
            category_evidence = Evidence(
                source=provider,
                hypothesis_set=frozenset(category_hypotheses),
                belief_mass=category_confidence,
                confidence=category_confidence,
                metadata={'field_name': field_name, 'evidence_type': 'category', 'description': description[:100]}
            )
            evidence_list.append(category_evidence)
        
        return evidence_list
    
    def compute_basic_probability_assignment(self, evidence_list: List[Evidence]) -> Dict[FrozenSet[str], float]:
        """
        Compute basic probability assignment (mass function) from evidence
        
        Args:
            evidence_list: List of evidence to process
            
        Returns:
            Dictionary mapping hypothesis sets to belief masses
        """
        mass_function = defaultdict(float)
        
        # Group evidence by source
        source_evidence = defaultdict(list)
        for evidence in evidence_list:
            source_evidence[evidence.source].append(evidence)
        
        # Compute mass for each source
        for source, evidences in source_evidence.items():
            # Normalize belief masses for this source
            total_mass = sum(evidence.belief_mass for evidence in evidences)
            
            if total_mass > 0:
                normalization_factor = min(1.0, 1.0 / total_mass)
                
                for evidence in evidences:
                    normalized_mass = evidence.belief_mass * normalization_factor
                    mass_function[evidence.hypothesis_set] += normalized_mass
        
        # Add mass to uncertainty (frame of discernment)
        assigned_mass = sum(mass_function.values())
        if assigned_mass < 1.0:
            uncertainty_mass = 1.0 - assigned_mass
            mass_function[frozenset(self.frame_of_discernment)] += uncertainty_mass
        
        logger.info(f"Computed basic probability assignment with {len(mass_function)} mass assignments")
        return dict(mass_function)
    
    def dempster_combination_rule(self, 
                                mass_function1: Dict[FrozenSet[str], float],
                                mass_function2: Dict[FrozenSet[str], float]) -> Dict[FrozenSet[str], float]:
        """
        Combine two mass functions using Dempster's rule of combination
        
        Args:
            mass_function1: First mass function
            mass_function2: Second mass function
            
        Returns:
            Combined mass function
        """
        combined_mass = defaultdict(float)
        conflict_mass = 0.0
        
        # Compute combinations
        for A1, mass1 in mass_function1.items():
            for A2, mass2 in mass_function2.items():
                intersection = A1 & A2
                product = mass1 * mass2
                
                if intersection:  # Non-empty intersection
                    combined_mass[intersection] += product
                else:  # Empty intersection - conflict
                    conflict_mass += product
        
        # Handle conflict
        if conflict_mass >= 1.0:
            logger.warning("Total conflict detected in Dempster combination")
            # Return uniform distribution over frame of discernment
            theta_mass = 1.0 / len(self.frame_of_discernment)
            return {frozenset([hyp]): theta_mass for hyp in self.frame_of_discernment}
        
        # Normalize by (1 - conflict)
        normalization_factor = 1.0 / (1.0 - conflict_mass) if conflict_mass < 1.0 else 1.0
        
        normalized_mass = {}
        for hypothesis_set, mass in combined_mass.items():
            normalized_mass[hypothesis_set] = mass * normalization_factor
        
        logger.info(f"Combined mass functions with conflict level: {conflict_mass:.3f}")
        return normalized_mass
    
    def compute_belief_function(self, mass_function: Dict[FrozenSet[str], float]) -> Dict[FrozenSet[str], float]:
        """
        Compute belief function from mass function
        
        Args:
            mass_function: Basic probability assignment
            
        Returns:
            Belief function values
        """
        belief_function = {}
        
        for A in mass_function.keys():
            belief_value = 0.0
            
            # Bel(A) = sum of m(B) for all B ⊆ A
            for B, mass_B in mass_function.items():
                if B.issubset(A):
                    belief_value += mass_B
            
            belief_function[A] = belief_value
        
        return belief_function
    
    def compute_plausibility_function(self, mass_function: Dict[FrozenSet[str], float]) -> Dict[FrozenSet[str], float]:
        """
        Compute plausibility function from mass function
        
        Args:
            mass_function: Basic probability assignment
            
        Returns:
            Plausibility function values
        """
        plausibility_function = {}
        
        for A in mass_function.keys():
            plausibility_value = 0.0
            
            # Pl(A) = sum of m(B) for all B ∩ A ≠ ∅
            for B, mass_B in mass_function.items():
                if A & B:  # Non-empty intersection
                    plausibility_value += mass_B
            
            plausibility_function[A] = plausibility_value
        
        return plausibility_function
    
    def compute_uncertainty_measures(self, 
                                   mass_function: Dict[FrozenSet[str], float],
                                   belief_function: Dict[FrozenSet[str], float],
                                   plausibility_function: Dict[FrozenSet[str], float]) -> Dict[str, float]:
        """
        Compute various uncertainty measures
        
        Args:
            mass_function: Basic probability assignment
            belief_function: Belief function
            plausibility_function: Plausibility function
            
        Returns:
            Dictionary of uncertainty measures
        """
        measures = {}
        
        # Shannon entropy of mass function
        shannon_entropy = 0.0
        for mass in mass_function.values():
            if mass > 0:
                shannon_entropy -= mass * np.log2(mass)
        measures['shannon_entropy'] = shannon_entropy
        
        # Confusion (total mass assigned to non-singleton sets)
        confusion = 0.0
        for hypothesis_set, mass in mass_function.items():
            if len(hypothesis_set) > 1:
                confusion += mass
        measures['confusion'] = confusion
        
        # Discord (measure of conflict)
        discord = 0.0
        for A in mass_function.keys():
            if len(A) == 1:  # Singleton
                belief_val = belief_function.get(A, 0.0)
                plausibility_val = plausibility_function.get(A, 0.0)
                discord += plausibility_val - belief_val
        measures['discord'] = discord
        
        # Non-specificity (Hartley measure)
        nonspecificity = 0.0
        for hypothesis_set, mass in mass_function.items():
            if len(hypothesis_set) > 0:
                nonspecificity += mass * np.log2(len(hypothesis_set))
        measures['nonspecificity'] = nonspecificity
        
        # Total uncertainty
        measures['total_uncertainty'] = measures['confusion'] + measures['discord'] + measures['nonspecificity']
        
        return measures
    
    def field_consensus_analysis(self, 
                               provider_results: Dict[str, Dict],
                               field_name: str) -> Dict[str, Any]:
        """
        Perform complete Dempster-Shafer analysis for field consensus
        
        Args:
            provider_results: Dictionary of provider results
            field_name: Name of field to analyze
            
        Returns:
            Complete DS analysis results
        """
        try:
            # Define frame of discernment if not already done
            if not self.frame_of_discernment:
                self.define_frame_of_discernment(provider_results)
            
            # Extract evidence for this field
            evidence_list = self.extract_evidence_from_providers(provider_results, field_name)
            
            if not evidence_list:
                return {'error': f'No evidence found for field {field_name}'}
            
            # Compute basic probability assignment
            mass_function = self.compute_basic_probability_assignment(evidence_list)
            
            # Combine evidence from multiple sources using Dempster's rule
            combined_mass = self._combine_multiple_sources(evidence_list)
            
            # Compute belief and plausibility functions
            belief_function = self.compute_belief_function(combined_mass)
            plausibility_function = self.compute_plausibility_function(combined_mass)
            
            # Compute uncertainty measures
            uncertainty_measures = self.compute_uncertainty_measures(
                combined_mass, belief_function, plausibility_function
            )
            
            # Extract consensus recommendations
            consensus_recommendations = self._extract_consensus_recommendations(
                combined_mass, belief_function, plausibility_function
            )
            
            # Detect conflicts
            conflicts = self._detect_evidence_conflicts(evidence_list, combined_mass)
            
            analysis_result = {
                'field_name': field_name,
                'evidence_count': len(evidence_list),
                'mass_function': self._serialize_frozen_sets(combined_mass),
                'belief_function': self._serialize_frozen_sets(belief_function),
                'plausibility_function': self._serialize_frozen_sets(plausibility_function),
                'uncertainty_measures': uncertainty_measures,
                'consensus_recommendations': consensus_recommendations,
                'conflicts': conflicts,
                'frame_of_discernment': list(self.frame_of_discernment)
            }
            
            logger.info(f"DS analysis completed for field '{field_name}' with uncertainty: {uncertainty_measures['total_uncertainty']:.3f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"DS analysis failed for field '{field_name}': {e}")
            return {'error': str(e)}
    
    def _combine_multiple_sources(self, evidence_list: List[Evidence]) -> Dict[FrozenSet[str], float]:
        """Combine evidence from multiple sources using Dempster's rule"""
        # Group evidence by source
        source_evidence = defaultdict(list)
        for evidence in evidence_list:
            source_evidence[evidence.source].append(evidence)
        
        # Compute mass function for each source
        source_mass_functions = []
        for source, evidences in source_evidence.items():
            source_mass = self.compute_basic_probability_assignment(evidences)
            source_mass_functions.append(source_mass)
        
        if not source_mass_functions:
            return {}
        
        # Combine mass functions using Dempster's rule
        combined_mass = source_mass_functions[0]
        
        for i in range(1, len(source_mass_functions)):
            combined_mass = self.dempster_combination_rule(combined_mass, source_mass_functions[i])
        
        return combined_mass
    
    def _extract_consensus_recommendations(self, 
                                         mass_function: Dict[FrozenSet[str], float],
                                         belief_function: Dict[FrozenSet[str], float],
                                         plausibility_function: Dict[FrozenSet[str], float]) -> Dict[str, Any]:
        """Extract consensus recommendations from DS analysis"""
        recommendations = {
            'high_belief_hypotheses': [],
            'high_plausibility_hypotheses': [],
            'consensus_field_properties': {},
            'confidence_intervals': {}
        }
        
        # Find hypotheses with high belief
        belief_threshold = 0.6
        for hypothesis_set, belief_val in belief_function.items():
            if belief_val >= belief_threshold and len(hypothesis_set) == 1:
                hypothesis = list(hypothesis_set)[0]
                recommendations['high_belief_hypotheses'].append({
                    'hypothesis': hypothesis,
                    'belief': belief_val,
                    'plausibility': plausibility_function.get(hypothesis_set, 0.0)
                })
        
        # Find hypotheses with high plausibility but low belief (uncertain)
        plausibility_threshold = 0.7
        uncertainty_threshold = 0.3
        
        for hypothesis_set, plausibility_val in plausibility_function.items():
            belief_val = belief_function.get(hypothesis_set, 0.0)
            uncertainty = plausibility_val - belief_val
            
            if plausibility_val >= plausibility_threshold and uncertainty >= uncertainty_threshold and len(hypothesis_set) == 1:
                hypothesis = list(hypothesis_set)[0]
                recommendations['high_plausibility_hypotheses'].append({
                    'hypothesis': hypothesis,
                    'belief': belief_val,
                    'plausibility': plausibility_val,
                    'uncertainty': uncertainty
                })
        
        # Extract field properties from high-belief hypotheses
        for hyp_info in recommendations['high_belief_hypotheses']:
            hypothesis = hyp_info['hypothesis']
            
            if hypothesis.startswith('type_'):
                recommendations['consensus_field_properties']['type'] = {
                    'value': hypothesis[5:],  # Remove 'type_' prefix
                    'confidence': hyp_info['belief']
                }
            elif hypothesis.endswith('_importance'):
                recommendations['consensus_field_properties']['importance_level'] = {
                    'value': hypothesis.replace('_importance', ''),
                    'confidence': hyp_info['belief']
                }
            elif any(hypothesis.startswith(fw.lower() + '_') for fw in ['OCSF', 'ECS', 'OSSEM']):
                framework = hypothesis.split('_')[0].upper()
                mapping = '_'.join(hypothesis.split('_')[1:])
                if 'framework_mappings' not in recommendations['consensus_field_properties']:
                    recommendations['consensus_field_properties']['framework_mappings'] = {}
                
                recommendations['consensus_field_properties']['framework_mappings'][framework] = {
                    'value': mapping,
                    'confidence': hyp_info['belief']
                }
        
        return recommendations
    
    def _detect_evidence_conflicts(self, 
                                 evidence_list: List[Evidence], 
                                 combined_mass: Dict[FrozenSet[str], float]) -> List[Dict[str, Any]]:
        """Detect conflicts in evidence"""
        conflicts = []
        
        # Group evidence by type
        evidence_by_type = defaultdict(list)
        for evidence in evidence_list:
            evidence_type = evidence.metadata.get('evidence_type', 'unknown')
            evidence_by_type[evidence_type].append(evidence)
        
        # Check for conflicts within each evidence type
        for evidence_type, evidences in evidence_by_type.items():
            if len(evidences) < 2:
                continue
            
            # Check for conflicting hypotheses
            hypothesis_sources = defaultdict(list)
            for evidence in evidences:
                for hypothesis in evidence.hypothesis_set:
                    hypothesis_sources[hypothesis].append(evidence.source)
            
            # Find conflicting evidence (multiple sources for different hypotheses of same type)
            type_hypotheses = list(hypothesis_sources.keys())
            if len(type_hypotheses) > 1:
                conflicts.append({
                    'evidence_type': evidence_type,
                    'conflicting_hypotheses': type_hypotheses,
                    'source_distribution': dict(hypothesis_sources),
                    'conflict_level': len(type_hypotheses) / len(evidences)
                })
        
        # Check for high mass on empty set (indicator of conflict)
        empty_set_mass = combined_mass.get(frozenset(), 0.0)
        if empty_set_mass > 0.1:
            conflicts.append({
                'type': 'fundamental_conflict',
                'empty_set_mass': empty_set_mass,
                'description': 'High mass assigned to empty set indicates fundamental conflict'
            })
        
        return conflicts
    
    def _serialize_frozen_sets(self, frozen_set_dict: Dict[FrozenSet[str], float]) -> Dict[str, float]:
        """Convert frozenset keys to string representation for JSON serialization"""
        serialized = {}
        for frozen_set, value in frozen_set_dict.items():
            key = '{' + ', '.join(sorted(frozen_set)) + '}'
            serialized[key] = value
        return serialized
    
    def multi_field_consensus_analysis(self, provider_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Perform Dempster-Shafer analysis for multiple fields across providers
        
        Args:
            provider_results: Dictionary of provider results
            
        Returns:
            Multi-field DS analysis results
        """
        try:
            # Get all unique field names
            all_field_names = set()
            for result in provider_results.values():
                if 'parsed_json' in result and 'fields' in result['parsed_json']:
                    all_field_names.update(result['parsed_json']['fields'].keys())
            
            multi_field_results = {
                'total_fields': len(all_field_names),
                'field_analyses': {},
                'overall_uncertainty': {},
                'global_conflicts': [],
                'consensus_summary': {}
            }
            
            # Analyze each field
            field_uncertainties = []
            for field_name in all_field_names:
                field_analysis = self.field_consensus_analysis(provider_results, field_name)
                multi_field_results['field_analyses'][field_name] = field_analysis
                
                if 'uncertainty_measures' in field_analysis:
                    field_uncertainties.append(field_analysis['uncertainty_measures']['total_uncertainty'])
            
            # Compute overall uncertainty statistics
            if field_uncertainties:
                multi_field_results['overall_uncertainty'] = {
                    'average_uncertainty': np.mean(field_uncertainties),
                    'max_uncertainty': np.max(field_uncertainties),
                    'min_uncertainty': np.min(field_uncertainties),
                    'std_uncertainty': np.std(field_uncertainties),
                    'high_uncertainty_fields': sum(1 for u in field_uncertainties if u > 2.0),
                    'low_uncertainty_fields': sum(1 for u in field_uncertainties if u < 0.5)
                }
            
            # Generate consensus summary
            multi_field_results['consensus_summary'] = self._generate_multi_field_consensus_summary(
                multi_field_results['field_analyses']
            )
            
            logger.info(f"Multi-field DS analysis completed for {len(all_field_names)} fields")
            return multi_field_results
            
        except Exception as e:
            logger.error(f"Multi-field DS analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_multi_field_consensus_summary(self, field_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary of multi-field consensus analysis"""
        summary = {
            'fields_with_strong_consensus': [],
            'fields_with_conflicts': [],
            'common_field_types': defaultdict(int),
            'framework_alignment_quality': {},
            'overall_confidence': 0.0
        }
        
        confidence_scores = []
        
        for field_name, analysis in field_analyses.items():
            if 'error' in analysis:
                continue
            
            uncertainty = analysis.get('uncertainty_measures', {}).get('total_uncertainty', 0.0)
            conflicts = analysis.get('conflicts', [])
            
            # Classify field consensus quality
            if uncertainty < 0.5 and len(conflicts) == 0:
                summary['fields_with_strong_consensus'].append(field_name)
                confidence_scores.append(1.0 - uncertainty)
            elif uncertainty > 2.0 or len(conflicts) > 2:
                summary['fields_with_conflicts'].append({
                    'field_name': field_name,
                    'uncertainty': uncertainty,
                    'conflict_count': len(conflicts)
                })
                confidence_scores.append(max(0.0, 1.0 - uncertainty))
            else:
                confidence_scores.append(max(0.0, 1.0 - uncertainty / 2.0))
            
            # Extract field type consensus
            recommendations = analysis.get('consensus_recommendations', {})
            field_properties = recommendations.get('consensus_field_properties', {})
            
            if 'type' in field_properties:
                field_type = field_properties['type']['value']
                summary['common_field_types'][field_type] += 1
        
        # Calculate overall confidence
        if confidence_scores:
            summary['overall_confidence'] = np.mean(confidence_scores)
        
        # Assess framework alignment quality
        for framework in ['OCSF', 'ECS', 'OSSEM']:
            aligned_fields = 0
            total_fields = 0
            
            for field_name, analysis in field_analyses.items():
                if 'error' in analysis:
                    continue
                
                total_fields += 1
                recommendations = analysis.get('consensus_recommendations', {})
                field_properties = recommendations.get('consensus_field_properties', {})
                framework_mappings = field_properties.get('framework_mappings', {})
                
                if framework in framework_mappings and framework_mappings[framework]['confidence'] > 0.5:
                    aligned_fields += 1
            
            if total_fields > 0:
                summary['framework_alignment_quality'][framework] = aligned_fields / total_fields
        
        return summary
    
    def consensus_uncertainty_analysis(self, 
                                     consensus_results: Dict[str, Any],
                                     centralized_embeddings: Dict[str, np.ndarray] = None,
                                     ted_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform Dempster-Shafer uncertainty analysis on consensus results
        
        Args:
            consensus_results: Results from consensus processing
            centralized_embeddings: Pre-computed embeddings (optional)
            ted_analysis: TED analysis results for additional evidence (optional)
            
        Returns:
            Dempster-Shafer uncertainty analysis results
        """
        try:
            final_consensus = consensus_results.get('final_consensus', {})
            
            if not final_consensus:
                return {'error': 'No consensus data available'}
            
            # Prepare evidence from consensus results
            evidence_list = []
            
            for cluster_id, consensus_data in final_consensus.items():
                content = consensus_data.get('consensus_content', '')
                base_confidence = consensus_data.get('consensus_confidence', 0.5)
                
                # Enhance confidence with TED structural information if available
                if ted_analysis and ted_analysis.get('structural_analysis'):
                    ted_boost = ted_analysis['structural_analysis'].get('average_similarity', 0.0) * 0.1
                    enhanced_confidence = min(1.0, base_confidence + ted_boost)
                else:
                    enhanced_confidence = base_confidence
                
                # Create evidence for this cluster
                evidence = Evidence(
                    source=f"cluster_{cluster_id}",
                    hypothesis_set=frozenset([str(hash(str(content)))]),  # Convert content to hypothesis
                    belief_mass=enhanced_confidence,
                    confidence=enhanced_confidence,
                    metadata={
                        'cluster_id': cluster_id,
                        'original_confidence': base_confidence,
                        'ted_enhanced': ted_analysis is not None,
                        'content_hash': str(hash(str(content))),
                        'method': consensus_data.get('method', 'unknown')
                    }
                )
                
                evidence_list.append(evidence)
            
            if not evidence_list:
                return {'error': 'No valid evidence extracted from consensus'}
            
            # Update frame of discernment with consensus hypotheses
            self.frame_of_discernment = set()
            for evidence in evidence_list:
                self.frame_of_discernment.update(evidence.hypothesis_set)
            
            # Compute basic probability assignment
            mass_function = self.compute_basic_probability_assignment(evidence_list)
            
            # Combine evidence using Dempster's rule
            if len(evidence_list) > 1:
                combined_mass = self._combine_multiple_sources(evidence_list)
            else:
                combined_mass = mass_function
            
            # Compute belief and plausibility functions
            belief_function = self.compute_belief_function(combined_mass)
            plausibility_function = self.compute_plausibility_function(combined_mass)
            
            # Compute uncertainty measures
            uncertainty_measures = self.compute_uncertainty_measures(
                combined_mass, belief_function, plausibility_function
            )
            
            # Calculate overall confidence metrics
            total_belief = sum(belief_function.values())
            total_uncertainty = uncertainty_measures.get('total_uncertainty', 0.5)
            conflict_mass = uncertainty_measures.get('conflict_mass', 0.0)
            
            overall_confidence = total_belief * (1.0 - conflict_mass)
            calibration_score = 1.0 - total_uncertainty
            
            logger.info(f"DS consensus analysis: confidence={overall_confidence:.3f}, "
                       f"calibration={calibration_score:.3f}, conflict={conflict_mass:.3f}")
            
            return {
                'overall_confidence': overall_confidence,
                'calibration_score': calibration_score,
                'uncertainty_mass': total_uncertainty,
                'conflict_mass': conflict_mass,
                'evidence_count': len(evidence_list),
                'frame_size': len(self.frame_of_discernment),
                'detailed_results': {
                    'mass_function': self._serialize_frozen_sets(combined_mass),
                    'belief_function': self._serialize_frozen_sets(belief_function),
                    'plausibility_function': self._serialize_frozen_sets(plausibility_function),
                    'uncertainty_measures': uncertainty_measures
                },
                'method': 'Dempster_Shafer_Consensus'
            }
            
        except Exception as e:
            logger.error(f"Dempster-Shafer consensus uncertainty analysis failed: {e}")
            return {
                'error': str(e),
                'overall_confidence': 0.5,
                'calibration_score': 0.0
            }

def run_dempster_shafer(input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Standardized API entry point for Dempster-Shafer uncertainty quantification
    
    Args:
        input_data: Dictionary containing provider_data or sections
        config: Optional configuration parameters
        
    Returns:
        Dictionary with DS analysis results
    """
    try:
        ds_engine = DempsterShaferEngine()
        
        # Extract provider data
        provider_data = input_data.get('provider_data', {})
        if not provider_data:
            return {
                'success': False,
                'error': 'No provider_data found in input',
                'ds_results': {}
            }
        
        # Run multi-field consensus analysis
        results = ds_engine.multi_field_consensus_analysis(provider_data)
        
        if 'error' in results:
            return {
                'success': False,
                'error': results['error'],
                'ds_results': {}
            }
        
        return {
            'success': True,
            'ds_results': results,
            'metadata': {
                'total_fields_analyzed': results.get('total_fields', 0),
                'overall_confidence': results.get('consensus_summary', {}).get('overall_confidence', 0.0),
                'strong_consensus_fields': len(results.get('consensus_summary', {}).get('fields_with_strong_consensus', [])),
                'conflicted_fields': len(results.get('consensus_summary', {}).get('fields_with_conflicts', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Dempster-Shafer analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'ds_results': {}
        }