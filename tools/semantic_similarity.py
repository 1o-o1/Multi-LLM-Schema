"""
Semantic Similarity Service for Multi-Agent Schema Consensus
Advanced similarity metrics with cross-model comparison and deduplication
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import re
import difflib
from scipy.spatial.distance import cosine, jaccard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import editdistance
import json

logger = logging.getLogger(__name__)

class SemanticSimilarity:
    """Advanced similarity metrics and cross-model comparison"""
    
    def __init__(self):
        """Initialize semantic similarity service"""
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.field_cache = {}
        
    def cosine_similarity_text(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two text strings using TF-IDF
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            # Vectorize texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def jaccard_similarity(self, text1: str, text2: str, tokenize: bool = True) -> float:
        """
        Calculate Jaccard similarity between two texts
        
        Args:
            text1: First text string
            text2: Second text string
            tokenize: Whether to tokenize texts into words
            
        Returns:
            Jaccard similarity score (0-1)
        """
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            if tokenize:
                # Tokenize and create sets
                tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
                tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
            else:
                # Character-level Jaccard
                tokens1 = set(text1.lower())
                tokens2 = set(text2.lower())
            
            if not tokens1 and not tokens2:
                return 1.0
            if not tokens1 or not tokens2:
                return 0.0
                
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            logger.error(f"Jaccard similarity calculation failed: {e}")
            return 0.0
    
    def edit_distance_similarity(self, text1: str, text2: str, normalize: bool = True) -> float:
        """
        Calculate edit distance similarity between two texts
        
        Args:
            text1: First text string
            text2: Second text string
            normalize: Whether to normalize by maximum string length
            
        Returns:
            Edit distance similarity score (0-1 if normalized)
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
            
        try:
            distance = editdistance.eval(text1.lower(), text2.lower())
            
            if normalize:
                max_len = max(len(text1), len(text2))
                if max_len == 0:
                    return 1.0
                return 1.0 - (distance / max_len)
            else:
                return distance
                
        except Exception as e:
            logger.error(f"Edit distance calculation failed: {e}")
            return 0.0 if normalize else float('inf')
    
    def sequence_matcher_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using Python's difflib SequenceMatcher
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity ratio (0-1)
        """
        try:
            matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
            return matcher.ratio()
            
        except Exception as e:
            logger.error(f"SequenceMatcher similarity calculation failed: {e}")
            return 0.0
    
    def field_similarity_comprehensive(self, 
                                     field1: Dict[str, Any], 
                                     field2: Dict[str, Any],
                                     weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive similarity between two field definitions
        
        Args:
            field1: First field definition
            field2: Second field definition
            weights: Weights for different similarity components
            
        Returns:
            Dictionary of similarity scores
        """
        if weights is None:
            weights = {
                'description': 0.4,
                'type': 0.2,
                'name': 0.2,
                'schema_mapping': 0.2
            }
        
        similarities = {}
        
        # Description similarity
        desc1 = field1.get('description', '')
        desc2 = field2.get('description', '')
        similarities['description'] = self.multi_metric_similarity(desc1, desc2)
        
        # Type similarity
        type1 = field1.get('type', '').lower()
        type2 = field2.get('type', '').lower()
        similarities['type'] = self._type_similarity(type1, type2)
        
        # Name similarity (if available)
        name1 = field1.get('name', field1.get('field_name', ''))
        name2 = field2.get('name', field2.get('field_name', ''))
        similarities['name'] = self.multi_metric_similarity(name1, name2)
        
        # Schema mapping similarity
        similarities['schema_mapping'] = self._schema_mapping_similarity(field1, field2)
        
        # Weighted average
        weighted_score = sum(
            similarities[key] * weights.get(key, 0.0) 
            for key in similarities.keys()
        )
        similarities['weighted_average'] = weighted_score
        
        return similarities
    
    def multi_metric_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using multiple metrics and return average
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Average similarity score across multiple metrics
        """
        if not text1.strip() and not text2.strip():
            return 1.0
        if not text1.strip() or not text2.strip():
            return 0.0
            
        try:
            metrics = [
                self.cosine_similarity_text(text1, text2),
                self.jaccard_similarity(text1, text2),
                self.edit_distance_similarity(text1, text2),
                self.sequence_matcher_similarity(text1, text2)
            ]
            
            return np.mean(metrics)
            
        except Exception as e:
            logger.error(f"Multi-metric similarity calculation failed: {e}")
            return 0.0
    
    def _type_similarity(self, type1: str, type2: str) -> float:
        """Calculate similarity between data types"""
        if not type1 or not type2:
            return 0.0
            
        if type1 == type2:
            return 1.0
        
        # Type compatibility groups
        type_groups = [
            {'string', 'str', 'text'},
            {'integer', 'int', 'number', 'numeric'},
            {'float', 'double', 'decimal', 'number', 'numeric'},
            {'datetime', 'timestamp', 'date', 'time'},
            {'boolean', 'bool'},
            {'array', 'list'},
            {'object', 'dict', 'map'},
            {'ip', 'ipv4', 'ipv6', 'address'}
        ]
        
        # Check if types are in the same group
        for group in type_groups:
            if type1 in group and type2 in group:
                return 0.8  # High similarity for compatible types
        
        # Fallback to string similarity
        return self.edit_distance_similarity(type1, type2)
    
    def _schema_mapping_similarity(self, field1: Dict, field2: Dict) -> float:
        """Calculate similarity based on schema mappings (OCSF, ECS, OSSEM)"""
        schemas = ['OCSF', 'ECS', 'OSSEM']
        mapping_similarities = []
        
        for schema in schemas:
            mapping1 = field1.get(schema, '')
            mapping2 = field2.get(schema, '')
            
            if mapping1 and mapping2 and mapping1 != 'null' and mapping2 != 'null':
                if mapping1 == mapping2:
                    mapping_similarities.append(1.0)
                else:
                    # Calculate similarity between mapping strings
                    sim = self.multi_metric_similarity(mapping1, mapping2)
                    mapping_similarities.append(sim)
        
        if not mapping_similarities:
            return 0.0
            
        return np.mean(mapping_similarities)
    
    def cross_model_result_comparison(self, 
                                    provider_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare and analyze results across different model providers
        
        Args:
            provider_results: Dictionary of provider results
            
        Returns:
            Cross-model comparison analysis
        """
        comparison_results = {
            'provider_similarities': {},
            'field_consensus': {},
            'conflict_analysis': {},
            'coverage_analysis': {}
        }
        
        try:
            providers = list(provider_results.keys())
            
            # Pairwise provider similarity
            for i, provider1 in enumerate(providers):
                for j, provider2 in enumerate(providers):
                    if i >= j:
                        continue
                        
                    sim_score = self._compare_provider_results(
                        provider_results[provider1],
                        provider_results[provider2]
                    )
                    comparison_results['provider_similarities'][f"{provider1}_vs_{provider2}"] = sim_score
            
            # Field consensus analysis
            comparison_results['field_consensus'] = self._analyze_field_consensus(provider_results)
            
            # Conflict analysis
            comparison_results['conflict_analysis'] = self._analyze_conflicts(provider_results)
            
            # Coverage analysis
            comparison_results['coverage_analysis'] = self._analyze_coverage(provider_results)
            
            logger.info(f"Cross-model comparison completed for {len(providers)} providers")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Cross-model comparison failed: {e}")
            return comparison_results
    
    def _compare_provider_results(self, result1: Dict, result2: Dict) -> float:
        """Compare results from two providers"""
        try:
            # Extract fields
            fields1 = result1.get('parsed_json', {}).get('fields', {})
            fields2 = result2.get('parsed_json', {}).get('fields', {})
            
            if not fields1 or not fields2:
                return 0.0
            
            # Calculate field-level similarities
            similarities = []
            
            # Find best matches between fields
            for field1_name, field1_data in fields1.items():
                best_similarity = 0.0
                
                for field2_name, field2_data in fields2.items():
                    field_sim = self.field_similarity_comprehensive(field1_data, field2_data)
                    sim_score = field_sim.get('weighted_average', 0.0)
                    best_similarity = max(best_similarity, sim_score)
                
                similarities.append(best_similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Provider result comparison failed: {e}")
            return 0.0
    
    def _analyze_field_consensus(self, provider_results: Dict) -> Dict[str, Any]:
        """Analyze consensus across providers for each field"""
        field_consensus = {}
        all_field_names = set()
        
        # Collect all unique field names
        for result in provider_results.values():
            fields = result.get('parsed_json', {}).get('fields', {})
            all_field_names.update(fields.keys())
        
        # Analyze consensus for each field
        for field_name in all_field_names:
            field_data_list = []
            providers_with_field = []
            
            for provider, result in provider_results.items():
                fields = result.get('parsed_json', {}).get('fields', {})
                if field_name in fields:
                    field_data_list.append(fields[field_name])
                    providers_with_field.append(provider)
            
            consensus_score = len(providers_with_field) / len(provider_results)
            
            # Calculate agreement on field properties
            property_agreement = {}
            if field_data_list:
                for prop in ['type', 'description', 'OCSF', 'ECS', 'OSSEM']:
                    values = [fd.get(prop, '') for fd in field_data_list]
                    unique_values = set(v for v in values if v and v != 'null')
                    
                    if len(unique_values) <= 1:
                        property_agreement[prop] = 1.0
                    else:
                        # Calculate average similarity between all pairs
                        similarities = []
                        for i, val1 in enumerate(values):
                            for j, val2 in enumerate(values):
                                if i >= j:
                                    continue
                                if val1 and val2:
                                    sim = self.multi_metric_similarity(str(val1), str(val2))
                                    similarities.append(sim)
                        
                        property_agreement[prop] = np.mean(similarities) if similarities else 0.0
            
            field_consensus[field_name] = {
                'presence_consensus': consensus_score,
                'providers_with_field': providers_with_field,
                'property_agreement': property_agreement,
                'average_agreement': np.mean(list(property_agreement.values())) if property_agreement else 0.0
            }
        
        return field_consensus
    
    def _analyze_conflicts(self, provider_results: Dict) -> Dict[str, List]:
        """Analyze conflicts between provider results"""
        conflicts = {
            'type_conflicts': [],
            'description_conflicts': [],
            'schema_conflicts': []
        }
        
        # Get field consensus data
        field_consensus = self._analyze_field_consensus(provider_results)
        
        for field_name, consensus_data in field_consensus.items():
            if consensus_data['presence_consensus'] < 1.0:
                continue  # Skip fields not present in all providers
            
            property_agreement = consensus_data.get('property_agreement', {})
            
            # Identify conflicts (low agreement scores)
            conflict_threshold = 0.5
            
            for prop, agreement in property_agreement.items():
                if agreement < conflict_threshold:
                    conflict_info = {
                        'field_name': field_name,
                        'property': prop,
                        'agreement_score': agreement,
                        'providers': consensus_data['providers_with_field']
                    }
                    
                    if prop == 'type':
                        conflicts['type_conflicts'].append(conflict_info)
                    elif prop == 'description':
                        conflicts['description_conflicts'].append(conflict_info)
                    else:
                        conflicts['schema_conflicts'].append(conflict_info)
        
        return conflicts
    
    def _analyze_coverage(self, provider_results: Dict) -> Dict[str, Any]:
        """Analyze field coverage across providers"""
        all_fields = set()
        provider_field_counts = {}
        
        # Collect all fields and counts
        for provider, result in provider_results.items():
            fields = result.get('parsed_json', {}).get('fields', {})
            provider_field_counts[provider] = len(fields)
            all_fields.update(fields.keys())
        
        # Calculate coverage metrics
        total_unique_fields = len(all_fields)
        
        coverage_analysis = {
            'total_unique_fields': total_unique_fields,
            'provider_field_counts': provider_field_counts,
            'average_fields_per_provider': np.mean(list(provider_field_counts.values())),
            'field_coverage_by_provider': {}
        }
        
        # Field coverage by provider
        for provider in provider_results.keys():
            fields = set(provider_results[provider].get('parsed_json', {}).get('fields', {}).keys())
            coverage_percentage = len(fields) / total_unique_fields if total_unique_fields > 0 else 0.0
            coverage_analysis['field_coverage_by_provider'][provider] = coverage_percentage
        
        return coverage_analysis
    
    def threshold_based_similarity_clustering(self, 
                                            similarity_matrix: np.ndarray,
                                            threshold: float = 0.7) -> List[List[int]]:
        """
        Perform threshold-based similarity clustering
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters (each cluster is a list of indices)
        """
        n_items = similarity_matrix.shape[0]
        visited = set()
        clusters = []
        
        for i in range(n_items):
            if i in visited:
                continue
            
            # Start new cluster
            cluster = [i]
            visited.add(i)
            
            # Find all items similar to item i
            for j in range(n_items):
                if j != i and j not in visited and similarity_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited.add(j)
            
            clusters.append(cluster)
        
        logger.info(f"Threshold-based clustering created {len(clusters)} clusters from {n_items} items")
        return clusters
    
    def multi_dimensional_similarity_analysis(self, 
                                            data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform multi-dimensional similarity analysis
        
        Args:
            data_points: List of data points to analyze
            
        Returns:
            Multi-dimensional similarity analysis results
        """
        if len(data_points) < 2:
            return {}
        
        try:
            analysis_results = {
                'similarity_dimensions': {},
                'cluster_quality': {},
                'dimensional_correlations': {}
            }
            
            # Extract different dimensions
            dimensions = ['description', 'type', 'importance', 'schema_mappings']
            similarity_matrices = {}
            
            for dim in dimensions:
                # Extract dimension data
                dim_data = []
                for point in data_points:
                    if dim == 'schema_mappings':
                        # Combine schema mappings
                        mappings = []
                        for schema in ['OCSF', 'ECS', 'OSSEM']:
                            mapping = point.get(schema, '')
                            if mapping and mapping != 'null':
                                mappings.append(mapping)
                        dim_data.append(' '.join(mappings))
                    else:
                        dim_data.append(str(point.get(dim, '')))
                
                # Calculate similarity matrix for this dimension
                n_points = len(dim_data)
                sim_matrix = np.zeros((n_points, n_points))
                
                for i in range(n_points):
                    for j in range(n_points):
                        if i == j:
                            sim_matrix[i, j] = 1.0
                        else:
                            sim_matrix[i, j] = self.multi_metric_similarity(dim_data[i], dim_data[j])
                
                similarity_matrices[dim] = sim_matrix
                analysis_results['similarity_dimensions'][dim] = {
                    'average_similarity': float(np.mean(sim_matrix)),
                    'std_similarity': float(np.std(sim_matrix)),
                    'max_similarity': float(np.max(sim_matrix[np.triu_indices(n_points, k=1)])),
                    'min_similarity': float(np.min(sim_matrix[np.triu_indices(n_points, k=1)]))
                }
            
            # Calculate correlations between dimensions
            dim_names = list(similarity_matrices.keys())
            for i, dim1 in enumerate(dim_names):
                for j, dim2 in enumerate(dim_names):
                    if i >= j:
                        continue
                    
                    # Flatten upper triangular matrices
                    n = similarity_matrices[dim1].shape[0]
                    indices = np.triu_indices(n, k=1)
                    
                    vec1 = similarity_matrices[dim1][indices]
                    vec2 = similarity_matrices[dim2][indices]
                    
                    if len(vec1) > 1:
                        correlation = np.corrcoef(vec1, vec2)[0, 1]
                        analysis_results['dimensional_correlations'][f"{dim1}_vs_{dim2}"] = float(correlation)
            
            logger.info(f"Multi-dimensional similarity analysis completed for {len(data_points)} data points")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Multi-dimensional similarity analysis failed: {e}")
            return {}
    
    def calculate_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two security patterns
        
        Args:
            pattern1: First security pattern dictionary
            pattern2: Second security pattern dictionary
            
        Returns:
            Similarity score (0-1)
        """
        try:
            if not isinstance(pattern1, dict) or not isinstance(pattern2, dict):
                return 0.0
            
            total_similarity = 0.0
            weight_sum = 0.0
            
            # Compare pattern names (high weight)
            name_fields = ['pattern_name', 'name', 'title']
            for field in name_fields:
                if field in pattern1 and field in pattern2:
                    name_sim = self.cosine_similarity_text(
                        str(pattern1[field]), str(pattern2[field])
                    )
                    total_similarity += name_sim * 0.4
                    weight_sum += 0.4
                    break
            
            # Compare instructions/descriptions (medium weight)
            desc_fields = ['instruction', 'description', 'desc']
            for field in desc_fields:
                if field in pattern1 and field in pattern2:
                    desc_sim = self.cosine_similarity_text(
                        str(pattern1[field]), str(pattern2[field])
                    )
                    total_similarity += desc_sim * 0.3
                    weight_sum += 0.3
                    break
            
            # Compare identifiable fields (medium weight)
            if 'identifiable_fields' in pattern1 and 'identifiable_fields' in pattern2:
                fields1 = set(pattern1['identifiable_fields']) if isinstance(pattern1['identifiable_fields'], list) else set()
                fields2 = set(pattern2['identifiable_fields']) if isinstance(pattern2['identifiable_fields'], list) else set()
                
                if fields1 and fields2:
                    field_jaccard = len(fields1 & fields2) / len(fields1 | fields2)
                    total_similarity += field_jaccard * 0.2
                    weight_sum += 0.2
            
            # Compare overall structure similarity (low weight)
            pattern1_keys = set(pattern1.keys())
            pattern2_keys = set(pattern2.keys())
            if pattern1_keys and pattern2_keys:
                structure_jaccard = len(pattern1_keys & pattern2_keys) / len(pattern1_keys | pattern2_keys)
                total_similarity += structure_jaccard * 0.1
                weight_sum += 0.1
            
            # Normalize by total weight
            if weight_sum > 0:
                return total_similarity / weight_sum
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Pattern similarity calculation failed: {e}")
            return 0.0