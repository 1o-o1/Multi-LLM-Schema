# preprocessing.py
"""
🔄 FEW-SHOT MULTI-LLM PIPELINE SYSTEM - Data Preprocessing
Log preprocessing and data preparation functionality

This file is part of System 1: Few-Shot Multi-LLM Pipeline
Handles raw log data cleaning and preparation for analysis
See SYSTEM_ARCHITECTURE.md for complete system documentation
"""
import re
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import time
from collections import defaultdict
from config import MAX_LOG_CHARS, MAX_SAMPLE_LINES
from utils import get_nested_value, calculate_data_hash

logger = logging.getLogger(__name__)

@dataclass
class ProcessedLogData:
    """Simplified processed log data"""
    raw_data: str
    cleaned_data: str
    sample_lines: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ConsensusPreparationResult:
    """Result of consensus data preparation"""
    prepared_data: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    validation_report: Dict[str, Any]
    preprocessing_metadata: Dict[str, Any] = field(default_factory=dict)

class LogPreprocessor:
    """Minimal log preprocessor - just clean and sample"""
    
    def clean_log_data(self, log_data: str) -> str:
        """Basic cleaning only"""
        # Remove ANSI color codes
        log_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', log_data)
        
        # Normalize line endings
        log_data = log_data.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace
        log_data = re.sub(r' {2,}', ' ', log_data)
        log_data = re.sub(r'\n{3,}', '\n\n', log_data)
        
        return log_data.strip()
    
    def sample_lines(self, log_data: str, max_lines: int = MAX_SAMPLE_LINES) -> List[str]:
        """Get a representative sample of log lines"""
        lines = log_data.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) <= max_lines:
            return lines
        
        # Take some from beginning, middle, and end
        sample_size = max_lines // 3
        sample = []
        sample.extend(lines[:sample_size])
        
        mid_start = len(lines) // 2 - sample_size // 2
        sample.extend(lines[mid_start:mid_start + sample_size])
        
        sample.extend(lines[-sample_size:])
        
        return sample
    
    def process(self, log_data: str, max_chars: int = MAX_LOG_CHARS) -> ProcessedLogData:
        """Minimal processing - just clean and sample"""
        logger.info(f"Processing {len(log_data)} characters of log data")
        
        # Clean the log data
        cleaned_data = self.clean_log_data(log_data)
        
        # Truncate if too long
        if len(cleaned_data) > max_chars:
            cleaned_data = cleaned_data[:max_chars] + "\n... [truncated]"
            logger.info(f"Truncated log data to {max_chars} characters")
        
        # Get sample lines
        sample_lines = self.sample_lines(cleaned_data)
        
        result = ProcessedLogData(
            raw_data=log_data[:1000],  # Keep small sample of raw
            cleaned_data=cleaned_data,
            sample_lines=sample_lines,
            metadata={
                "line_count": len(cleaned_data.split('\n')),
                "char_count": len(cleaned_data),
                "processing_timestamp": time.time()
            }
        )
        
        logger.info(f"Processed log data: {result.metadata['line_count']} lines, {result.metadata['char_count']} chars")
        
        return result

class ConsensusDataPreprocessor:
    """Preprocessor for consensus data preparation"""
    
    def __init__(self):
        self.processed_cache = {}
    
    def prepare_provider_results(self, 
                               provider_results: Dict[str, Dict],
                               target_key_path: str) -> Dict[str, Any]:
        """
        Prepare provider results for consensus processing
        
        Args:
            provider_results: Raw provider results
            target_key_path: Path to target data for consensus
            
        Returns:
            Prepared data structure for consensus algorithms
        """
        prepared_data = {
            'providers': list(provider_results.keys()),
            'target_key_path': target_key_path,
            'extracted_data': {},
            'data_statistics': {},
            'preprocessing_metadata': {
                'timestamp': time.time(),
                'provider_count': len(provider_results)
            }
        }
        
        # Extract target data from each provider
        for provider, result in provider_results.items():
            target_data = get_nested_value(result, target_key_path)
            
            if target_data is not None:
                prepared_data['extracted_data'][provider] = target_data
                
                # Calculate statistics
                stats = self._calculate_data_statistics(target_data)
                prepared_data['data_statistics'][provider] = stats
        
        # Calculate overall statistics
        prepared_data['overall_statistics'] = self._calculate_overall_statistics(
            prepared_data['data_statistics']
        )
        
        logger.info(f"Prepared consensus data for {len(prepared_data['extracted_data'])} providers")
        
        return prepared_data
    
    def _calculate_data_statistics(self, data: Any) -> Dict[str, Any]:
        """Calculate statistics for data structure"""
        stats = {
            'data_type': type(data).__name__,
            'data_hash': calculate_data_hash(data),
            'size_estimate': len(str(data))
        }
        
        if isinstance(data, dict):
            stats.update({
                'key_count': len(data),
                'keys': list(data.keys()),
                'nested_levels': self._calculate_nesting_depth(data)
            })
        elif isinstance(data, list):
            stats.update({
                'item_count': len(data),
                'item_types': list(set(type(item).__name__ for item in data)),
                'nested_levels': max([self._calculate_nesting_depth(item) for item in data], default=0)
            })
        elif isinstance(data, str):
            stats.update({
                'char_count': len(data),
                'word_count': len(data.split()),
                'line_count': data.count('\n') + 1
            })
        
        return stats
    
    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of data structure"""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                self._calculate_nesting_depth(value, current_depth + 1) 
                for value in data.values()
            )
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(
                self._calculate_nesting_depth(item, current_depth + 1) 
                for item in data
            )
        else:
            return current_depth
    
    def _calculate_overall_statistics(self, provider_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall statistics across providers"""
        if not provider_stats:
            return {}
        
        overall = {
            'provider_count': len(provider_stats),
            'data_types': defaultdict(int),
            'size_range': {'min': float('inf'), 'max': 0, 'avg': 0},
            'common_keys': set(),
            'unique_hashes': set()
        }
        
        sizes = []
        all_keys = []
        
        for provider, stats in provider_stats.items():
            # Data types
            overall['data_types'][stats['data_type']] += 1
            
            # Sizes
            size = stats.get('size_estimate', 0)
            sizes.append(size)
            overall['size_range']['min'] = min(overall['size_range']['min'], size)
            overall['size_range']['max'] = max(overall['size_range']['max'], size)
            
            # Keys (for dict data)
            if 'keys' in stats:
                all_keys.extend(stats['keys'])
            
            # Hashes
            overall['unique_hashes'].add(stats['data_hash'])
        
        # Calculate averages
        if sizes:
            overall['size_range']['avg'] = sum(sizes) / len(sizes)
        
        # Find common keys
        if all_keys:
            key_counts = defaultdict(int)
            for key in all_keys:
                key_counts[key] += 1
            
            threshold = len(provider_stats) * 0.5  # Present in at least 50% of providers
            overall['common_keys'] = set(
                key for key, count in key_counts.items() if count >= threshold
            )
        
        # Convert sets to lists for JSON serialization
        overall['common_keys'] = list(overall['common_keys'])
        overall['unique_hashes'] = list(overall['unique_hashes'])
        overall['data_types'] = dict(overall['data_types'])
        
        return overall
    
    def normalize_consensus_targets(self, 
                                  provider_results: Dict[str, Dict],
                                  target_key_path: str) -> Dict[str, List[Dict]]:
        """
        Normalize different data structures for consensus processing
        
        Args:
            provider_results: Provider results
            target_key_path: Path to target data
            
        Returns:
            Normalized data as list of items from each provider
        """
        normalized = {}
        
        for provider, result in provider_results.items():
            target_data = get_nested_value(result, target_key_path)
            
            if target_data is None:
                normalized[provider] = []
                continue
            
            # Normalize to list of items
            if isinstance(target_data, dict):
                # Convert dict to list of key-value items
                items = []
                for key, value in target_data.items():
                    if isinstance(value, dict):
                        item = value.copy()
                        item['_key'] = key
                    else:
                        item = {'_key': key, '_value': value}
                    items.append(item)
                normalized[provider] = items
                
            elif isinstance(target_data, list):
                # Already a list
                items = []
                for i, item in enumerate(target_data):
                    if isinstance(item, dict):
                        items.append(item)
                    else:
                        items.append({'_index': i, '_value': item})
                normalized[provider] = items
                
            else:
                # Single value - wrap in item
                normalized[provider] = [{'_value': target_data}]
        
        logger.info(f"Normalized data from {len(provider_results)} providers")
        
        return normalized
    
    def extract_consensus_candidates(self, 
                                   normalized_data: Dict[str, List[Dict]],
                                   similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Extract candidate items for consensus based on similarity
        
        Args:
            normalized_data: Normalized provider data
            similarity_threshold: Threshold for considering items similar
            
        Returns:
            List of consensus candidates with metadata
        """
        candidates = []
        
        # Collect all items with provider information
        all_items = []
        for provider, items in normalized_data.items():
            for item in items:
                all_items.append({
                    'provider': provider,
                    'data': item,
                    'hash': calculate_data_hash(item)
                })
        
        # Group items by hash (exact matches)
        hash_groups = defaultdict(list)
        for item in all_items:
            hash_groups[item['hash']].append(item)
        
        # Create candidates from groups with multiple providers
        for item_hash, group_items in hash_groups.items():
            if len(group_items) > 1:
                providers = [item['provider'] for item in group_items]
                
                candidate = {
                    'type': 'exact_match',
                    'item_hash': item_hash,
                    'providers': list(set(providers)),
                    'provider_count': len(set(providers)),
                    'data': group_items[0]['data'],  # They're identical, so take first
                    'confidence': 1.0  # Exact match has highest confidence
                }
                candidates.append(candidate)
        
        # For remaining items, look for semantic similarities
        # This would require more sophisticated similarity matching
        # For now, we'll identify potential candidates based on key overlap
        
        unmatched_items = []
        for item in all_items:
            if item['hash'] not in [c['item_hash'] for c in candidates]:
                unmatched_items.append(item)
        
        # Simple key-based similarity for dict items
        for i, item1 in enumerate(unmatched_items):
            if not isinstance(item1['data'], dict):
                continue
                
            similar_items = [item1]
            
            for j, item2 in enumerate(unmatched_items[i+1:], i+1):
                if not isinstance(item2['data'], dict):
                    continue
                    
                # Calculate key overlap
                keys1 = set(item1['data'].keys())
                keys2 = set(item2['data'].keys())
                
                if keys1 and keys2:
                    overlap = len(keys1 & keys2) / len(keys1 | keys2)
                    
                    if overlap >= similarity_threshold:
                        similar_items.append(item2)
            
            if len(similar_items) > 1:
                providers = [item['provider'] for item in similar_items]
                
                # Create merged data (simple merge)
                merged_data = {}
                for item in similar_items:
                    merged_data.update(item['data'])
                
                candidate = {
                    'type': 'semantic_match',
                    'item_hash': calculate_data_hash(merged_data),
                    'providers': list(set(providers)),
                    'provider_count': len(set(providers)),
                    'data': merged_data,
                    'confidence': overlap,
                    'source_items': [item['data'] for item in similar_items]
                }
                candidates.append(candidate)
                
                # Remove matched items from unmatched list
                for item in similar_items:
                    if item in unmatched_items:
                        unmatched_items.remove(item)
        
        logger.info(f"Extracted {len(candidates)} consensus candidates")
        
        return candidates
    
    def prepare_for_algorithm(self, 
                            consensus_candidates: List[Dict[str, Any]],
                            algorithm: str) -> Dict[str, Any]:
        """
        Prepare consensus candidates for specific algorithm
        
        Args:
            consensus_candidates: List of consensus candidates
            algorithm: Target algorithm name
            
        Returns:
            Algorithm-specific data format
        """
        if algorithm == 'embedding_similarity':
            # Prepare text data for embedding
            texts = []
            metadata = []
            
            for candidate in consensus_candidates:
                # Convert data to text representation
                if isinstance(candidate['data'], dict):
                    text_parts = []
                    for key, value in candidate['data'].items():
                        text_parts.append(f"{key}: {value}")
                    text = " | ".join(text_parts)
                else:
                    text = str(candidate['data'])
                
                texts.append(text)
                metadata.append({
                    'candidate_hash': candidate['item_hash'],
                    'providers': candidate['providers'],
                    'confidence': candidate['confidence']
                })
            
            return {'texts': texts, 'metadata': metadata}
            
        elif algorithm in ['bft_consensus', 'dempster_shafer']:
            # Prepare as provider proposals
            provider_proposals = defaultdict(dict)
            
            for i, candidate in enumerate(consensus_candidates):
                for provider in candidate['providers']:
                    provider_proposals[provider][f"item_{i}"] = candidate['data']
            
            return dict(provider_proposals)
            
        else:
            # Generic format - return as-is
            return {'candidates': consensus_candidates}
    
    def create_consensus_preparation_result(self, 
                                          provider_results: Dict[str, Dict],
                                          target_key_path: str,
                                          enable_validation: bool = True) -> ConsensusPreparationResult:
        """
        Complete consensus preparation pipeline
        
        Args:
            provider_results: Raw provider results
            target_key_path: Path to target data for consensus
            enable_validation: Whether to run validation
            
        Returns:
            Complete consensus preparation result
        """
        try:
            # Prepare provider results
            prepared_data = self.prepare_provider_results(provider_results, target_key_path)
            
            # Normalize data
            normalized_data = self.normalize_consensus_targets(provider_results, target_key_path)
            
            # Extract candidates
            candidates = self.extract_consensus_candidates(normalized_data)
            
            # Validate if enabled
            validation_report = {}
            if enable_validation:
                validation_report = self.validate_preprocessing_result(prepared_data)
            
            # Create metadata
            preprocessing_metadata = {
                'normalization_summary': {
                    'total_providers': len(provider_results),
                    'providers_with_data': len(normalized_data),
                    'total_candidates': len(candidates),
                    'exact_matches': len([c for c in candidates if c['type'] == 'exact_match']),
                    'semantic_matches': len([c for c in candidates if c['type'] == 'semantic_match'])
                },
                'target_analysis': prepared_data.get('overall_statistics', {}),
                'processing_timestamp': time.time()
            }
            
            logger.info(f"Consensus preparation complete: {len(candidates)} candidates from {len(provider_results)} providers")
            
            return ConsensusPreparationResult(
                prepared_data=prepared_data,
                candidates=candidates,
                validation_report=validation_report,
                preprocessing_metadata=preprocessing_metadata
            )
            
        except Exception as e:
            logger.error(f"Consensus preparation failed: {e}")
            return ConsensusPreparationResult(
                prepared_data={},
                candidates=[],
                validation_report={'valid': False, 'errors': [str(e)]},
                preprocessing_metadata={'error': str(e), 'timestamp': time.time()}
            )
    
    def batch_prepare_consensus_data(self, 
                                   provider_results: Dict[str, Dict],
                                   target_paths: List[str]) -> Dict[str, ConsensusPreparationResult]:
        """
        Prepare consensus data for multiple target paths in batch
        
        Args:
            provider_results: Provider results
            target_paths: List of target paths to process
            
        Returns:
            Dictionary mapping target paths to preparation results
        """
        batch_results = {}
        
        for target_path in target_paths:
            logger.info(f"Preparing consensus data for: {target_path}")
            
            preparation_result = self.create_consensus_preparation_result(
                provider_results, target_path
            )
            
            batch_results[target_path] = preparation_result
        
        logger.info(f"Batch preparation complete for {len(target_paths)} target paths")
        return batch_results
    
    def extract_pattern_specific_candidates(self, 
                                          provider_results: Dict[str, Dict],
                                          pattern_types: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Extract candidates specifically for malicious pattern consensus
        
        Args:
            provider_results: Provider results containing patterns
            pattern_types: Types of patterns to extract
            
        Returns:
            Dictionary mapping pattern types to candidate lists
        """
        if pattern_types is None:
            pattern_types = ['malicious_patterns', 'behavioral_patterns', 'attack_patterns', 'threat_patterns']
        
        pattern_candidates = {}
        
        for pattern_type in pattern_types:
            # Try multiple possible paths for each pattern type
            possible_paths = [
                f'parsed_json.{pattern_type}',
                f'parsed_json.patterns.{pattern_type}',
                f'parsed_json.analysis.{pattern_type}',
                f'{pattern_type}'
            ]
            
            all_patterns = []
            
            for provider, result in provider_results.items():
                for path in possible_paths:
                    pattern_data = get_nested_value(result, path)
                    
                    if pattern_data:
                        if isinstance(pattern_data, list):
                            for pattern in pattern_data:
                                if isinstance(pattern, dict):
                                    pattern['_source_provider'] = provider
                                    pattern['_source_path'] = path
                                all_patterns.append(pattern)
                        elif isinstance(pattern_data, dict):
                            pattern_data['_source_provider'] = provider
                            pattern_data['_source_path'] = path
                            all_patterns.append(pattern_data)
                        break  # Found data at this path, don't check others
            
            if all_patterns:
                # Convert to consensus candidate format
                candidates = []
                for i, pattern in enumerate(all_patterns):
                    candidate = {
                        'type': 'pattern_item',
                        'item_hash': calculate_data_hash(pattern),
                        'providers': [pattern.get('_source_provider', 'unknown')],
                        'provider_count': 1,
                        'data': pattern,
                        'confidence': 1.0,
                        'pattern_type': pattern_type
                    }
                    candidates.append(candidate)
                
                pattern_candidates[pattern_type] = candidates
        
        logger.info(f"Extracted pattern candidates: {[(pt, len(candidates)) for pt, candidates in pattern_candidates.items()]}")
        return pattern_candidates
    
    def validate_preprocessing_result(self, 
                                    prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate preprocessing results
        
        Args:
            prepared_data: Preprocessed data
            
        Returns:
            Validation report
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check required fields
            required_fields = ['providers', 'target_key_path', 'extracted_data']
            
            for field in required_fields:
                if field not in prepared_data:
                    validation['errors'].append(f"Missing required field: {field}")
                    validation['valid'] = False
            
            # Check data quality
            extracted_data = prepared_data.get('extracted_data', {})
            
            if not extracted_data:
                validation['errors'].append("No data extracted from any provider")
                validation['valid'] = False
            else:
                validation['statistics']['providers_with_data'] = len(extracted_data)
                validation['statistics']['total_providers'] = len(prepared_data.get('providers', []))
                
                coverage = validation['statistics']['providers_with_data'] / max(1, validation['statistics']['total_providers'])
                
                if coverage < 0.5:
                    validation['warnings'].append(f"Low data coverage: {coverage:.1%}")
            
            # Check for data diversity
            overall_stats = prepared_data.get('overall_statistics', {})
            unique_hashes = overall_stats.get('unique_hashes', [])
            
            if len(unique_hashes) == 1:
                validation['warnings'].append("All providers have identical data - limited consensus value")
            elif len(unique_hashes) == len(extracted_data):
                validation['warnings'].append("All providers have unique data - consensus may be difficult")
            
            validation['statistics']['unique_data_variants'] = len(unique_hashes)
            
            return validation
            
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Validation error: {str(e)}")
            return validation