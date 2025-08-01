# utils.py
"""
🔄 SHARED UTILITIES - Used by Both Systems
Utility functions for schema generation and consensus processing

This file is SHARED between:
- System 1: Few-Shot Multi-LLM Pipeline
- System 2: JSON Consensus Framework
See SYSTEM_ARCHITECTURE.md for complete system documentation
"""
import json
import re
import logging
import os
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text response with various fallbacks"""
    if not text:
        logger.error("Empty text provided for JSON extraction")
        return {}
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON object
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            result = json.loads(json_str)
            logger.debug(f"Successfully extracted JSON with {len(result)} keys")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Failed JSON string: {json_str[:200]}...")
            
            # Try to fix common issues
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            try:
                result = json.loads(json_str)
                logger.debug("Successfully parsed JSON after cleanup")
                return result
            except:
                logger.error("Failed to parse JSON even after cleanup")
                return {}
    
    logger.error("No JSON object found in text")
    return {}

def extract_json_array_from_text(text: str) -> list:
    """Extract JSON array from text response"""
    if not text:
        return []
    
    # Remove markdown
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON array
    array_match = re.search(r'(\[.*\])', text, re.DOTALL)
    if array_match:
        json_str = array_match.group(1)
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                result = json.loads(json_str)
                if isinstance(result, list):
                    return result
            except:
                pass
    
    return []

def make_dirs(path: str) -> None:
    """Create directory structure for path"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    """Write object as JSON file"""
    make_dirs(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote JSON to {path}")

def write_text(path: str, text: str) -> None:
    """Write text to file"""
    make_dirs(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    logger.info(f"Wrote text to {path}")

def write_markdown(path: str, md: str) -> None:
    """Write markdown to file"""
    write_text(path, md)

def print_debug_info(stage: str, data: Any, truncate: int = 500):
    """Print debug information for a stage"""
    print(f"\n{'='*60}")
    print(f"DEBUG: {stage}")
    print(f"{'='*60}")
    
    if isinstance(data, dict):
        print(f"Dict with keys: {list(data.keys())}")
        sample = json.dumps(data, indent=2)
        if len(sample) > truncate:
            print(f"{sample[:truncate]}...")
        else:
            print(sample)
    elif isinstance(data, list):
        print(f"List with {len(data)} items")
        if data and len(str(data[0])) > 100:
            print(f"First item: {str(data[0])[:100]}...")
        else:
            print(f"First item: {data[0] if data else 'Empty'}")
    else:
        data_str = str(data)
        if len(data_str) > truncate:
            print(f"{data_str[:truncate]}...")
        else:
            print(data_str)
    
    print(f"{'='*60}\n")

def get_nested_value(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested value from dictionary using dot-separated key path
    
    Args:
        data: Dictionary to search in
        key_path: Dot-separated path (e.g., 'parsed_json.fields')
        default: Default value if path not found
        
    Returns:
        Value at the specified path, or default if not found
    """
    try:
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                index = int(key)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return default
            else:
                return default
                
        return current
    except Exception as e:
        logger.error(f"Error accessing nested value '{key_path}': {e}")
        return default

def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    Set nested value in dictionary using dot-separated key path
    
    Args:
        data: Dictionary to modify
        key_path: Dot-separated path (e.g., 'parsed_json.fields')
        value: Value to set
        
    Returns:
        Modified dictionary
    """
    keys = key_path.split('.')
    current = data
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value
    return data

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    
    return dict(items)

def unflatten_dict(flat_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary back to nested structure
    
    Args:
        flat_dict: Flattened dictionary
        sep: Separator used for nested keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in flat_dict.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if '[' in part and part.endswith(']'):
                # Handle array indices
                array_key, index_str = part.split('[', 1)
                index = int(index_str.rstrip(']'))
                
                if array_key not in current:
                    current[array_key] = []
                    
                # Extend array if necessary
                while len(current[array_key]) <= index:
                    current[array_key].append({})
                    
                current = current[array_key][index]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Set final value
        final_key = parts[-1]
        if '[' in final_key and final_key.endswith(']'):
            array_key, index_str = final_key.split('[', 1)
            index = int(index_str.rstrip(']'))
            
            if array_key not in current:
                current[array_key] = []
                
            while len(current[array_key]) <= index:
                current[array_key].append(None)
                
            current[array_key][index] = value
        else:
            current[final_key] = value
    
    return result

def calculate_data_hash(data: Any) -> str:
    """
    Calculate consistent hash for any data structure
    
    Args:
        data: Data to hash
        
    Returns:
        SHA256 hash string
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True, ensure_ascii=True)
    else:
        sorted_data = json.dumps(data, ensure_ascii=True)
    
    return hashlib.sha256(sorted_data.encode()).hexdigest()[:16]

def merge_provider_results(results: List[Dict[str, Any]], 
                         merge_strategy: str = 'union') -> Dict[str, Any]:
    """
    Merge results from multiple providers
    
    Args:
        results: List of provider results
        merge_strategy: Strategy for merging ('union', 'intersection', 'majority')
        
    Returns:
        Merged results
    """
    if not results:
        return {}
    
    if len(results) == 1:
        return results[0]
    
    merged = {}
    
    if merge_strategy == 'union':
        # Include all keys from all providers
        for result in results:
            for key, value in result.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **value}
                elif isinstance(value, list) and isinstance(merged[key], list):
                    merged[key].extend(value)
                    
    elif merge_strategy == 'intersection':
        # Only include keys present in all providers
        common_keys = set(results[0].keys())
        for result in results[1:]:
            common_keys &= set(result.keys())
        
        for key in common_keys:
            values = [result[key] for result in results]
            if all(v == values[0] for v in values):
                merged[key] = values[0]
                
    elif merge_strategy == 'majority':
        # Include keys present in majority of providers
        key_counts = defaultdict(int)
        key_values = defaultdict(list)
        
        for result in results:
            for key, value in result.items():
                key_counts[key] += 1
                key_values[key].append(value)
        
        threshold = len(results) // 2 + 1
        
        for key, count in key_counts.items():
            if count >= threshold:
                values = key_values[key]
                # Take most common value
                value_counts = defaultdict(int)
                for value in values:
                    value_hash = calculate_data_hash(value)
                    value_counts[value_hash] += 1
                
                most_common_hash = max(value_counts, key=value_counts.get)
                # Find corresponding value
                for value in values:
                    if calculate_data_hash(value) == most_common_hash:
                        merged[key] = value
                        break
    
    return merged

def validate_consensus_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate consensus result structure and content
    
    Args:
        result: Consensus result to validate
        
    Returns:
        Validation report with errors and warnings
    """
    validation_report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Check required fields
        required_fields = ['target_key_path', 'final_consensus', 'consensus_quality']
        
        for field in required_fields:
            if field not in result:
                validation_report['errors'].append(f"Missing required field: {field}")
                validation_report['valid'] = False
        
        # Check consensus quality
        if 'consensus_quality' in result:
            quality = result['consensus_quality']
            overall_score = quality.get('overall_score', 0)
            
            if overall_score < 0.3:
                validation_report['warnings'].append(f"Low consensus quality score: {overall_score:.3f}")
            
            validation_report['statistics']['quality_score'] = overall_score
        
        # Check final consensus
        if 'final_consensus' in result:
            consensus = result['final_consensus']
            validation_report['statistics']['consensus_items'] = len(consensus)
            
            if not consensus:
                validation_report['warnings'].append("No consensus items found")
        
        # Check for errors in algorithm results
        if 'algorithm_results' in result:
            algorithm_errors = []
            for algorithm, algo_result in result['algorithm_results'].items():
                if 'error' in algo_result:
                    algorithm_errors.append(algorithm)
            
            if algorithm_errors:
                validation_report['warnings'].append(f"Algorithms with errors: {algorithm_errors}")
                validation_report['statistics']['failed_algorithms'] = len(algorithm_errors)
        
        # Check processing time
        processing_time = result.get('processing_time', 0)
        if processing_time > 300:  # 5 minutes
            validation_report['warnings'].append(f"Long processing time: {processing_time:.2f}s")
        
        validation_report['statistics']['processing_time'] = processing_time
        
        return validation_report
        
    except Exception as e:
        validation_report['valid'] = False
        validation_report['errors'].append(f"Validation error: {str(e)}")
        return validation_report

def generate_consensus_summary(consensus_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics across multiple consensus results
    
    Args:
        consensus_results: List of consensus results
        
    Returns:
        Summary statistics
    """
    summary = {
        'total_results': len(consensus_results),
        'successful_results': 0,
        'failed_results': 0,
        'average_quality_score': 0.0,
        'total_consensus_items': 0,
        'average_processing_time': 0.0,
        'algorithm_success_rates': {},
        'target_paths': []
    }
    
    quality_scores = []
    processing_times = []
    algorithm_counts = defaultdict(lambda: {'success': 0, 'total': 0})
    
    for result in consensus_results:
        target_path = result.get('target_key_path', 'unknown')
        summary['target_paths'].append(target_path)
        
        if 'error' in result:
            summary['failed_results'] += 1
        else:
            summary['successful_results'] += 1
            
            # Quality score
            quality = result.get('consensus_quality', {}).get('overall_score', 0)
            quality_scores.append(quality)
            
            # Processing time
            proc_time = result.get('processing_time', 0)
            processing_times.append(proc_time)
            
            # Consensus items
            consensus_count = len(result.get('final_consensus', {}))
            summary['total_consensus_items'] += consensus_count
            
            # Algorithm success rates
            algorithm_results = result.get('algorithm_results', {})
            for algorithm, algo_result in algorithm_results.items():
                algorithm_counts[algorithm]['total'] += 1
                if 'error' not in algo_result:
                    algorithm_counts[algorithm]['success'] += 1
    
    # Calculate averages
    if quality_scores:
        summary['average_quality_score'] = sum(quality_scores) / len(quality_scores)
    
    if processing_times:
        summary['average_processing_time'] = sum(processing_times) / len(processing_times)
    
    # Algorithm success rates
    for algorithm, counts in algorithm_counts.items():
        if counts['total'] > 0:
            success_rate = counts['success'] / counts['total']
            summary['algorithm_success_rates'][algorithm] = success_rate
    
    return summary

def extract_unified_patterns(provider_results: Dict[str, Dict], 
                           pattern_keys: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Extract unified malicious behavioral patterns from all models
    
    Args:
        provider_results: Results from multiple LLM providers
        pattern_keys: Specific pattern keys to extract (optional)
        
    Returns:
        Dictionary mapping pattern types to unified pattern lists
    """
    if pattern_keys is None:
        pattern_keys = ['malicious_patterns', 'behavioral_patterns', 'attack_patterns', 'threat_patterns']
    
    unified_patterns = {}
    
    for pattern_key in pattern_keys:
        patterns = []
        
        for provider, result in provider_results.items():
            # Try multiple path locations for patterns
            possible_paths = [
                f'parsed_json.{pattern_key}',
                f'parsed_json.patterns.{pattern_key}',
                f'parsed_json.analysis.{pattern_key}',
                pattern_key
            ]
            
            for path in possible_paths:
                pattern_data = get_nested_value(result, path)
                
                if pattern_data:
                    if isinstance(pattern_data, list):
                        for pattern in pattern_data:
                            if isinstance(pattern, dict):
                                pattern['_source_provider'] = provider
                                pattern['_source_path'] = path
                            patterns.append(pattern)
                    elif isinstance(pattern_data, dict):
                        pattern_data['_source_provider'] = provider
                        pattern_data['_source_path'] = path
                        patterns.append(pattern_data)
                    break
        
        if patterns:
            unified_patterns[pattern_key] = patterns
    
    logger.info(f"Extracted unified patterns: {[(k, len(v)) for k, v in unified_patterns.items()]}")
    return unified_patterns

def merge_similar_patterns(patterns: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Merge similar patterns based on content similarity
    
    Args:
        patterns: List of pattern dictionaries
        similarity_threshold: Threshold for considering patterns similar
        
    Returns:
        List of merged patterns with consensus information
    """
    if not patterns:
        return []
    
    merged_patterns = []
    processed_indices = set()
    
    for i, pattern1 in enumerate(patterns):
        if i in processed_indices:
            continue
            
        # Find similar patterns
        similar_patterns = [pattern1]
        similar_indices = {i}
        
        for j, pattern2 in enumerate(patterns[i+1:], i+1):
            if j in processed_indices:
                continue
                
            # Calculate similarity based on key content
            similarity = calculate_pattern_similarity(pattern1, pattern2)
            
            if similarity >= similarity_threshold:
                similar_patterns.append(pattern2)
                similar_indices.add(j)
        
        # Merge similar patterns
        if len(similar_patterns) > 1:
            merged_pattern = merge_pattern_group(similar_patterns)
            merged_patterns.append(merged_pattern)
            processed_indices.update(similar_indices)
        else:
            # Add single pattern
            pattern1['_consensus_count'] = 1
            pattern1['_source_providers'] = [pattern1.get('_source_provider', 'unknown')]
            merged_patterns.append(pattern1)
            processed_indices.add(i)
    
    logger.info(f"Merged {len(patterns)} patterns into {len(merged_patterns)} unified patterns")
    return merged_patterns

def calculate_pattern_similarity(pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two patterns
    
    Args:
        pattern1: First pattern dictionary
        pattern2: Second pattern dictionary
        
    Returns:
        Similarity score between 0 and 1
    """
    # Key fields to compare for patterns
    compare_fields = ['name', 'pattern_name', 'title', 'description', 'type', 'category', 'attack_type']
    
    similarities = []
    
    for field in compare_fields:
        val1 = pattern1.get(field, '')
        val2 = pattern2.get(field, '')
        
        if val1 and val2:
            # Calculate string similarity
            if isinstance(val1, str) and isinstance(val2, str):
                # Simple word overlap similarity
                words1 = set(val1.lower().split())
                words2 = set(val2.lower().split())
                
                if words1 and words2:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
            elif val1 == val2:
                similarities.append(1.0)
    
    return sum(similarities) / len(similarities) if similarities else 0.0

def merge_pattern_group(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a group of similar patterns into a consensus pattern
    
    Args:
        patterns: List of similar patterns to merge
        
    Returns:
        Merged consensus pattern
    """
    if not patterns:
        return {}
    
    if len(patterns) == 1:
        patterns[0]['_consensus_count'] = 1
        patterns[0]['_source_providers'] = [patterns[0].get('_source_provider', 'unknown')]
        return patterns[0]
    
    merged = {}
    source_providers = []
    
    # Collect all keys
    all_keys = set()
    for pattern in patterns:
        all_keys.update(pattern.keys())
        if '_source_provider' in pattern:
            source_providers.append(pattern['_source_provider'])
    
    # Merge each field
    for key in all_keys:
        if key.startswith('_source'):
            continue
            
        values = []
        for pattern in patterns:
            if key in pattern and pattern[key]:
                values.append(pattern[key])
        
        if values:
            if isinstance(values[0], str):
                # For strings, take the longest or most detailed one
                merged[key] = max(values, key=len)
            elif isinstance(values[0], (int, float)):
                # For numbers, take the average or most common
                value_counts = defaultdict(int)
                for val in values:
                    value_counts[val] += 1
                merged[key] = max(value_counts, key=value_counts.get)
            elif isinstance(values[0], list):
                # For lists, combine and deduplicate
                combined = []
                for val_list in values:
                    combined.extend(val_list)
                merged[key] = list(set(str(item) for item in combined))
            else:
                # For other types, take the first
                merged[key] = values[0]
    
    # Add consensus metadata
    merged['_consensus_count'] = len(patterns)
    merged['_source_providers'] = list(set(source_providers))
    merged['_confidence_score'] = min(1.0, len(patterns) / 3.0)  # Higher confidence with more sources
    
    return merged

def create_unified_consensus(provider_results: Dict[str, Dict], 
                           target_paths: List[str] = None) -> Dict[str, Any]:
    """
    Create unified consensus for malicious behavioral patterns across all providers
    
    Args:
        provider_results: Results from multiple providers
        target_paths: List of paths to extract (optional)
        
    Returns:
        Unified consensus result
    """
    if target_paths is None:
        # Common paths for malicious patterns
        target_paths = [
            'parsed_json.malicious_patterns',
            'parsed_json.behavioral_patterns', 
            'parsed_json.attack_patterns',
            'parsed_json.threat_indicators',
            'parsed_json.patterns'
        ]
    
    unified_result = {
        'target_paths': target_paths,
        'provider_count': len(provider_results),
        'extracted_patterns': {},
        'unified_patterns': {},
        'consensus_summary': {},
        'processing_timestamp': time.time()
    }
    
    # Extract patterns from all paths
    for path in target_paths:
        patterns = []
        
        for provider, result in provider_results.items():
            pattern_data = get_nested_value(result, path)
            
            if pattern_data:
                if isinstance(pattern_data, list):
                    for pattern in pattern_data:
                        if isinstance(pattern, dict):
                            pattern['_source_provider'] = provider
                            pattern['_source_path'] = path
                        patterns.append(pattern)
                elif isinstance(pattern_data, dict):
                    pattern_data['_source_provider'] = provider  
                    pattern_data['_source_path'] = path
                    patterns.append(pattern_data)
        
        if patterns:
            unified_result['extracted_patterns'][path] = patterns
            
            # Merge similar patterns
            unified_patterns = merge_similar_patterns(patterns)
            unified_result['unified_patterns'][path] = unified_patterns
    
    # Generate summary
    total_patterns = sum(len(patterns) for patterns in unified_result['extracted_patterns'].values())
    total_unified = sum(len(patterns) for patterns in unified_result['unified_patterns'].values())
    
    unified_result['consensus_summary'] = {
        'total_extracted_patterns': total_patterns,
        'total_unified_patterns': total_unified,
        'compression_ratio': (total_patterns - total_unified) / max(1, total_patterns),
        'paths_with_data': len(unified_result['extracted_patterns']),
        'providers_contributing': len(set(
            pattern.get('_source_provider', '') 
            for patterns in unified_result['extracted_patterns'].values()
            for pattern in patterns
        ))
    }
    
    logger.info(f"Created unified consensus: {total_patterns} → {total_unified} patterns")
    return unified_result

def create_preprocessing():
    """Placeholder for ConsensusDataPreprocessor"""
    class ConsensusDataPreprocessor:
        def __init__(self):
            pass
    return ConsensusDataPreprocessor