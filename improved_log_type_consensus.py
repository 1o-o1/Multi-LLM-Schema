#!/usr/bin/env python3
"""
Improved log_type consensus resolution using semantic similarity
Instead of LCS, uses embedding-based semantic similarity to find the best log_type
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class SemanticLogTypeSelector:
    """
    Select the best log_type using semantic similarity between candidates
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with SBERT model for semantic embeddings"""
        self.model = SentenceTransformer(model_name)
        
    def calculate_pairwise_similarities(self, log_types: List[str]) -> Dict[str, float]:
        """
        Calculate pairwise semantic similarities between log_types
        
        Returns:
            Dictionary mapping each log_type to its average similarity with others
        """
        if len(log_types) <= 1:
            return {log_types[0]: 1.0} if log_types else {}
        
        # Generate embeddings for all log_types
        embeddings = self.model.encode(log_types)
        
        # Calculate pairwise similarities
        similarities = {}
        
        for i, log_type_a in enumerate(log_types):
            total_similarity = 0.0
            comparison_count = 0
            
            for j, log_type_b in enumerate(log_types):
                if i != j:  # Don't compare with self
                    # Cosine similarity between embeddings
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    total_similarity += sim
                    comparison_count += 1
                    
                    logger.debug(f"Similarity '{log_type_a}' vs '{log_type_b}': {sim:.3f}")
            
            # Average similarity with other log_types
            avg_similarity = total_similarity / comparison_count if comparison_count > 0 else 0.0
            similarities[log_type_a] = avg_similarity
            
            logger.info(f"'{log_type_a}' average similarity: {avg_similarity:.3f}")
        
        return similarities
    
    def select_best_log_type(self, log_types: List[str]) -> str:
        """
        Select the log_type with highest average semantic similarity to others
        
        Args:
            log_types: List of candidate log_type strings
            
        Returns:
            The best log_type string
        """
        if not log_types:
            return ""
        
        if len(log_types) == 1:
            return log_types[0]
        
        # Remove duplicates while preserving order
        unique_types = []
        for lt in log_types:
            if lt not in unique_types:
                unique_types.append(lt)
        
        if len(unique_types) == 1:
            return unique_types[0]
        
        logger.info(f"Selecting best log_type from: {unique_types}")
        
        # Calculate semantic similarities
        similarities = self.calculate_pairwise_similarities(unique_types)
        
        # Select the one with highest average similarity
        best_log_type = max(similarities.keys(), key=lambda k: similarities[k])
        best_score = similarities[best_log_type]
        
        logger.info(f"Selected '{best_log_type}' with similarity score: {best_score:.3f}")
        
        return best_log_type

def fix_log_type_and_descriptions_in_consensus():
    """
    Fix the missing log_type using semantic similarity and ensure descriptions are preserved
    """
    
    # Load the three original JSON files to get log_type values
    stage1_dir = Path("output_v3/stage1_prompt1")
    json_files = list(stage1_dir.glob("*.json"))
    
    log_types = []
    field_descriptions = {}  # Store all field descriptions for consensus
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Extract log_type
            if 'parsed_json' in data and 'log_type' in data['parsed_json']:
                log_types.append(data['parsed_json']['log_type'])
                logger.info(f"{json_file.name}: {data['parsed_json']['log_type']}")
            
            # Extract field descriptions for consensus
            if 'parsed_json' in data and 'fields' in data['parsed_json']:
                fields = data['parsed_json']['fields']
                for field_name, field_info in fields.items():
                    if field_name not in field_descriptions:
                        field_descriptions[field_name] = {}
                    
                    # Collect descriptions from all models
                    if 'description' in field_info:
                        model_name = json_file.stem  # e.g., 'claude_search_prompt1'
                        field_descriptions[field_name][model_name] = field_info['description']
    
    if not log_types:
        logger.error("No log_type values found in source files")
        return
    
    # Use semantic similarity to select best log_type
    selector = SemanticLogTypeSelector()
    best_log_type = selector.select_best_log_type(log_types)
    logger.info(f"\nSemantically selected log_type: {best_log_type}")
    
    # Select best descriptions using semantic similarity
    best_descriptions = {}
    for field_name, descriptions in field_descriptions.items():
        if descriptions:
            desc_list = list(descriptions.values())
            if len(desc_list) == 1:
                best_descriptions[field_name] = desc_list[0]
            else:
                # Use semantic similarity to select best description
                best_desc = selector.select_best_log_type(desc_list)
                best_descriptions[field_name] = best_desc
                logger.info(f"Field '{field_name}' best description: {best_desc}")
    
    # Update the consensus files
    consensus_files = [
        "test_consensus_semantic_core.json",
        "test_consensus_with_ice.json"
    ]
    
    for consensus_file in consensus_files:
        if Path(consensus_file).exists():
            with open(consensus_file, 'r') as f:
                consensus_data = json.load(f)
            
            # Add log_type to the parsed_json section
            if 'parsed_json' not in consensus_data:
                consensus_data['parsed_json'] = {}
            
            consensus_data['parsed_json']['log_type'] = best_log_type
            
            # Add descriptions to fields
            if 'fields' in consensus_data['parsed_json']:
                fields = consensus_data['parsed_json']['fields']
                for field_name, field_info in fields.items():
                    if field_name in best_descriptions:
                        field_info['description'] = best_descriptions[field_name]
            
            # Save back
            with open(consensus_file, 'w') as f:
                json.dump(consensus_data, f, indent=2)
            
            logger.info(f"Updated {consensus_file} with:")
            logger.info(f"  - log_type: {best_log_type}")
            logger.info(f"  - field descriptions: {len(best_descriptions)} fields")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    fix_log_type_and_descriptions_in_consensus()