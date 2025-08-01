"""
Semantically-Informed Tree Edit Distance (Section 4.3)
Implementation of hybridized structural and semantic similarity metrics
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class TreeNode:
    """Tree node representation for edit distance calculations"""
    
    def __init__(self, label: str, content: str = "", children: List['TreeNode'] = None):
        self.label = label
        self.content = content
        self.children = children or []
        self.embedding = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'label': self.label,
            'content': self.content,
            'children': [child.to_dict() for child in self.children]
        }

class SemanticallyInformedTreeEditDistance:
    """
    Section 4.3: Semantically-Informed Tree Edit Distance
    
    Hybridizes structural Tree Edit Distance with semantic similarity
    using Hungarian algorithm for optimal node mapping
    """
    
    def __init__(self, embedding_service):
        """
        Initialize with embedding service for semantic similarity
        
        Args:
            embedding_service: EmbeddingService instance for SBERT embeddings
        """
        self.embedding_service = embedding_service
        self.cost_cache = {}
        
    def build_tree_from_json(self, json_data: Dict[str, Any], prefix: str = "") -> TreeNode:
        """
        Build tree structure from JSON data
        
        Args:
            json_data: JSON data to convert to tree
            prefix: Prefix for node labels
            
        Returns:
            TreeNode representing the JSON structure
        """
        if isinstance(json_data, dict):
            # Dictionary node
            label = prefix if prefix else "root"
            content = ""
            children = []
            
            for key, value in json_data.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                child_node = self.build_tree_from_json(value, child_prefix)
                children.append(child_node)
            
            return TreeNode(label, content, children)
            
        elif isinstance(json_data, list):
            # Array node
            label = prefix
            content = f"array[{len(json_data)}]"
            children = []
            
            for i, item in enumerate(json_data):
                child_prefix = f"{prefix}[{i}]"
                child_node = self.build_tree_from_json(item, child_prefix)
                children.append(child_node)
                
            return TreeNode(label, content, children)
            
        else:
            # Leaf node
            return TreeNode(prefix, str(json_data), [])
    
    def compute_semantic_similarity_matrix(self, tree1: TreeNode, tree2: TreeNode, 
                                         centralized_embeddings: Dict[str, np.ndarray] = None) -> Tuple[np.ndarray, List[TreeNode], List[TreeNode]]:
        """
        Step 1 of Section 4.3: Pairwise Semantic Mapping
        
        Compute cosine similarity matrix between all nodes in two trees
        using pre-computed or SBERT embeddings
        
        Args:
            tree1: First tree
            tree2: Second tree
            centralized_embeddings: Pre-computed embeddings dict (text -> embedding)
            
        Returns:
            Tuple of (similarity_matrix, tree1_nodes, tree2_nodes)
        """
        # Extract all nodes from both trees
        nodes1 = self._extract_all_nodes(tree1)
        nodes2 = self._extract_all_nodes(tree2)
        
        # Generate text representations for embedding
        texts1 = [self._node_to_text(node) for node in nodes1]
        texts2 = [self._node_to_text(node) for node in nodes2]
        
        # Use pre-computed embeddings if available, otherwise generate new ones
        if centralized_embeddings and texts1 and texts2:
            logger.debug("Using pre-computed centralized embeddings for TED analysis")
            
            # Get embeddings from centralized cache
            embeddings1 = []
            embeddings2 = []
            
            for text in texts1:
                if text in centralized_embeddings:
                    embeddings1.append(centralized_embeddings[text])
                else:
                    # Fallback to embedding service for missing texts
                    logger.debug(f"Text not found in centralized embeddings, generating: {text[:50]}...")
                    embeddings1.append(self.embedding_service.embed_text([text])[0])
            
            for text in texts2:
                if text in centralized_embeddings:
                    embeddings2.append(centralized_embeddings[text])
                else:
                    # Fallback to embedding service for missing texts
                    logger.debug(f"Text not found in centralized embeddings, generating: {text[:50]}...")
                    embeddings2.append(self.embedding_service.embed_text([text])[0])
            
            embeddings1 = np.array(embeddings1)
            embeddings2 = np.array(embeddings2)
            
            # Compute similarity matrix
            similarity_matrix = self.embedding_service.cosine_similarity_matrix(embeddings1, embeddings2)
            
        elif texts1 and texts2:
            logger.debug("Using embedding service for TED analysis (no centralized embeddings)")
            # Generate SBERT embeddings (fallback behavior)
            all_texts = texts1 + texts2
            embeddings = self.embedding_service.embed_text(all_texts)
            
            embeddings1 = embeddings[:len(texts1)]
            embeddings2 = embeddings[len(texts1):]
            
            # Compute similarity matrix
            similarity_matrix = self.embedding_service.cosine_similarity_matrix(embeddings1, embeddings2)
        else:
            similarity_matrix = np.zeros((len(nodes1), len(nodes2)))
        
        logger.debug(f"Computed semantic similarity matrix: {similarity_matrix.shape}")
        return similarity_matrix, nodes1, nodes2
    
    def hungarian_optimal_assignment(self, similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        """
        Step 1 continued: Hungarian Algorithm for Optimal Assignment
        
        Find optimal one-to-one mapping between nodes based on semantic similarity
        
        Args:
            similarity_matrix: Cosine similarity matrix
            
        Returns:
            Tuple of (optimal_assignments, total_similarity)
        """
        if similarity_matrix.size == 0:
            return [], 0.0
            
        # Hungarian algorithm minimizes cost, so we use 1 - similarity as cost
        cost_matrix = 1.0 - similarity_matrix
        
        # Handle non-square matrices by padding
        rows, cols = cost_matrix.shape
        if rows != cols:
            max_dim = max(rows, cols)
            padded_matrix = np.ones((max_dim, max_dim))  # High cost for dummy assignments
            padded_matrix[:rows, :cols] = cost_matrix
            cost_matrix = padded_matrix
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid assignments (non-dummy)
        assignments = []
        total_similarity = 0.0
        
        for row, col in zip(row_indices, col_indices):
            if row < rows and col < cols:  # Valid assignment (not dummy)
                similarity = similarity_matrix[row, col]
                if similarity > 0.1:  # Minimum similarity threshold
                    assignments.append((row, col))
                    total_similarity += similarity
        
        logger.debug(f"Hungarian algorithm found {len(assignments)} optimal assignments")
        return assignments, total_similarity
    
    def compute_hybrid_tree_edit_distance(self, 
                                        tree1: TreeNode, 
                                        tree2: TreeNode,
                                        alpha: float = 0.5,
                                        centralized_embeddings: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """
        Step 2 of Section 4.3: Structurally-Aware TED Calculation
        
        Compute Tree Edit Distance with dynamically defined costs based on semantic similarity
        
        Args:
            tree1: First tree
            tree2: Second tree  
            alpha: Weight balance between structural (1-alpha) and semantic (alpha) components
            centralized_embeddings: Pre-computed embeddings dict (text -> embedding)
            
        Returns:
            Dictionary containing hybrid distance and detailed analysis
        """
        # Step 1: Compute semantic similarity and optimal mapping
        similarity_matrix, nodes1, nodes2 = self.compute_semantic_similarity_matrix(tree1, tree2, centralized_embeddings)
        assignments, total_similarity = self.hungarian_optimal_assignment(similarity_matrix)
        
        # Create mapping from assignments
        semantic_mapping = {row: col for row, col in assignments}
        
        # Step 2: Compute TED with semantic-informed costs
        ted_result = self._compute_ted_with_semantic_costs(
            tree1, tree2, nodes1, nodes2, similarity_matrix, semantic_mapping
        )
        
        # Step 3: Hybrid score calculation
        structural_score = 1.0 - (ted_result['edit_distance'] / max(len(nodes1), len(nodes2), 1))
        semantic_score = total_similarity / len(assignments) if assignments else 0.0
        
        hybrid_score = alpha * semantic_score + (1 - alpha) * structural_score
        
        result = {
            'hybrid_similarity_score': hybrid_score,
            'structural_similarity': structural_score,
            'semantic_similarity': semantic_score,
            'tree_edit_distance': ted_result['edit_distance'],
            'semantic_assignments': len(assignments),
            'total_nodes': (len(nodes1), len(nodes2)),
            'edit_operations': ted_result['operations'],
            'optimal_node_mapping': assignments,
            'similarity_matrix_shape': similarity_matrix.shape,
            'alpha_weight': alpha
        }
        
        logger.info(f"Hybrid TED computed: {hybrid_score:.3f} (structural: {structural_score:.3f}, semantic: {semantic_score:.3f})")
        return result
    
    def _extract_all_nodes(self, tree: TreeNode) -> List[TreeNode]:
        """Extract all nodes from tree in depth-first order"""
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self._extract_all_nodes(child))
        return nodes
    
    def _node_to_text(self, node: TreeNode) -> str:
        """Convert node to text representation for embedding"""
        text_parts = []
        
        if node.label:
            text_parts.append(f"label: {node.label}")
        if node.content:
            text_parts.append(f"content: {node.content}")
        if node.children:
            text_parts.append(f"children_count: {len(node.children)}")
            
        return " | ".join(text_parts) if text_parts else "empty_node"
    
    def _compute_ted_with_semantic_costs(self, 
                                       tree1: TreeNode, 
                                       tree2: TreeNode,
                                       nodes1: List[TreeNode],
                                       nodes2: List[TreeNode], 
                                       similarity_matrix: np.ndarray,
                                       semantic_mapping: Dict[int, int]) -> Dict[str, Any]:
        """
        Compute Tree Edit Distance with semantic-informed costs
        
        Uses dynamic programming with semantic similarity to adjust rename costs
        """
        # Simplified TED calculation with semantic costs
        # In practice, would implement full TED DP algorithm with custom costs
        
        n1, n2 = len(nodes1), len(nodes2)
        
        # Initialize DP table
        dp = np.zeros((n1 + 1, n2 + 1))
        
        # Base cases
        for i in range(n1 + 1):
            dp[i][0] = i  # Delete all nodes from tree1
        for j in range(n2 + 1):
            dp[0][j] = j  # Insert all nodes from tree2
        
        operations = []
        
        # Fill DP table
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                node1_idx = i - 1
                node2_idx = j - 1
                
                # Calculate costs
                delete_cost = dp[i-1][j] + 1
                insert_cost = dp[i][j-1] + 1
                
                # Semantic-informed rename cost
                if node1_idx in semantic_mapping and semantic_mapping[node1_idx] == node2_idx:
                    # Nodes are semantically mapped - low rename cost
                    similarity = similarity_matrix[node1_idx, node2_idx]
                    rename_cost = dp[i-1][j-1] + (1.0 - similarity)  # Section 4.3 formula
                else:
                    # Nodes not mapped - high rename cost (prohibitively expensive)
                    rename_cost = dp[i-1][j-1] + 2.0  # High constant value
                
                # Choose minimum cost operation
                min_cost = min(delete_cost, insert_cost, rename_cost)
                dp[i][j] = min_cost
                
                # Record operation for analysis
                if min_cost == rename_cost and rename_cost < delete_cost and rename_cost < insert_cost:
                    operations.append(('rename', node1_idx, node2_idx, rename_cost - dp[i-1][j-1]))
                elif min_cost == delete_cost:
                    operations.append(('delete', node1_idx, -1, 1.0))
                elif min_cost == insert_cost:
                    operations.append(('insert', -1, node2_idx, 1.0))
        
        return {
            'edit_distance': dp[n1][n2],
            'operations': operations,
            'dp_table_shape': dp.shape
        }
    
    def batch_tree_comparison(self, 
                            trees: List[TreeNode],
                            tree_ids: List[str] = None,
                            centralized_embeddings: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare multiple trees using hybrid similarity metrics
        
        Args:
            trees: List of TreeNode objects to compare
            tree_ids: Optional identifiers for trees
            centralized_embeddings: Pre-computed embeddings dict (text -> embedding)
            
        Returns:
            Dictionary containing pairwise similarity matrix and analysis
        """
        if tree_ids is None:
            tree_ids = [f"tree_{i}" for i in range(len(trees))]
            
        n_trees = len(trees)
        similarity_matrix = np.zeros((n_trees, n_trees))
        detailed_comparisons = {}
        
        # Compute pairwise similarities
        for i in range(n_trees):
            for j in range(i, n_trees):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    comparison = self.compute_hybrid_tree_edit_distance(trees[i], trees[j], centralized_embeddings=centralized_embeddings)
                    similarity = comparison['hybrid_similarity_score']
                    
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity  # Symmetric
                    
                    detailed_comparisons[f"{tree_ids[i]}_vs_{tree_ids[j]}"] = comparison
        
        # Compute analysis metrics
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(n_trees, k=1)])
        max_similarity = np.max(similarity_matrix[np.triu_indices(n_trees, k=1)])
        min_similarity = np.min(similarity_matrix[np.triu_indices(n_trees, k=1)])
        
        result = {
            'similarity_matrix': similarity_matrix,
            'tree_ids': tree_ids,
            'detailed_comparisons': detailed_comparisons,
            'analysis': {
                'n_trees': n_trees,
                'average_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'min_similarity': min_similarity,
                'total_comparisons': len(detailed_comparisons)
            }
        }
        
        logger.info(f"Batch comparison complete: {n_trees} trees, avg similarity: {avg_similarity:.3f}")
        return result
    
    def convert_json_results_to_trees(self, provider_results: Dict[str, Dict]) -> List[Tuple[str, TreeNode]]:
        """
        Convert provider JSON results to tree representations
        
        Args:
            provider_results: Dictionary of provider results
            
        Returns:
            List of (provider_name, tree) tuples
        """
        trees = []
        
        for provider, result in provider_results.items():
            try:
                # Extract the parsed JSON or use full result
                json_data = result.get('parsed_json', result)
                tree = self.build_tree_from_json(json_data, f"{provider}_root")
                trees.append((provider, tree))
                
            except Exception as e:
                logger.error(f"Failed to convert {provider} result to tree: {e}")
                # Create minimal tree for failed conversions
                fallback_tree = TreeNode(f"{provider}_error", str(result), [])
                trees.append((provider, fallback_tree))
        
        logger.info(f"Converted {len(trees)} provider results to tree structures")
        return trees