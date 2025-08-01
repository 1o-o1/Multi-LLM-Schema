"""
Graph Clustering for Multi-Agent Schema Consensus
GNN-based relationship analysis with community detection algorithms
"""

import numpy as np
import networkx as nx
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
try:
    import community as community_louvain
except ImportError:
    try:
        from networkx.algorithms import community as nx_community
        community_louvain = None
        logger.info("Using NetworkX community detection as fallback")
    except ImportError:
        logger.warning("Community detection module not available")
        community_louvain = None
        nx_community = None
from sklearn.metrics import adjusted_rand_score
import json

logger = logging.getLogger(__name__)

class GraphClustering:
    """GNN-based relationship analysis and community detection"""
    
    def __init__(self):
        """Initialize graph clustering service"""
        self.graph = None
        self.communities = {}
        self.centrality_scores = {}
        
    def build_field_relationship_graph(self, 
                                     provider_results: Dict[str, Dict],
                                     similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Build a graph representing relationships between fields across providers
        
        Args:
            provider_results: Dictionary of provider results
            similarity_threshold: Threshold for creating edges
            
        Returns:
            Dictionary with graph information (compatible with existing code)
        """
        self.graph = nx.Graph()
        
        # Extract all fields with metadata from actual JSON structure
        all_fields = {}
        for provider, result in provider_results.items():
            # Handle both direct content and nested structure
            if isinstance(result, dict):
                if 'content' in result:
                    data_to_analyze = result['content']
                elif 'parsed_json' in result:
                    data_to_analyze = result['parsed_json']
                else:
                    data_to_analyze = result
            else:
                data_to_analyze = result
            
            # Flatten the JSON structure to extract field relationships
            flattened_fields = self._flatten_json_structure(data_to_analyze, provider)
            
            for field_path, field_data in flattened_fields.items():
                node_id = f"{provider}::{field_path}"
                
                # Add node with attributes
                self.graph.add_node(node_id, 
                                  provider=provider,
                                  field_path=field_path,
                                  field_type=type(field_data).__name__,
                                  field_value=str(field_data)[:100] if field_data is not None else 'None',
                                  is_nested=isinstance(field_data, (dict, list)))
                
                all_fields[node_id] = field_data
        
        # Create edges based on path similarity (simpler approach)
        self._add_path_similarity_edges(similarity_threshold)
        
        logger.info(f"Built field relationship graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Return compatible format
        return {
            'graph': self.graph,
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'clusters': self._detect_communities() if self.graph.number_of_nodes() > 0 else [],
            'all_fields': all_fields
        }
    
    def _flatten_json_structure(self, data: Any, provider: str, path: str = "") -> Dict[str, Any]:
        """Flatten JSON structure to extract all field paths"""
        flattened = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)) and len(str(value)) > 1000:
                    # For large nested structures, just store the type
                    flattened[new_path] = f"<{type(value).__name__} with {len(value)} items>"
                elif isinstance(value, (dict, list)):
                    # Recursively flatten smaller structures
                    flattened.update(self._flatten_json_structure(value, provider, new_path))
                else:
                    flattened[new_path] = value
        elif isinstance(data, list):
            flattened[path or "list_root"] = f"<list with {len(data)} items>"
            # Sample first few items if they're simple
            for i, item in enumerate(data[:3]):
                if not isinstance(item, (dict, list)):
                    flattened[f"{path}[{i}]"] = item
        else:
            flattened[path or "value"] = data
        
        return flattened
    
    def _add_path_similarity_edges(self, threshold: float):
        """Add edges based on path similarity (simpler than semantic similarity)"""
        nodes = list(self.graph.nodes())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # Extract path information
                path1 = node1.split('::', 1)[1] if '::' in node1 else node1
                path2 = node2.split('::', 1)[1] if '::' in node2 else node2
                
                # Calculate path similarity
                similarity = self._calculate_path_similarity(path1, path2)
                
                if similarity >= threshold:
                    self.graph.add_edge(node1, node2, weight=similarity, similarity=similarity)
    
    def _calculate_path_similarity(self, path1: str, path2: str) -> float:
        """Calculate similarity between two field paths"""
        # Split paths into components
        parts1 = path1.split('.')
        parts2 = path2.split('.')
        
        # Jaccard similarity
        set1 = set(parts1)
        set2 = set(parts2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_communities(self) -> List[Dict[str, Any]]:
        """Detect communities in the graph"""
        if self.graph.number_of_nodes() < 2:
            return []
        
        try:
            # Use Louvain community detection
            if community_louvain is not None:
                partition = community_louvain.best_partition(self.graph)
            elif 'nx_community' in globals() and nx_community is not None:
                # Use NetworkX community detection as fallback
                communities_generator = nx_community.greedy_modularity_communities(self.graph)
                partition = {}
                for i, community_set in enumerate(communities_generator):
                    for node in community_set:
                        partition[node] = i
            else:
                logger.warning("Community detection module not available")
                return []
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            return [{'id': cid, 'nodes': nodes} for cid, nodes in communities.items()]
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []
    
    def _add_similarity_edges(self, threshold: float):
        """Add edges based on semantic similarity"""
        from .embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService()
        nodes = list(self.graph.nodes())
        
        if len(nodes) < 2:
            return
            
        # Generate embeddings for field descriptions
        descriptions = []
        for node in nodes:
            desc = self.graph.nodes[node].get('description', '')
            field_name = self.graph.nodes[node].get('field_name', '')
            combined = f"{field_name}: {desc}"
            descriptions.append(combined)
        
        try:
            embeddings = embedding_service.embed_text(descriptions)
            similarity_matrix = embedding_service.cosine_similarity_matrix(embeddings)
            
            # Add edges for similar fields
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i >= j:  # Avoid duplicate edges and self-loops
                        continue
                        
                    similarity = similarity_matrix[i, j]
                    if similarity >= threshold:
                        self.graph.add_edge(node1, node2, 
                                          weight=float(similarity),
                                          edge_type='semantic_similarity')
                        
        except Exception as e:
            logger.error(f"Failed to add similarity edges: {e}")
    
    def _add_schema_mapping_edges(self):
        """Add edges based on schema mappings (OCSF, ECS, OSSEM)"""
        schema_groups = defaultdict(list)
        
        # Group nodes by schema mappings
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            for schema in ['OCSF', 'ECS', 'OSSEM']:
                mapping = node_data.get(schema)
                if mapping and mapping != 'null':
                    schema_groups[f"{schema}::{mapping}"].append(node)
        
        # Add edges within schema groups
        for group_nodes in schema_groups.values():
            if len(group_nodes) > 1:
                for i, node1 in enumerate(group_nodes):
                    for j, node2 in enumerate(group_nodes):
                        if i >= j:
                            continue
                            
                        self.graph.add_edge(node1, node2, 
                                          weight=1.0,
                                          edge_type='schema_mapping')
    
    def _add_type_compatibility_edges(self):
        """Add edges based on data type compatibility"""
        type_compatibility = {
            'string': ['string'],
            'integer': ['integer', 'float', 'number'],
            'float': ['float', 'integer', 'number'],
            'number': ['number', 'integer', 'float'],
            'datetime': ['datetime', 'timestamp', 'date'],
            'timestamp': ['timestamp', 'datetime', 'date'],
            'date': ['date', 'datetime', 'timestamp'],
            'ip': ['ip', 'string'],
            'array': ['array', 'list'],
            'list': ['list', 'array'],
            'boolean': ['boolean', 'bool'],
            'bool': ['bool', 'boolean']
        }
        
        nodes = list(self.graph.nodes())
        for i, node1 in enumerate(nodes):
            type1 = self.graph.nodes[node1].get('type', '').lower()
            
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue
                    
                type2 = self.graph.nodes[node2].get('type', '').lower()
                
                # Check type compatibility
                if (type1 in type_compatibility and 
                    type2 in type_compatibility.get(type1, [])):
                    
                    # Add weak edge for type compatibility
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, 
                                          weight=0.3,
                                          edge_type='type_compatibility')
    
    def leiden_community_detection(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Perform Leiden community detection algorithm
        
        Args:
            resolution: Resolution parameter for community detection
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call build_field_relationship_graph first.")
        
        try:
            # Use Louvain as Leiden is not available in standard networkx
            # This is a reasonable approximation
            if community_louvain is not None:
                communities = community_louvain.best_partition(self.graph, resolution=resolution)
            elif 'nx_community' in globals() and nx_community is not None:
                # Use NetworkX community detection as fallback
                communities_generator = nx_community.greedy_modularity_communities(self.graph)
                communities = {}
                for i, community_set in enumerate(communities_generator):
                    for node in community_set:
                        communities[node] = i
            else:
                logger.warning("Community detection module not available")
                return {}
            
            self.communities['leiden'] = communities
            
            n_communities = len(set(communities.values()))
            logger.info(f"Leiden community detection found {n_communities} communities")
            
            return communities
            
        except Exception as e:
            logger.error(f"Leiden community detection failed: {e}")
            # Fallback: each node is its own community
            return {node: i for i, node in enumerate(self.graph.nodes())}
    
    def louvain_community_detection(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Perform Louvain community detection algorithm
        
        Args:
            resolution: Resolution parameter for community detection
            
        Returns:
            Dictionary mapping node IDs to community IDs
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call build_field_relationship_graph first.")
        
        try:
            if community_louvain is None:
                logger.warning("Community detection module not available")
                return {}
            communities = community_louvain.best_partition(self.graph, resolution=resolution)
            
            self.communities['louvain'] = communities
            
            n_communities = len(set(communities.values()))
            logger.info(f"Louvain community detection found {n_communities} communities")
            
            return communities
            
        except Exception as e:
            logger.error(f"Louvain community detection failed: {e}")
            # Fallback: each node is its own community
            return {node: i for i, node in enumerate(self.graph.nodes())}
    
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate various centrality measures for importance scoring
        
        Returns:
            Dictionary of centrality measures for each node
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call build_field_relationship_graph first.")
        
        centrality_measures = {}
        
        try:
            # Degree centrality
            centrality_measures['degree'] = nx.degree_centrality(self.graph)
            
            # Betweenness centrality
            centrality_measures['betweenness'] = nx.betweenness_centrality(self.graph)
            
            # Closeness centrality
            if nx.is_connected(self.graph):
                centrality_measures['closeness'] = nx.closeness_centrality(self.graph)
            else:
                # Handle disconnected graph
                centrality_measures['closeness'] = {}
                for component in nx.connected_components(self.graph):
                    subgraph = self.graph.subgraph(component)
                    closeness = nx.closeness_centrality(subgraph)
                    centrality_measures['closeness'].update(closeness)
            
            # Eigenvector centrality (if possible)
            try:
                centrality_measures['eigenvector'] = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                logger.warning("Could not compute eigenvector centrality")
                centrality_measures['eigenvector'] = {node: 0.0 for node in self.graph.nodes()}
            
            # PageRank
            centrality_measures['pagerank'] = nx.pagerank(self.graph)
            
            self.centrality_scores = centrality_measures
            
            logger.info(f"Calculated centrality measures for {len(self.graph.nodes())} nodes")
            return centrality_measures
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality measures: {e}")
            return {}
    
    def temporal_relationship_discovery(self, 
                                     behavioral_patterns: Dict[str, List[Dict]]) -> Dict[str, List[Tuple]]:
        """
        Discover temporal relationships and causality analysis
        
        Args:
            behavioral_patterns: Behavioral patterns from schema analysis
            
        Returns:
            Dictionary of temporal relationships
        """
        temporal_relationships = defaultdict(list)
        
        try:
            # Analyze patterns for temporal keywords
            temporal_keywords = [
                'before', 'after', 'during', 'followed by', 'precede', 
                'subsequent', 'prior', 'then', 'next', 'finally',
                'temporal', 'time', 'sequence', 'order'
            ]
            
            for pattern_type, patterns in behavioral_patterns.items():
                for pattern in patterns:
                    if 'Instruction' in pattern:
                        instruction = pattern['Instruction'].lower()
                        
                        # Look for temporal relationships
                        for keyword in temporal_keywords:
                            if keyword in instruction:
                                # Extract fields involved in temporal relationship
                                fields = pattern.get('identifiable_fields', [])
                                if len(fields) > 1:
                                    # Create temporal relationship
                                    relationship = (
                                        pattern['pattern_name'],
                                        keyword,
                                        fields,
                                        pattern_type
                                    )
                                    temporal_relationships[keyword].append(relationship)
            
            logger.info(f"Discovered {sum(len(v) for v in temporal_relationships.values())} temporal relationships")
            return dict(temporal_relationships)
            
        except Exception as e:
            logger.error(f"Temporal relationship discovery failed: {e}")
            return {}
    
    def calculate_graph_metrics(self) -> Dict[str, Any]:
        """
        Calculate various graph metrics and visualization data
        
        Returns:
            Dictionary of graph metrics
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call build_field_relationship_graph first.")
        
        metrics = {}
        
        try:
            # Basic graph metrics
            metrics['n_nodes'] = self.graph.number_of_nodes()
            metrics['n_edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)
            metrics['is_connected'] = nx.is_connected(self.graph)
            
            # Connected components
            n_components = nx.number_connected_components(self.graph)
            metrics['n_connected_components'] = n_components
            
            if n_components > 1:
                component_sizes = [len(c) for c in nx.connected_components(self.graph)]
                metrics['largest_component_size'] = max(component_sizes)
                metrics['component_sizes'] = component_sizes
            
            # Clustering coefficient
            metrics['average_clustering'] = nx.average_clustering(self.graph)
            
            # Path-based metrics (for connected graphs)
            if nx.is_connected(self.graph):
                metrics['diameter'] = nx.diameter(self.graph)
                metrics['average_shortest_path_length'] = nx.average_shortest_path_length(self.graph)
            
            # Degree distribution
            degrees = [d for n, d in self.graph.degree()]
            metrics['average_degree'] = np.mean(degrees)
            metrics['degree_distribution'] = {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': min(degrees),
                'max': max(degrees)
            }
            
            # Edge type distribution
            edge_types = defaultdict(int)
            for u, v, data in self.graph.edges(data=True):
                edge_type = data.get('edge_type', 'unknown')
                edge_types[edge_type] += 1
            
            metrics['edge_type_distribution'] = dict(edge_types)
            
            # Provider distribution
            provider_counts = defaultdict(int)
            for node in self.graph.nodes():
                provider = self.graph.nodes[node].get('provider', 'unknown')
                provider_counts[provider] += 1
                
            metrics['provider_distribution'] = dict(provider_counts)
            
            logger.info(f"Calculated graph metrics: {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate graph metrics: {e}")
            return {}
    
    def export_graph_for_visualization(self) -> Dict[str, Any]:
        """
        Export graph data for visualization
        
        Returns:
            Dictionary containing nodes and edges for visualization
        """
        if not self.graph:
            return {'nodes': [], 'edges': []}
        
        # Export nodes
        nodes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            nodes.append({
                'id': node,
                'label': node_data.get('field_name', node),
                'provider': node_data.get('provider', 'unknown'),
                'type': node_data.get('type', 'unknown'),
                'importance': node_data.get('importance', 0),
                'description': node_data.get('description', '')[:100],  # Truncate for visualization
                'community': self.communities.get('louvain', {}).get(node, 0)
            })
        
        # Export edges
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 1.0),
                'type': data.get('edge_type', 'unknown')
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': self.calculate_graph_metrics()
        }
    
    def build_semantic_relationship_graph(self, 
                                        provider_results: Dict[str, Dict],
                                        centralized_embeddings: Dict[str, np.ndarray],
                                        similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Build semantic relationship graph using pre-generated embeddings
        
        This fixes the architectural issue where graph clustering happened before
        semantic embeddings were available.
        
        Args:
            provider_results: Dictionary of provider results
            centralized_embeddings: Pre-generated embeddings for all texts
            similarity_threshold: Threshold for creating semantic edges
            
        Returns:
            Dictionary with semantic graph information
        """
        self.graph = nx.Graph()
        
        # Extract all content with embeddings
        all_content = {}
        for provider, result in provider_results.items():
            if isinstance(result, dict):
                if 'content' in result:
                    data_to_analyze = result['content']
                elif 'parsed_json' in result:
                    data_to_analyze = result['parsed_json']
                else:
                    data_to_analyze = result
            else:
                data_to_analyze = result
            
            # Flatten and extract content
            flattened_content = self._flatten_json_structure(data_to_analyze, provider)
            
            for content_path, content_value in flattened_content.items():
                node_id = f"{provider}::{content_path}"
                content_text = str(content_value)
                
                # Add node with semantic embedding if available
                embedding = centralized_embeddings.get(content_text)
                
                self.graph.add_node(node_id,
                                  provider=provider,
                                  content_path=content_path,
                                  content_text=content_text,
                                  has_embedding=embedding is not None,
                                  content_type=type(content_value).__name__)
                
                all_content[node_id] = {
                    'text': content_text,
                    'embedding': embedding
                }
        
        # Add semantic edges based on embedding similarity
        self._add_semantic_similarity_edges(all_content, similarity_threshold)
        
        logger.info(f"Built semantic relationship graph with {self.graph.number_of_nodes()} nodes "
                   f"and {self.graph.number_of_edges()} semantic edges")
        
        return {
            'graph': self.graph,
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'clusters': self._detect_communities() if self.graph.number_of_nodes() > 0 else [],
            'semantic_metrics': self._calculate_semantic_metrics(all_content)
        }
    
    def _add_semantic_similarity_edges(self, 
                                     all_content: Dict[str, Dict],
                                     similarity_threshold: float):
        """Add edges based on semantic similarity using pre-computed embeddings"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        nodes_with_embeddings = [
            (node_id, content) for node_id, content in all_content.items()
            if content['embedding'] is not None
        ]
        
        if len(nodes_with_embeddings) < 2:
            logger.warning("Not enough nodes with embeddings for semantic edges")
            return
        
        # Calculate pairwise similarities
        for i, (node1_id, content1) in enumerate(nodes_with_embeddings):
            for j, (node2_id, content2) in enumerate(nodes_with_embeddings):
                if i >= j:  # Avoid duplicates and self-loops
                    continue
                
                # Calculate cosine similarity between embeddings
                embedding1 = content1['embedding'].reshape(1, -1)
                embedding2 = content2['embedding'].reshape(1, -1)
                similarity = cosine_similarity(embedding1, embedding2)[0, 0]
                
                if similarity >= similarity_threshold:
                    self.graph.add_edge(node1_id, node2_id,
                                      semantic_similarity=similarity,
                                      edge_type='semantic',
                                      weight=similarity)
    
    def _calculate_semantic_metrics(self, all_content: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate metrics for the semantic graph"""
        total_nodes = len(all_content)
        nodes_with_embeddings = sum(1 for content in all_content.values() 
                                  if content['embedding'] is not None)
        
        return {
            'total_nodes': total_nodes,
            'nodes_with_embeddings': nodes_with_embeddings,
            'embedding_coverage': nodes_with_embeddings / max(total_nodes, 1),
            'semantic_edges': self.graph.number_of_edges(),
            'semantic_density': (2 * self.graph.number_of_edges()) / max(total_nodes * (total_nodes - 1), 1)
        }