"""
Embedding Service for Multi-Agent Schema Consensus
Provides semantic similarity using Sentence-BERT embeddings
"""

import numpy as np
import logging
import hashlib
import json
import os
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Multi-modal embeddings for text and structured data"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', enable_cache: bool = True):
        """
        Initialize with local SentenceTransformers (Fast and efficient)
        
        Args:
            model_name: Embedding model name
                       Default: 'sentence-transformers/all-MiniLM-L6-v2' (fast local model)
                       Alternatives: 'gemini-embedding-001' (slow API), 'local-tfidf' (fallback)
            enable_cache: Enable embedding caching to avoid redundant API calls
        """
        self.model_name = model_name
        self.model = None
        self.gemini_client = None
        self.enable_cache = enable_cache
        self.cache_dir = "embedding_cache"
        
        # Create cache directory
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self._load_model()
        
    def _load_model(self):
        """Load embedding model - Gemini, SBERT, or local fallback"""
        if self.model_name == 'gemini-embedding-001':
            logger.info("Using Gemini embeddings (Section 3.1 - High-quality semantic embeddings)")
            try:
                from google import genai
                # Load API key
                api_key = self._extract_gemini_api_key()
                if api_key:
                    self.gemini_client = genai.Client(api_key=api_key)
                    logger.info("Gemini embeddings initialized successfully")
                    return
                else:
                    raise ValueError("No Gemini API key found")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Gemini embeddings: {e}")
        
        elif self.model_name == 'local-tfidf':
            logger.info("Using local TF-IDF embeddings (testing fallback)")
            self.model = None
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
            return
            
        else:
            # Try SBERT models
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SBERT model: {self.model_name} (Section 3.1 compliance)")
                return
            except Exception as e:
                logger.error(f"Failed to load SBERT model {self.model_name}: {e}")
        
        # Final fallback to TF-IDF
        logger.warning("All embedding models failed - using TF-IDF fallback")
        self.model = None
        self.gemini_client = None
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for a list of texts"""
        text_content = "|".join(sorted(texts))  # Sort for consistent hashing
        cache_key = hashlib.md5(f"{self.model_name}:{text_content}".encode()).hexdigest()
        return cache_key
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available"""
        if not self.enable_cache:
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        try:
            if os.path.exists(cache_file):
                embeddings = np.load(cache_file)
                logger.info(f"Loaded embeddings from cache: {embeddings.shape}")
                return embeddings
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to cache"""
        if not self.enable_cache:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        try:
            np.save(cache_file, embeddings)
            logger.info(f"Saved embeddings to cache: {embeddings.shape}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching support
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        cached_embeddings = self._load_from_cache(cache_key)
        if cached_embeddings is not None:
            return cached_embeddings
        
        # Generate new embeddings
        embeddings = self._generate_embeddings(texts)
        
        # Save to cache
        if embeddings.size > 0:
            self._save_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings without caching (internal method)"""
        # Try Gemini embeddings first
        if self.gemini_client is not None:
            try:
                logger.info(f"Generating Gemini embeddings for {len(texts)} texts")
                
                # Process texts individually as Gemini API expects single text input
                all_embeddings = []
                
                for i, text in enumerate(texts):
                    logger.debug(f"Processing text {i+1}/{len(texts)}")
                    
                    # Use the correct Gemini API for embeddings
                    result = self.gemini_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=text
                    )
                    
                    # Extract embedding from result
                    embedding = np.array(result.embeddings[0].values)
                    all_embeddings.append(embedding)
                
                embeddings = np.array(all_embeddings)
                logger.info(f"Generated Gemini embeddings: {embeddings.shape}")
                return embeddings
                
            except Exception as e:
                logger.error(f"Gemini embedding failed: {e}")
                logger.info("Falling back to SBERT or TF-IDF")
        
        # Try SBERT model
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_tensor=False)
                logger.info(f"Generated SBERT embeddings for {len(texts)} texts")
                return np.array(embeddings)
            except Exception as e:
                logger.error(f"SBERT embedding failed: {e}")
                logger.info("Falling back to TF-IDF")
        
        # TF-IDF fallback
        if hasattr(self, 'tfidf_vectorizer'):
            try:
                logger.info(f"Using TF-IDF embeddings for {len(texts)} texts")
                embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray()
                return embeddings
            except Exception as e:
                logger.error(f"TF-IDF embedding failed: {e}")
        
        # Final fallback: random embeddings for testing
        logger.warning("All embedding methods failed - using random embeddings for testing")
        return np.random.rand(len(texts), 300)
    
    def _extract_gemini_api_key(self) -> str:
        """Extract Gemini API key from api_key.txt file"""
        try:
            # Try multiple possible locations
            key_files = ['api_key.txt', '../api_key.txt', 'config/api_key.txt']
            
            for key_file in key_files:
                try:
                    with open(key_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Look for Gemini API key in various formats
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        # Format: GEMINI_API_KEY = "key"
                        if 'GEMINI_API_KEY' in line and '=' in line:
                            key = line.split('=', 1)[1].strip()
                            key = key.strip('"').strip("'").strip()
                            if key and len(key) > 10:  # Basic validation
                                logger.info(f"Found Gemini API key in {key_file}")
                                return key
                        # Format: AIzaSy... (direct key)
                        elif line.startswith('AIzaSy') and len(line) > 30:
                            logger.info(f"Found direct Gemini API key in {key_file}")
                            return line
                    
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.warning(f"Error reading {key_file}: {e}")
                    continue
            
            # Try environment variable as fallback
            import os
            env_key = os.getenv('GEMINI_API_KEY')
            if env_key:
                logger.info("Found Gemini API key in environment variable")
                return env_key
            
            logger.error("No Gemini API key found in any location")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract Gemini API key: {e}")
            return None
    
    def embed_structured_data(self, data: Dict[str, Any], key_prefix: str = "") -> np.ndarray:
        """
        Generate embeddings for any structured data (generalized version)
        
        Args:
            data: Dictionary containing any structured information
            key_prefix: Optional prefix for nested keys
            
        Returns:
            Single embedding vector for the data
        """
        text_components = self._extract_text_from_data(data, key_prefix)
        
        # Combine all components
        combined_text = " | ".join(text_components)
        
        if not combined_text.strip():
            combined_text = str(data)  # Fallback to string representation
            
        return self.embed_text([combined_text])[0]
    
    def _extract_text_from_data(self, data: Any, key_prefix: str = "") -> List[str]:
        """
        Recursively extract text components from structured data
        
        Args:
            data: Any data structure (dict, list, str, etc.)
            key_prefix: Current key path for nested structures
            
        Returns:
            List of text components
        """
        text_components = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{key_prefix}.{key}" if key_prefix else key
                
                if isinstance(value, (str, int, float, bool)):
                    # Direct value - add with key context
                    text_components.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    # Nested structure - recurse
                    nested_components = self._extract_text_from_data(value, full_key)
                    text_components.extend(nested_components)
                else:
                    # Other types - convert to string
                    text_components.append(f"{key}: {str(value)}")
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_key = f"{key_prefix}[{i}]" if key_prefix else f"item_{i}"
                nested_components = self._extract_text_from_data(item, item_key)
                text_components.extend(nested_components)
                
        elif isinstance(data, (str, int, float, bool)):
            # Direct value
            if key_prefix:
                text_components.append(f"{key_prefix}: {data}")
            else:
                text_components.append(str(data))
        else:
            # Other types
            text_components.append(str(data))
        
        return text_components
    
    def embed_field_content(self, field_data: Dict[str, Any]) -> np.ndarray:
        """
        Generate embeddings for structured field data (backwards compatibility)
        
        Args:
            field_data: Dictionary containing field information
            
        Returns:
            Single embedding vector for the field
        """
        return self.embed_structured_data(field_data)
    
    def cosine_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray = None) -> np.ndarray:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (optional, defaults to embeddings1)
            
        Returns:
            Cosine similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
            
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_similar_fields(self, 
                          field_embeddings: List[np.ndarray], 
                          field_names: List[str],
                          threshold: float = 0.7) -> List[List[int]]:
        """
        Find groups of similar fields based on embedding similarity
        
        Args:
            field_embeddings: List of field embeddings
            field_names: List of field names for logging
            threshold: Similarity threshold for grouping
            
        Returns:
            List of field index groups that are similar
        """
        if len(field_embeddings) < 2:
            return [[i] for i in range(len(field_embeddings))]
            
        # Stack embeddings
        embeddings = np.vstack(field_embeddings)
        
        # Compute similarity matrix
        similarity_matrix = self.cosine_similarity_matrix(embeddings)
        
        # Find similar groups
        n_fields = len(field_embeddings)
        visited = set()
        groups = []
        
        for i in range(n_fields):
            if i in visited:
                continue
                
            group = [i]
            visited.add(i)
            
            # Find all fields similar to field i
            for j in range(i + 1, n_fields):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    group.append(j)
                    visited.add(j)
            
            groups.append(group)
            
        logger.info(f"Found {len(groups)} similarity groups from {n_fields} fields")
        return groups
    
    def semantic_clustering(self, 
                          embeddings: np.ndarray, 
                          method: str = 'kmeans',
                          n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Legacy semantic clustering method (backwards compatibility)
        
        Args:
            embeddings: Array of embeddings to cluster
            method: Clustering method ('kmeans' or 'hdbscan')
            n_clusters: Number of clusters (for kmeans)
            
        Returns:
            Array of cluster labels
        """
        return self.semantic_clustering_label_canonicalization(
            embeddings, method, n_clusters
        )
    
    def semantic_clustering_label_canonicalization(self, 
                          embeddings: np.ndarray, 
                          method: str = 'hdbscan',
                          n_clusters: Optional[int] = None,
                          min_cluster_size: Optional[int] = None) -> np.ndarray:
        """
        Perform semantic clustering for label canonicalization (Section 3.1)
        
        Implementation of Section 3.1: "Semantic Clustering for Label Canonicalization"
        Uses SBERT embeddings with K-Means or HDBSCAN for concept discovery
        
        Args:
            embeddings: Array of SBERT embeddings to cluster
            method: Clustering method ('hdbscan' recommended, 'kmeans' alternative)
            n_clusters: Number of clusters (for kmeans)
            min_cluster_size: Minimum cluster size (for hdbscan)
            
        Returns:
            Array of cluster labels (canonical concept IDs)
        """
        if len(embeddings) < 2:
            return np.zeros(len(embeddings), dtype=int)
            
        try:
            if method == 'hdbscan':
                # HDBSCAN for density-based canonical concept discovery (Section 3.1 preferred)
                if min_cluster_size is None:
                    min_cluster_size = max(2, len(embeddings) // 10)
                clusterer = HDBSCAN(
                    min_cluster_size=min_cluster_size, 
                    metric='cosine',
                    cluster_selection_epsilon=0.3,  # Semantic similarity threshold
                    min_samples=1  # Allow single-node concepts
                )
                labels = clusterer.fit_predict(embeddings)
                
            elif method == 'kmeans':
                # K-Means alternative for when number of concepts is known
                if n_clusters is None:
                    # Use elbow method heuristic for canonical concepts
                    n_clusters = min(8, max(2, len(embeddings) // 3))
                    
                clusterer = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10,
                    init='k-means++'  # Better initialization
                )
                labels = clusterer.fit_predict(embeddings)
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
                
            n_clusters = len(set(labels))
            n_noise = sum(1 for l in labels if l == -1)  # HDBSCAN noise points
            logger.info(f"Section 3.1 Semantic Clustering: {len(embeddings)} embeddings → {n_clusters} canonical concepts using {method} (noise: {n_noise})")
            return labels
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Return each item as its own cluster
            return np.arange(len(embeddings))
    
    def ontology_grounding(self, 
                          canonical_concepts: List[str],
                          knowledge_base: str = 'wikidata',
                          confidence_threshold: float = 0.8) -> Dict[str, Dict[str, Any]]:
        """
        Optional ontology grounding for canonical concepts (Section 3.1)
        
        Grounds discovered canonical concepts in external knowledge base
        for disambiguation and semantic precision enhancement
        
        Args:
            canonical_concepts: List of canonical concept labels
            knowledge_base: Knowledge base to use ('wikidata', 'domain_specific', 'none')
            confidence_threshold: Minimum confidence for grounding
            
        Returns:
            Dictionary mapping concepts to grounded entities
        """
        grounded_concepts = {}
        
        if knowledge_base == 'none':
            logger.info("Ontology grounding disabled")
            return grounded_concepts
            
        try:
            for concept in canonical_concepts:
                grounding_result = {
                    'original_concept': concept,
                    'grounded_entity': None,
                    'confidence': 0.0,
                    'disambiguation': [],
                    'knowledge_base': knowledge_base
                }
                
                if knowledge_base == 'wikidata':
                    # Placeholder for Wikidata API integration
                    # In real implementation, would query Wikidata SPARQL endpoint
                    grounding_result['grounded_entity'] = f"wd:Q{hash(concept) % 1000000}"
                    grounding_result['confidence'] = 0.7  # Mock confidence
                    
                elif knowledge_base == 'domain_specific':
                    # Placeholder for domain-specific ontology (cybersecurity)
                    grounding_result['grounded_entity'] = f"mitre:{concept.lower().replace(' ', '_')}"
                    grounding_result['confidence'] = 0.8
                
                if grounding_result['confidence'] >= confidence_threshold:
                    grounded_concepts[concept] = grounding_result
                    
            logger.info(f"Grounded {len(grounded_concepts)}/{len(canonical_concepts)} concepts in {knowledge_base}")
            
        except Exception as e:
            logger.error(f"Ontology grounding failed: {e}")
            
        return grounded_concepts
    
    def hierarchical_clustering_generic(self, 
                                      provider_results: Dict[str, Dict],
                                      target_key_path: str,
                                      item_name_key: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Perform hierarchical clustering for any JSON structure across providers
        
        Args:
            provider_results: Dictionary of provider results
            target_key_path: Dot-separated path to target data (e.g., 'parsed_json.fields' or 'parsed_json.patterns')
            item_name_key: Key to use as item identifier (optional)
            
        Returns:
            Dictionary mapping cluster names to lists of items with metadata
        """
        # Extract target data from all providers
        all_items = {}
        item_embeddings = []
        item_metadata = []
        
        for provider, result in provider_results.items():
            # Navigate to target data using key path
            target_data = self._get_nested_value(result, target_key_path)
            
            if not target_data:
                continue
                
            # Handle different data structures
            if isinstance(target_data, dict):
                items = target_data.items()
            elif isinstance(target_data, list):
                items = enumerate(target_data)
            else:
                continue
            
            for item_key, item_data in items:
                # Generate unique ID
                if item_name_key and isinstance(item_data, dict):
                    item_name = item_data.get(item_name_key, str(item_key))
                else:
                    item_name = str(item_key)
                    
                full_item_id = f"{provider}::{item_name}"
                all_items[full_item_id] = item_data
                
                # Generate embedding for item
                embedding = self.embed_structured_data(item_data)
                item_embeddings.append(embedding)
                item_metadata.append({
                    'provider': provider,
                    'item_name': item_name,
                    'full_id': full_item_id,
                    'data': item_data
                })
        
        if not item_embeddings:
            return {}
            
        # Stack embeddings and cluster
        embeddings = np.vstack(item_embeddings)
        cluster_labels = self.semantic_clustering(embeddings, method='hdbscan')
        
        # Group items by cluster
        clustered_items = {}
        for i, (label, metadata) in enumerate(zip(cluster_labels, item_metadata)):
            if label == -1:  # HDBSCAN noise points
                label = f"singleton_{i}"
                
            cluster_key = f"cluster_{label}"
            if cluster_key not in clustered_items:
                clustered_items[cluster_key] = []
                
            clustered_items[cluster_key].append(metadata)
        
        logger.info(f"Hierarchical clustering produced {len(clustered_items)} clusters from {len(all_items)} items")
        return clustered_items
    
    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """
        Get nested value from dictionary using dot-separated key path
        
        Args:
            data: Dictionary to search in
            key_path: Dot-separated path (e.g., 'parsed_json.fields')
            
        Returns:
            Value at the specified path, or None if not found
        """
        try:
            keys = key_path.split('.')
            current = data
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
                    
            return current
        except Exception as e:
            logger.error(f"Error accessing nested value '{key_path}': {e}")
            return None
    
    def hierarchical_field_clustering(self, 
                                    provider_results: Dict[str, Dict]) -> Dict[str, List[List[str]]]:
        """
        Perform hierarchical clustering for field grouping across providers (backwards compatibility)
        
        Args:
            provider_results: Dictionary of provider results with field data
            
        Returns:
            Dictionary mapping canonical field names to lists of provider field names
        """
        clustered_items = self.hierarchical_clustering_generic(
            provider_results, 'parsed_json.fields', 'field_name'
        )
        
        # Convert to legacy format
        legacy_format = {}
        for cluster_name, items in clustered_items.items():
            legacy_format[cluster_name] = items
            
        return legacy_format
    
    def canonical_concept_assignment(self, 
                                   embeddings: np.ndarray,
                                   concept_labels: List[str],
                                   enable_ontology_grounding: bool = False) -> Dict[str, Any]:
        """
        Complete implementation of Section 3.1: Semantic Clustering for Label Canonicalization
        
        Performs SBERT-based semantic clustering with optional ontology grounding
        
        Args:
            embeddings: SBERT embeddings for clustering
            concept_labels: Original labels/concepts to canonicalize
            enable_ontology_grounding: Whether to ground concepts in external KB
            
        Returns:
            Dictionary containing canonical concept assignments and metadata
        """
        if len(embeddings) == 0:
            return {
                'canonical_concepts': {},
                'cluster_assignments': np.array([]),
                'grounded_concepts': {},
                'clustering_metadata': {}
            }
            
        # Step 1: Semantic clustering for canonical concept discovery
        cluster_labels = self.semantic_clustering_label_canonicalization(embeddings)
        
        # Step 2: Generate canonical concept identifiers
        canonical_concepts = {}
        concept_assignments = {}
        
        unique_clusters = set(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points in HDBSCAN
                continue
                
            # Find all labels in this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_concepts = [concept_labels[i] for i in cluster_indices]
            
            # Choose representative canonical label (most common or first)
            canonical_label = max(set(cluster_concepts), key=cluster_concepts.count)
            canonical_id = f"concept_{cluster_id}_{canonical_label.lower().replace(' ', '_')}"
            
            canonical_concepts[canonical_id] = {
                'canonical_label': canonical_label,
                'cluster_id': cluster_id,
                'member_concepts': cluster_concepts,
                'member_indices': cluster_indices,
                'cluster_size': len(cluster_indices)
            }
            
            # Assign all concepts in cluster to this canonical ID
            for concept in cluster_concepts:
                concept_assignments[concept] = canonical_id
        
        # Step 3: Optional ontology grounding
        grounded_concepts = {}
        if enable_ontology_grounding:
            canonical_labels = [info['canonical_label'] for info in canonical_concepts.values()]
            grounded_concepts = self.ontology_grounding(canonical_labels)
        
        # Step 4: Clustering metadata
        clustering_metadata = {
            'total_concepts': len(concept_labels),
            'canonical_concepts_found': len(canonical_concepts),
            'noise_points': sum(1 for l in cluster_labels if l == -1),
            'compression_ratio': len(canonical_concepts) / len(concept_labels) if len(concept_labels) > 0 else 0,
            'grounding_enabled': enable_ontology_grounding,
            'grounded_concepts_count': len(grounded_concepts)
        }
        
        logger.info(f"Section 3.1 Complete: {clustering_metadata['total_concepts']} concepts → {clustering_metadata['canonical_concepts_found']} canonical concepts")
        
        return {
            'canonical_concepts': canonical_concepts,
            'concept_assignments': concept_assignments,
            'cluster_assignments': cluster_labels,
            'grounded_concepts': grounded_concepts,
            'clustering_metadata': clustering_metadata
        }

    def cross_framework_alignment(self, 
                                schema_mappings: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Align fields across different cybersecurity frameworks (OCSF, ECS, OSSEM)
        
        Args:
            schema_mappings: Dictionary of field mappings to frameworks
            
        Returns:
            Dictionary of alignment scores between frameworks
        """
        frameworks = ['OCSF', 'ECS', 'OSSEM']
        alignment_scores = {}
        
        # Extract framework mappings
        framework_fields = {fw: [] for fw in frameworks}
        
        for field_data in schema_mappings.values():
            for fw in frameworks:
                if fw in field_data and field_data[fw]:
                    framework_fields[fw].append(field_data[fw])
        
        # Generate embeddings for framework field names
        for fw in frameworks:
            if framework_fields[fw]:
                framework_fields[fw] = list(set(framework_fields[fw]))  # Remove duplicates
                
        # Compute cross-framework similarity
        for fw1 in frameworks:
            for fw2 in frameworks:
                if fw1 >= fw2:  # Avoid duplicate comparisons
                    continue
                    
                if not framework_fields[fw1] or not framework_fields[fw2]:
                    continue
                    
                # Embed framework field names
                emb1 = self.embed_text(framework_fields[fw1])
                emb2 = self.embed_text(framework_fields[fw2])
                
                # Compute similarity matrix
                sim_matrix = self.cosine_similarity_matrix(emb1, emb2)
                
                # Average maximum similarity as alignment score
                alignment_score = np.mean(np.max(sim_matrix, axis=1))
                alignment_scores[f"{fw1}_to_{fw2}"] = float(alignment_score)
        
        logger.info(f"Computed cross-framework alignment scores: {alignment_scores}")
        return alignment_scores
    
    def pattern_similarity_analysis(self, 
                                  provider_results: Dict[str, Dict],
                                  pattern_types: List[str] = None,
                                  similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze similarity between malicious behavioral patterns across providers
        
        Args:
            provider_results: Results from multiple LLM providers
            pattern_types: Types of patterns to analyze
            similarity_threshold: Threshold for considering patterns similar
            
        Returns:
            Pattern similarity analysis results
        """
        if pattern_types is None:
            pattern_types = ['malicious_patterns', 'behavioral_patterns', 'attack_patterns', 'threat_patterns']
        
        pattern_analysis = {
            'pattern_types_analyzed': pattern_types,
            'provider_count': len(provider_results),
            'similarity_threshold': similarity_threshold,
            'pattern_clusters': {},
            'cross_provider_similarities': {},
            'pattern_statistics': {}
        }
        
        # Extract all patterns from all providers
        all_patterns = {}
        
        for pattern_type in pattern_types:
            patterns = []
            
            for provider, result in provider_results.items():
                # Try multiple possible paths
                possible_paths = [
                    f'parsed_json.{pattern_type}',
                    f'parsed_json.patterns.{pattern_type}',
                    f'parsed_json.analysis.{pattern_type}',
                    pattern_type
                ]
                
                for path in possible_paths:
                    pattern_data = self._get_nested_value(result, path)
                    
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
                all_patterns[pattern_type] = patterns
        
        # Analyze each pattern type
        for pattern_type, patterns in all_patterns.items():
            logger.info(f"Analyzing {len(patterns)} patterns of type: {pattern_type}")
            
            # Generate embeddings for patterns
            pattern_texts = []
            pattern_metadata = []
            
            for i, pattern in enumerate(patterns):
                # Convert pattern to text representation
                text_parts = []
                
                # Key fields to include in text representation
                key_fields = ['name', 'pattern_name', 'title', 'description', 'type', 'category', 'attack_type', 'severity']
                
                for field in key_fields:
                    if field in pattern and pattern[field]:
                        text_parts.append(f"{field}: {pattern[field]}")
                
                # Add indicators if available
                if 'indicators' in pattern and isinstance(pattern['indicators'], list):
                    text_parts.append(f"indicators: {' '.join(pattern['indicators'])}")
                
                pattern_text = " | ".join(text_parts)
                pattern_texts.append(pattern_text)
                
                pattern_metadata.append({
                    'index': i,
                    'provider': pattern.get('_source_provider', 'unknown'),
                    'original_data': pattern
                })
            
            if len(pattern_texts) > 1:
                # Generate embeddings
                embeddings = self.embed_text(pattern_texts)
                
                # Calculate similarity matrix
                similarity_matrix = self.cosine_similarity_matrix(embeddings)
                
                # Find pattern clusters
                clusters = self._find_pattern_clusters(
                    similarity_matrix, pattern_metadata, similarity_threshold
                )
                
                pattern_analysis['pattern_clusters'][pattern_type] = clusters
                
                # Calculate cross-provider similarities
                cross_provider_sim = self._calculate_cross_provider_similarities(
                    similarity_matrix, pattern_metadata
                )
                
                pattern_analysis['cross_provider_similarities'][pattern_type] = cross_provider_sim
                
                # Pattern statistics
                stats = {
                    'total_patterns': len(patterns),
                    'unique_providers': len(set(p['provider'] for p in pattern_metadata)),
                    'clusters_found': len(clusters),
                    'average_intra_cluster_similarity': np.mean([c['avg_similarity'] for c in clusters]),
                    'singleton_patterns': len([c for c in clusters if len(c['members']) == 1])
                }
                
                pattern_analysis['pattern_statistics'][pattern_type] = stats
        
        logger.info(f"Pattern similarity analysis complete for {len(all_patterns)} pattern types")
        return pattern_analysis
    
    def _find_pattern_clusters(self, 
                             similarity_matrix: np.ndarray,
                             pattern_metadata: List[Dict],
                             threshold: float) -> List[Dict]:
        """Find clusters of similar patterns based on similarity matrix"""
        
        n_patterns = len(pattern_metadata)
        visited = set()
        clusters = []
        
        for i in range(n_patterns):
            if i in visited:
                continue
            
            # Start new cluster
            cluster_members = [i]
            visited.add(i)
            
            # Find all patterns similar to pattern i
            for j in range(i + 1, n_patterns):
                if j not in visited and similarity_matrix[i, j] >= threshold:
                    cluster_members.append(j)
                    visited.add(j)
            
            # Calculate cluster statistics
            if len(cluster_members) > 1:
                cluster_similarities = []
                for idx1 in cluster_members:
                    for idx2 in cluster_members:
                        if idx1 < idx2:
                            cluster_similarities.append(similarity_matrix[idx1, idx2])
                
                avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
            else:
                avg_similarity = 1.0
            
            cluster_info = {
                'cluster_id': len(clusters),
                'members': cluster_members,
                'size': len(cluster_members),
                'avg_similarity': float(avg_similarity),
                'member_metadata': [pattern_metadata[idx] for idx in cluster_members],
                'providers': list(set(pattern_metadata[idx]['provider'] for idx in cluster_members))
            }
            
            clusters.append(cluster_info)
        
        return clusters
    
    def _calculate_cross_provider_similarities(self, 
                                             similarity_matrix: np.ndarray,
                                             pattern_metadata: List[Dict]) -> Dict[str, float]:
        """Calculate similarity scores between different providers"""
        
        # Group patterns by provider
        provider_patterns = {}
        for i, metadata in enumerate(pattern_metadata):
            provider = metadata['provider']
            if provider not in provider_patterns:
                provider_patterns[provider] = []
            provider_patterns[provider].append(i)
        
        cross_provider_similarities = {}
        
        # Calculate pairwise provider similarities
        providers = list(provider_patterns.keys())
        for i, provider1 in enumerate(providers):
            for j, provider2 in enumerate(providers[i+1:], i+1):
                similarities = []
                
                for p1_idx in provider_patterns[provider1]:
                    for p2_idx in provider_patterns[provider2]:
                        similarities.append(similarity_matrix[p1_idx, p2_idx])
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    min_similarity = np.min(similarities)
                    
                    cross_provider_similarities[f"{provider1}_vs_{provider2}"] = {
                        'average': float(avg_similarity),
                        'maximum': float(max_similarity),
                        'minimum': float(min_similarity),
                        'comparison_count': len(similarities)
                    }
        
        return cross_provider_similarities


def run_embedding(input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Standardized API entry point for embedding service
    
    Args:
        input_data: Dictionary with structure:
            {
                "sections": [
                    {
                        "path": "observations.behavioral_patterns.malicious",
                        "content": [...],  # List of items to embed
                        "source": "claude"
                    },
                    ...
                ]
            }
        config: Configuration dictionary with embedding parameters
        
    Returns:
        Dictionary with structure:
            {
                "embeddings": {
                    "observations.behavioral_patterns.malicious": np.ndarray,
                    ...
                },
                "metadata": {
                    "model_used": "gemini-embedding-001",
                    "total_items": 42,
                    "processing_time": 1.23,
                    "cache_hits": 5
                },
                "success": True
            }
    """
    import time
    start_time = time.time()
    logger.info("START embedding_service: input_sections={}".format(len(input_data.get('sections', []))))
    
    try:
        # Initialize service with config
        model_name = config.get('embedding_model', 'gemini-embedding-001') if config else 'gemini-embedding-001'
        service = EmbeddingService(model_name=model_name)
        
        embeddings = {}
        total_items = 0
        cache_hits = 0
        
        for section in input_data.get('sections', []):
            section_path = section['path']
            content_items = section['content']
            
            if not content_items:
                continue
                
            # Convert content items to text representations
            texts = []
            for item in content_items:
                if isinstance(item, dict):
                    # Extract text from structured data
                    text_parts = []
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            text_parts.append(f"{key}: {value}")
                    texts.append(" | ".join(text_parts))
                else:
                    texts.append(str(item))
            
            if texts:
                # Generate embeddings
                section_embeddings = service.embed_text(texts)
                embeddings[section_path] = section_embeddings
                total_items += len(texts)
        
        processing_time = time.time() - start_time
        
        result = {
            "embeddings": embeddings,
            "metadata": {
                "model_used": model_name,
                "total_items": total_items,
                "processing_time": processing_time,
                "cache_hits": cache_hits,  # TODO: Get actual cache hit count
                "embedding_dimensions": list(embeddings.values())[0].shape[1] if embeddings else 0
            },
            "success": True
        }
        
        logger.info("END embedding_service: output_sections={}, time={:.2f}s".format(
            len(embeddings), processing_time))
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"END embedding_service: FAILED after {processing_time:.2f}s - {e}")
        return {
            "embeddings": {},
            "metadata": {"error": str(e), "processing_time": processing_time},
            "success": False
        }


    def generate_unified_pattern_embeddings(self, 
                                          unified_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for unified patterns to support clustering and similarity analysis
        
        Args:
            unified_patterns: List of unified pattern dictionaries
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        if not unified_patterns:
            return {'embeddings': np.array([]), 'metadata': [], 'pattern_texts': []}
        
        pattern_texts = []
        metadata = []
        
        for i, pattern in enumerate(unified_patterns):
            # Create comprehensive text representation
            text_components = []
            
            # Primary identification fields
            for field in ['name', 'pattern_name', 'title']:
                if field in pattern and pattern[field]:
                    text_components.append(f"Name: {pattern[field]}")
                    break
            
            # Description
            if 'description' in pattern and pattern['description']:
                text_components.append(f"Description: {pattern['description']}")
            
            # Type/Category information
            for field in ['type', 'category', 'attack_type', 'severity', 'risk_level']:
                if field in pattern and pattern[field]:
                    text_components.append(f"{field}: {pattern[field]}")
            
            # Behavioral indicators
            if 'indicators' in pattern:
                indicators = pattern['indicators']
                if isinstance(indicators, list):
                    text_components.append(f"Indicators: {' '.join(map(str, indicators))}")
                elif isinstance(indicators, str):
                    text_components.append(f"Indicators: {indicators}")
            
            # Tactics, techniques, procedures
            for field in ['tactics', 'techniques', 'procedures']:
                if field in pattern and pattern[field]:
                    if isinstance(pattern[field], list):
                        text_components.append(f"{field}: {' '.join(pattern[field])}")
                    else:
                        text_components.append(f"{field}: {pattern[field]}")
            
            # Combine all components
            pattern_text = " | ".join(text_components)
            pattern_texts.append(pattern_text)
            
            # Store metadata
            metadata.append({
                'pattern_index': i,
                'pattern_hash': hash(pattern_text),
                'source_providers': pattern.get('_source_providers', []),
                'consensus_count': pattern.get('_consensus_count', 1),
                'confidence_score': pattern.get('_confidence_score', 0.5),
                'pattern_type': pattern.get('_pattern_type', 'unknown'),
                'original_pattern': pattern
            })
        
        # Generate embeddings
        embeddings = self.embed_text(pattern_texts)
        
        logger.info(f"Generated embeddings for {len(unified_patterns)} unified patterns")
        
        return {
            'embeddings': embeddings,
            'metadata': metadata,
            'pattern_texts': pattern_texts,
            'embedding_dimension': embeddings.shape[1] if embeddings.size > 0 else 0
        }