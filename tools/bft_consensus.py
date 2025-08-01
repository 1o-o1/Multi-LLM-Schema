"""
Byzantine Fault Tolerance Consensus for Multi-Agent Schema Consensus
Adapted BFT algorithms (PBFT, Hashgraph, Algorand) for schema field consensus
"""

import numpy as np
import logging
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
from enum import Enum

logger = logging.getLogger(__name__)

class ConsensusPhase(Enum):
    PREPARE = "prepare"
    COMMIT = "commit"
    FINALIZE = "finalize"

@dataclass
class BFTMessage:
    """BFT protocol message"""
    sender: str
    phase: ConsensusPhase
    field_proposal: Dict[str, Any]
    sequence_number: int
    timestamp: float
    signature: str
    
class BFTConsensus:
    """Byzantine Fault Tolerance consensus engine for schema fields"""
    
    def __init__(self, fault_tolerance: float = 0.33):
        """
        Initialize BFT consensus engine
        
        Args:
            fault_tolerance: Maximum fraction of faulty nodes (default 1/3 for BFT)
        """
        self.fault_tolerance = fault_tolerance
        self.nodes = {}  # provider_name -> node_info
        self.messages = []
        self.consensus_log = []
        self.field_proposals = {}
        self.consensus_state = {}
        
    def register_provider_nodes(self, provider_results: Dict[str, Dict]):
        """
        Register provider nodes for consensus
        
        Args:
            provider_results: Dictionary of provider results
        """
        for provider, result in provider_results.items():
            self.nodes[provider] = {
                'id': provider,
                'reputation': self._calculate_provider_reputation(provider, result),
                'stake': 1.0,  # Equal stake for all providers initially
                'last_active': time.time(),
                'message_count': 0,
                'byzantine_score': 0.0
            }
        
        logger.info(f"Registered {len(self.nodes)} provider nodes for BFT consensus")
    
    def _calculate_provider_reputation(self, provider: str, result: Dict) -> float:
        """Calculate initial reputation score for a provider"""
        try:
            score = 1.0  # Base reputation
            
            # Safety check: ensure result is a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Provider {provider} result is not a dict: {type(result)}")
                return score
            
            # Check result quality indicators - Stage 2 field-based format
            if 'parsed_json' in result and isinstance(result['parsed_json'], dict):
                fields = result['parsed_json'].get('fields', {})
                
                if isinstance(fields, dict):
                    # Reward comprehensive results
                    if len(fields) > 10:
                        score += 0.2
                    
                    # Check for complete field definitions
                    complete_fields = 0
                    for field_data in fields.values():
                        if isinstance(field_data, dict):
                            if (field_data.get('description') and 
                                field_data.get('type') and 
                                any(field_data.get(fw) for fw in ['OCSF', 'ECS', 'OSSEM'])):
                                complete_fields += 1
                    
                    completeness = complete_fields / len(fields) if fields else 0
                    score += 0.3 * completeness
            
            # Check result quality indicators - Stage 5 cluster-based format (from consensus orchestrator)
            elif 'confidence' in result and 'timestamp' in result:
                # This is likely cluster-based data from Stage 5
                confidence = result.get('confidence', 0.5)
                score += 0.3 * confidence  # Reward higher confidence
                
                # Check if there's meaningful content
                if 'parsed_json' in result and 'fields' in result['parsed_json']:
                    content_length = len(str(result['parsed_json']['fields']))
                    if content_length > 100:  # Non-trivial content
                        score += 0.2
            
            return min(2.0, score)  # Cap at 2.0
            
        except Exception as e:
            logger.error(f"Reputation calculation failed for {provider}: {e}")
            return 1.0
    
    def pbft_consensus_round(self, 
                           field_proposals: Dict[str, Dict[str, Any]],
                           round_number: int) -> Dict[str, Any]:
        """
        Execute Practical Byzantine Fault Tolerance consensus round
        
        Args:
            field_proposals: Dictionary of field proposals from each provider
            round_number: Current consensus round number
            
        Returns:
            Consensus result for this round
        """
        try:
            consensus_result = {
                'round': round_number,
                'phase_results': {},
                'final_consensus': {},
                'participation': {},
                'byzantine_detected': []
            }
            
            # Phase 1: Prepare phase
            prepare_messages = self._pbft_prepare_phase(field_proposals, round_number)
            consensus_result['phase_results']['prepare'] = prepare_messages
            
            # Phase 2: Commit phase  
            commit_messages = self._pbft_commit_phase(prepare_messages, round_number)
            consensus_result['phase_results']['commit'] = commit_messages
            
            # Phase 3: Finalize phase
            final_consensus = self._pbft_finalize_phase(commit_messages, round_number)
            consensus_result['final_consensus'] = final_consensus
            
            # Calculate participation metrics
            consensus_result['participation'] = self._calculate_participation_metrics(
                prepare_messages, commit_messages
            )
            
            # Detect Byzantine behavior
            byzantine_nodes = self._detect_byzantine_behavior(
                field_proposals, prepare_messages, commit_messages
            )
            consensus_result['byzantine_detected'] = byzantine_nodes
            
            logger.info(f"PBFT round {round_number} completed with {len(final_consensus)} consensus fields")
            return consensus_result
            
        except Exception as e:
            logger.error(f"PBFT consensus round failed: {e}")
            return {'round': round_number, 'error': str(e)}
    
    def _pbft_prepare_phase(self, 
                          field_proposals: Dict[str, Dict[str, Any]], 
                          round_number: int) -> List[BFTMessage]:
        """Execute PBFT prepare phase"""
        prepare_messages = []
        
        for provider, proposals in field_proposals.items():
            if provider not in self.nodes:
                continue
                
            for field_name, field_data in proposals.items():
                message = BFTMessage(
                    sender=provider,
                    phase=ConsensusPhase.PREPARE,
                    field_proposal={field_name: field_data},
                    sequence_number=round_number,
                    timestamp=time.time(),
                    signature=self._generate_message_signature(provider, field_data)
                )
                prepare_messages.append(message)
        
        return prepare_messages
    
    def _pbft_commit_phase(self, 
                         prepare_messages: List[BFTMessage], 
                         round_number: int) -> List[BFTMessage]:
        """Execute PBFT commit phase"""
        commit_messages = []
        
        # Group prepare messages by field
        field_prepares = defaultdict(list)
        for msg in prepare_messages:
            for field_name, field_data in msg.field_proposal.items():
                field_prepares[field_name].append((msg.sender, field_data))
        
        # Generate commit messages for fields with sufficient support
        min_support = max(1, int(len(self.nodes) * (1 - self.fault_tolerance)))
        
        for field_name, prepares in field_prepares.items():
            if len(prepares) >= min_support:
                # Choose most supported field definition
                field_variants = defaultdict(list)
                for sender, field_data in prepares:
                    field_key = self._generate_field_signature(field_data)
                    field_variants[field_key].append((sender, field_data))
                
                # Select variant with most support
                best_variant = max(field_variants.items(), key=lambda x: len(x[1]))
                consensus_field_data = best_variant[1][0][1]  # Take first field_data of best variant
                
                # Generate commit messages from supporting nodes
                for sender, _ in best_variant[1]:
                    message = BFTMessage(
                        sender=sender,
                        phase=ConsensusPhase.COMMIT,
                        field_proposal={field_name: consensus_field_data},
                        sequence_number=round_number,
                        timestamp=time.time(),
                        signature=self._generate_message_signature(sender, consensus_field_data)
                    )
                    commit_messages.append(message)
        
        return commit_messages
    
    def _pbft_finalize_phase(self, 
                           commit_messages: List[BFTMessage], 
                           round_number: int) -> Dict[str, Any]:
        """Execute PBFT finalize phase"""
        final_consensus = {}
        
        # Group commit messages by field
        field_commits = defaultdict(list)
        for msg in commit_messages:
            for field_name, field_data in msg.field_proposal.items():
                field_commits[field_name].append((msg.sender, field_data))
        
        # Finalize fields with sufficient commit support
        min_commits = max(1, int(len(self.nodes) * (2/3)))  # BFT requires 2/3 majority
        
        for field_name, commits in field_commits.items():
            if len(commits) >= min_commits:
                # Generate consensus field definition
                consensus_field = self._merge_field_definitions([data for _, data in commits])
                consensus_field['consensus_support'] = len(commits)
                consensus_field['supporting_providers'] = [sender for sender, _ in commits]
                
                final_consensus[field_name] = consensus_field
        
        return final_consensus
    
    def _generate_message_signature(self, sender: str, data: Any) -> str:
        """Generate cryptographic signature for message"""
        message_content = f"{sender}:{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(message_content.encode()).hexdigest()[:16]
    
    def _generate_field_signature(self, field_data: Dict[str, Any]) -> str:
        """Generate signature for field data to detect variants"""
        # Create signature based on core field properties
        signature_data = {
            'type': field_data.get('type', ''),
            'description_hash': hashlib.md5(
                field_data.get('description', '').encode()
            ).hexdigest()[:8],
            'framework_mappings': {
                fw: field_data.get(fw) for fw in ['OCSF', 'ECS', 'OSSEM'] 
                if field_data.get(fw) and field_data.get(fw) != 'null'
            }
        }
        
        return hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def _merge_field_definitions(self, field_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple field definitions into consensus definition"""
        if not field_definitions:
            return {}
        
        if len(field_definitions) == 1:
            return field_definitions[0].copy()
        
        merged = {}
        
        # Merge by majority vote or highest importance
        for key in ['type', 'description', 'importance', 'OCSF', 'ECS', 'OSSEM']:
            values = []
            for field_def in field_definitions:
                if key in field_def and field_def[key] and field_def[key] != 'null':
                    values.append(field_def[key])
            
            if values:
                if key == 'importance':
                    # Take maximum importance
                    merged[key] = max(values)
                elif key == 'description':
                    # Take longest description (most informative)
                    merged[key] = max(values, key=len)
                else:
                    # Take most common value
                    value_counts = Counter(values)
                    merged[key] = value_counts.most_common(1)[0][0]
        
        return merged
    
    def _calculate_participation_metrics(self, 
                                       prepare_messages: List[BFTMessage],
                                       commit_messages: List[BFTMessage]) -> Dict[str, float]:
        """Calculate node participation metrics"""
        prepare_participation = Counter(msg.sender for msg in prepare_messages)
        commit_participation = Counter(msg.sender for msg in commit_messages)
        
        total_nodes = len(self.nodes)
        
        metrics = {
            'prepare_participation_rate': len(prepare_participation) / total_nodes,
            'commit_participation_rate': len(commit_participation) / total_nodes,
            'average_prepare_messages': np.mean(list(prepare_participation.values())) if prepare_participation else 0,
            'average_commit_messages': np.mean(list(commit_participation.values())) if commit_participation else 0,
            'node_participation': {}
        }
        
        for node_id in self.nodes:
            metrics['node_participation'][node_id] = {
                'prepare_messages': prepare_participation.get(node_id, 0),
                'commit_messages': commit_participation.get(node_id, 0),
                'total_messages': prepare_participation.get(node_id, 0) + commit_participation.get(node_id, 0)
            }
        
        return metrics
    
    def _detect_byzantine_behavior(self, 
                                 field_proposals: Dict[str, Dict[str, Any]],
                                 prepare_messages: List[BFTMessage],
                                 commit_messages: List[BFTMessage]) -> List[str]:
        """Detect potential Byzantine behavior patterns"""
        byzantine_nodes = []
        
        try:
            # Check for inconsistent messaging
            prepare_senders = set(msg.sender for msg in prepare_messages)
            commit_senders = set(msg.sender for msg in commit_messages)
            
            # Nodes that prepare but don't commit (potential Byzantine behavior)
            inconsistent_nodes = prepare_senders - commit_senders
            
            for node in inconsistent_nodes:
                prepare_count = len([msg for msg in prepare_messages if msg.sender == node])
                commit_count = len([msg for msg in commit_messages if msg.sender == node])
                
                # Flag nodes with high prepare/commit ratio as potentially Byzantine
                if prepare_count > 0 and commit_count / prepare_count < 0.5:
                    byzantine_nodes.append(node)
                    self.nodes[node]['byzantine_score'] += 0.3
            
            # Check for conflicting field proposals
            for node in self.nodes:
                if node in field_proposals:
                    node_proposals = field_proposals[node]
                    
                    # Compare with consensus to detect outliers
                    outlier_count = 0
                    for field_name, field_data in node_proposals.items():
                        # Check if this field conflicts with majority
                        other_proposals = []
                        for other_node, other_field_proposals in field_proposals.items():
                            if other_node != node and field_name in other_field_proposals:
                                other_proposals.append(other_field_proposals[field_name])
                        
                        if other_proposals:
                            # Check for significant differences
                            if self._is_field_outlier(field_data, other_proposals):
                                outlier_count += 1
                    
                    # Flag nodes with many outlier proposals
                    total_proposals = len(node_proposals)
                    if total_proposals > 0 and outlier_count / total_proposals > 0.4:
                        if node not in byzantine_nodes:
                            byzantine_nodes.append(node)
                        self.nodes[node]['byzantine_score'] += 0.4
            
            logger.info(f"Detected {len(byzantine_nodes)} potentially Byzantine nodes")
            return byzantine_nodes
            
        except Exception as e:
            logger.error(f"Byzantine detection failed: {e}")
            return []
    
    def _is_field_outlier(self, field_data: Dict[str, Any], other_proposals: List[Dict[str, Any]]) -> bool:
        """Check if a field proposal is an outlier compared to others"""
        try:
            # Check type consistency
            field_type = field_data.get('type', '').lower()
            other_types = [p.get('type', '').lower() for p in other_proposals]
            
            if field_type and other_types:
                type_counter = Counter(other_types)
                most_common_type = type_counter.most_common(1)[0][0]
                
                if field_type != most_common_type:
                    return True
            
            # Check framework mapping consistency
            for framework in ['OCSF', 'ECS', 'OSSEM']:
                field_mapping = field_data.get(framework)
                other_mappings = [p.get(framework) for p in other_proposals 
                                if p.get(framework) and p.get(framework) != 'null']
                
                if field_mapping and other_mappings:
                    mapping_counter = Counter(other_mappings)
                    if mapping_counter and field_mapping not in mapping_counter:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return False
    
    def hashgraph_inspired_consensus(self, 
                                   provider_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Hashgraph-inspired consensus using gossip protocol and virtual voting
        
        Args:
            provider_results: Dictionary of provider results
            
        Returns:
            Consensus result using hashgraph principles
        """
        try:
            consensus_result = {
                'algorithm': 'hashgraph_inspired',
                'events': [],
                'virtual_votes': {},
                'consensus_fields': {},
                'timestamp_consensus': {}
            }
            
            # Create events from provider results
            events = self._create_hashgraph_events(provider_results)
            consensus_result['events'] = events
            
            # Perform virtual voting
            virtual_votes = self._hashgraph_virtual_voting(events)
            consensus_result['virtual_votes'] = virtual_votes
            
            # Determine consensus based on virtual voting
            consensus_fields = self._determine_hashgraph_consensus(virtual_votes)
            consensus_result['consensus_fields'] = consensus_fields
            
            logger.info(f"Hashgraph consensus completed with {len(consensus_fields)} consensus fields")
            return consensus_result
            
        except Exception as e:
            logger.error(f"Hashgraph consensus failed: {e}")
            return {'error': str(e)}
    
    def _create_hashgraph_events(self, provider_results: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Create hashgraph events from provider results"""
        events = []
        
        for i, (provider, result) in enumerate(provider_results.items()):
            if 'parsed_json' not in result or 'fields' not in result['parsed_json']:
                continue
            
            event = {
                'id': f"event_{i}",
                'creator': provider,
                'timestamp': time.time() + i * 0.1,  # Simulate timing
                'self_parent': f"event_{i-1}" if i > 0 else None,
                'other_parent': None,  # Simplified - would be from gossip
                'payload': result['parsed_json']['fields'],
                'round_created': 1,
                'witness': True,
                'famous': None  # To be determined by virtual voting
            }
            events.append(event)
        
        return events
    
    def _hashgraph_virtual_voting(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Perform hashgraph virtual voting on events"""
        virtual_votes = {}
        
        # Simplified virtual voting - in real hashgraph this would be more complex
        for event in events:
            event_id = event['id']
            virtual_votes[event_id] = {
                'votes_for': [],
                'votes_against': [],
                'strongly_see': [],
                'can_see': []
            }
            
            # Simulate virtual voting based on event quality
            payload = event.get('payload', {})
            quality_score = len(payload)  # Simple quality metric
            
            # Other events "vote" based on quality
            for other_event in events:
                if other_event['id'] != event_id:
                    other_quality = len(other_event.get('payload', {}))
                    
                    if quality_score >= other_quality * 0.8:  # Within 80% quality
                        virtual_votes[event_id]['votes_for'].append(other_event['creator'])
                    else:
                        virtual_votes[event_id]['votes_against'].append(other_event['creator'])
        
        return virtual_votes
    
    def _determine_hashgraph_consensus(self, virtual_votes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Determine consensus based on virtual voting results"""
        consensus_fields = {}
        
        # Collect all field proposals from events with positive votes
        field_proposals = defaultdict(list)
        
        for event_id, votes in virtual_votes.items():
            if len(votes['votes_for']) > len(votes['votes_against']):
                # This event achieved consensus
                # Find the corresponding event
                for event in self.consensus_log:
                    if event.get('id') == event_id:
                        payload = event.get('payload', {})
                        for field_name, field_data in payload.items():
                            field_proposals[field_name].append(field_data)
                        break
        
        # Merge field proposals to create final consensus
        for field_name, proposals in field_proposals.items():
            if proposals:
                consensus_field = self._merge_field_definitions(proposals)
                consensus_field['consensus_method'] = 'hashgraph_virtual_voting'
                consensus_field['proposal_count'] = len(proposals)
                consensus_fields[field_name] = consensus_field
        
        return consensus_fields
    
    def multi_algorithm_consensus(self, 
                                provider_results: Dict[str, Dict],
                                algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run multiple consensus algorithms and combine results
        
        Args:
            provider_results: Dictionary of provider results
            algorithms: List of algorithms to run (default: all)
            
        Returns:
            Combined consensus results
        """
        if algorithms is None:
            algorithms = ['pbft', 'hashgraph']
        
        results = {}
        
        try:
            # Register nodes
            self.register_provider_nodes(provider_results)
            
            # Extract field proposals
            field_proposals = {}
            for provider, result in provider_results.items():
                try:
                    # Safety check: ensure result is a dictionary
                    if not isinstance(result, dict):
                        logger.warning(f"Provider {provider} result is not a dict: {type(result)}")
                        continue
                    
                    # Stage 2: Traditional field-based format
                    if 'parsed_json' in result and isinstance(result['parsed_json'], dict):
                        fields = result['parsed_json'].get('fields', {})
                        if isinstance(fields, dict) and fields:
                            field_proposals[provider] = fields
                            continue
                    
                    # Stage 5: Cluster-based format - create synthetic field from content
                    if 'parsed_json' in result and 'fields' in result['parsed_json']:
                        # This is the mock structure created by consensus orchestrator
                        synthetic_fields = result['parsed_json']['fields']
                        if isinstance(synthetic_fields, dict):
                            field_proposals[provider] = synthetic_fields
                            continue
                    
                    # Fallback: create a generic field from any available content
                    content_str = str(result)[:200]  # Truncate for safety
                    field_proposals[provider] = {
                        'generic_content': {
                            'type': 'string',
                            'description': f'Content from {provider}',
                            'content': content_str,
                            'OCSF': 'generic',
                            'ECS': 'text',
                            'OSSEM': 'content'
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error extracting fields for provider {provider}: {e}")
                    continue
            
            # Run each algorithm
            if 'pbft' in algorithms:
                pbft_result = self.pbft_consensus_round(field_proposals, 1)
                results['pbft'] = pbft_result
            
            if 'hashgraph' in algorithms:
                hashgraph_result = self.hashgraph_inspired_consensus(provider_results)
                results['hashgraph'] = hashgraph_result
            
            # Combine results using majority consensus
            combined_consensus = self._combine_algorithm_results(results)
            
            return {
                'individual_results': results,
                'combined_consensus': combined_consensus,
                'algorithms_used': algorithms,
                'total_providers': len(provider_results)
            }
            
        except Exception as e:
            logger.error(f"Multi-algorithm consensus failed: {e}")
            return {'error': str(e)}
    
    def _combine_algorithm_results(self, algorithm_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine results from multiple consensus algorithms"""
        combined = {}
        
        # Collect all consensus fields from all algorithms
        all_consensus_fields = {}
        
        for algorithm, result in algorithm_results.items():
            if algorithm == 'pbft' and 'final_consensus' in result:
                for field_name, field_data in result['final_consensus'].items():
                    if field_name not in all_consensus_fields:
                        all_consensus_fields[field_name] = []
                    all_consensus_fields[field_name].append((algorithm, field_data))
            
            elif algorithm == 'hashgraph' and 'consensus_fields' in result:
                for field_name, field_data in result['consensus_fields'].items():
                    if field_name not in all_consensus_fields:
                        all_consensus_fields[field_name] = []
                    all_consensus_fields[field_name].append((algorithm, field_data))
        
        # Create final consensus based on algorithm agreement
        for field_name, algorithm_results in all_consensus_fields.items():
            if len(algorithm_results) > 1:
                # Multiple algorithms agree on this field
                field_definitions = [field_data for _, field_data in algorithm_results]
                consensus_field = self._merge_field_definitions(field_definitions)
                consensus_field['algorithm_agreement'] = [alg for alg, _ in algorithm_results]
                consensus_field['confidence'] = len(algorithm_results) / len(self.nodes)
                combined[field_name] = consensus_field
            else:
                # Single algorithm result
                algorithm, field_data = algorithm_results[0]
                field_data['algorithm_source'] = algorithm
                field_data['confidence'] = 0.5  # Lower confidence for single algorithm
                combined[field_name] = field_data
        
        return combined
    
    def achieve_consensus(self, provider_data: Dict[str, Dict], consensus_threshold: float = 0.5) -> Dict[str, Any]:
        """
        High-level method to achieve consensus using BFT protocols
        
        Args:
            provider_data: Dictionary mapping provider names to their data
            consensus_threshold: Minimum threshold for consensus agreement
            
        Returns:
            Dictionary with consensus results and metadata
        """
        try:
            # Register provider nodes
            self.register_provider_nodes(provider_data)
            
            if len(self.nodes) < 2:
                return {
                    'consensus_achieved': False,
                    'error': 'Insufficient providers for consensus',
                    'provider_count': len(self.nodes)
                }
            
            # Run multi-algorithm consensus for robustness
            consensus_result = self.multi_algorithm_consensus(provider_data)
            
            # Check if consensus threshold is met
            consensus_fields = {k: v for k, v in consensus_result.items() 
                              if v.get('confidence', 0) >= consensus_threshold}
            
            consensus_achieved = len(consensus_fields) > 0
            
            return {
                'consensus_achieved': consensus_achieved,
                'consensus_fields': consensus_fields,
                'total_fields': len(consensus_result),
                'consensus_ratio': len(consensus_fields) / max(len(consensus_result), 1),
                'threshold_used': consensus_threshold,
                'algorithms_used': ['pbft', 'hashgraph'],
                'provider_count': len(self.nodes),
                'byzantine_nodes_detected': len([node for node in self.nodes.values() 
                                               if node.get('byzantine_detected', False)])
            }
            
        except Exception as e:
            logger.error(f"BFT consensus failed: {e}")
            return {
                'consensus_achieved': False,
                'error': str(e),
                'provider_count': len(provider_data)
            }


def run_bft_consensus(input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Standardized API entry point for BFT consensus
    
    Args:
        input_data: Dictionary with structure:
            {
                "provider_data": {
                    "claude": {
                        "observations.behavioral_patterns.malicious": [...],
                        ...
                    },
                    "gemini": {...},
                    "openai": {...}
                }
            }
        config: Configuration dictionary with BFT parameters
        
    Returns:
        Dictionary with structure:
            {
                "consensus_results": {
                    "observations.behavioral_patterns.malicious": {
                        "consensus_items": [...],
                        "confidence": 0.85,
                        "supporting_models": ["claude", "gemini"]
                    },
                    ...
                },
                "metadata": {
                    "algorithm": "bft_consensus", 
                    "providers_count": 3,
                    "processing_time": 2.34,
                    "byzantine_nodes_detected": 0
                },
                "success": True
            }
    """
    import time
    start_time = time.time()
    logger.info("START bft_consensus: providers={}".format(len(input_data.get('provider_data', {}))))
    
    try:
        # Initialize BFT consensus with config
        fault_tolerance = config.get('bft_fault_tolerance', 0.33) if config else 0.33
        bft = BFTConsensus(fault_tolerance=fault_tolerance)
        
        provider_data = input_data.get('provider_data', {})
        
        if len(provider_data) < 2:
            raise ValueError("BFT consensus requires at least 2 providers")
        
        # Run consensus
        consensus_result = bft.achieve_consensus(provider_data, 
                                               consensus_threshold=config.get('consensus_threshold', 0.5) if config else 0.5)
        
        processing_time = time.time() - start_time
        
        # Format results for standardized output
        consensus_results = {}
        if consensus_result.get('consensus_achieved'):
            consensus_fields = consensus_result.get('consensus_fields', {})
            for field_name, field_data in consensus_fields.items():
                consensus_results[field_name] = {
                    "consensus_items": [field_data],
                    "confidence": field_data.get('confidence', 0.5),
                    "supporting_models": field_data.get('supporting_providers', [])
                }
        
        result = {
            "consensus_results": consensus_results,
            "metadata": {
                "algorithm": "bft_consensus",
                "providers_count": len(provider_data),
                "processing_time": processing_time,
                "byzantine_nodes_detected": consensus_result.get('byzantine_nodes_detected', 0),
                "consensus_achieved": consensus_result.get('consensus_achieved', False),
                "consensus_ratio": consensus_result.get('consensus_ratio', 0.0)
            },
            "success": True
        }
        
        logger.info("END bft_consensus: consensus_fields={}, time={:.2f}s".format(
            len(consensus_results), processing_time))
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"END bft_consensus: FAILED after {processing_time:.2f}s - {e}")
        return {
            "consensus_results": {},
            "metadata": {"error": str(e), "processing_time": processing_time},
            "success": False
        }