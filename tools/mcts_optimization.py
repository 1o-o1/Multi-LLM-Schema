"""
Monte Carlo Tree Search Optimization for Multi-Agent Schema Consensus
MCTS for exploring and optimizing field alignment strategies and schema choices
"""

import numpy as np
import logging
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import copy

logger = logging.getLogger(__name__)

class ActionType(Enum):
    MERGE_FIELDS = "merge_fields"
    SPLIT_FIELD = "split_field" 
    ALIGN_FRAMEWORKS = "align_frameworks"
    ADJUST_IMPORTANCE = "adjust_importance"
    RESOLVE_CONFLICT = "resolve_conflict"

@dataclass
class MCTSAction:
    """Action in MCTS for schema optimization"""
    action_type: ActionType
    field_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    expected_reward: float = 0.0

@dataclass
class MCTSNode:
    """Node in MCTS tree"""
    state: Dict[str, Any]
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    actions: List[MCTSAction] = field(default_factory=list)
    visit_count: int = 0
    total_reward: float = 0.0
    untried_actions: List[MCTSAction] = field(default_factory=list)
    is_terminal: bool = False
    
    def ucb1_score(self, c: float = 1.41) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent is None or self.parent.visit_count == 0:
            return self.total_reward / self.visit_count
        
        exploitation = self.total_reward / self.visit_count
        exploration = c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        
        return exploitation + exploration

class MCTSOptimization:
    """Monte Carlo Tree Search optimization engine for schema consensus"""
    
    def __init__(self, exploration_constant: float = 1.41, max_iterations: int = 1000):
        """
        Initialize MCTS optimization engine
        
        Args:
            exploration_constant: UCB1 exploration parameter
            max_iterations: Maximum number of MCTS iterations
        """
        self.c = exploration_constant
        self.max_iterations = max_iterations
        self.root = None
        self.best_path = []
        self.evaluation_cache = {}
        
    def optimize_schema_consensus(self, 
                                provider_results: Dict[str, Dict],
                                optimization_objectives: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Optimize schema consensus using MCTS
        
        Args:
            provider_results: Dictionary of provider results
            optimization_objectives: Weights for different objectives
            
        Returns:
            Optimized consensus schema and search statistics
        """
        try:
            if optimization_objectives is None:
                optimization_objectives = {
                    'completeness': 0.3,
                    'consistency': 0.3,
                    'framework_alignment': 0.2,
                    'semantic_coherence': 0.2
                }
            
            # Initialize root state
            initial_state = self._create_initial_state(provider_results)
            self.root = MCTSNode(state=initial_state)
            
            # Generate initial actions
            self.root.untried_actions = self._generate_possible_actions(initial_state)
            
            start_time = time.time()
            iteration = 0
            
            # MCTS main loop
            while iteration < self.max_iterations and time.time() - start_time < 60:  # 60 second timeout
                # Selection - traverse tree using UCB1
                selected_node = self._select_node(self.root)
                
                # Expansion - add new child node if possible
                if selected_node.untried_actions and not selected_node.is_terminal:
                    selected_node = self._expand_node(selected_node)
                
                # Simulation - random rollout from selected node
                reward = self._simulate_rollout(selected_node, optimization_objectives)
                
                # Backpropagation - update node statistics
                self._backpropagate(selected_node, reward)
                
                iteration += 1
                
                if iteration % 100 == 0:
                    logger.info(f"MCTS iteration {iteration}, best reward: {self._get_best_child_reward()}")
            
            # Extract best solution
            best_solution = self._extract_best_solution()
            
            search_statistics = {
                'iterations': iteration,
                'search_time': time.time() - start_time,
                'tree_depth': self._calculate_tree_depth(),
                'nodes_explored': self._count_nodes(),
                'best_reward': self._get_best_child_reward(),
                'convergence_iteration': self._find_convergence_point()
            }
            
            logger.info(f"MCTS optimization completed: {iteration} iterations, best reward: {search_statistics['best_reward']:.4f}")
            
            return {
                'optimized_schema': best_solution['schema'],
                'action_sequence': best_solution['actions'],
                'final_reward': best_solution['reward'],
                'search_statistics': search_statistics,
                'optimization_objectives': optimization_objectives
            }
            
        except Exception as e:
            logger.error(f"MCTS optimization failed: {e}")
            return {'error': str(e)}
    
    def _create_initial_state(self, provider_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create initial state for MCTS search"""
        state = {
            'items': {},
            'provider_results': provider_results,
            'item_mappings': {},
            'conflicts': [],
            'alignment_groups': {},
            'applied_actions': [],
            'target_key_path': None
        }
        
        # Extract all items from a generalized path
        # This will be set during optimization call
        all_items = {}
        for provider, result in provider_results.items():
            # Try common paths for different data types
            possible_paths = [
                'parsed_json.fields',
                'parsed_json.malicious_patterns', 
                'parsed_json.behavioral_patterns',
                'parsed_json.patterns',
                'parsed_json'
            ]
            
            for path in possible_paths:
                items = self._extract_items_from_path(result, path)
                if items:
                    for item_name, item_data in items.items():
                        item_id = f"{provider}::{item_name}"
                        all_items[item_id] = {
                            'provider': provider,
                            'original_name': item_name,
                            'data': item_data.copy(),
                            'source_path': path
                        }
                    break  # Use first path that has data
        
        state['items'] = all_items
        
        # Identify initial conflicts
        state['conflicts'] = self._identify_conflicts(all_items)
        
        return state
    
    def _extract_items_from_path(self, result: Dict, path: str) -> Dict[str, Any]:
        """Extract items from nested path in result"""
        try:
            keys = path.split('.')
            current = result
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return {}
            
            # Handle different data structures
            if isinstance(current, dict):
                return current
            elif isinstance(current, list):
                # Convert list to dict with index keys
                items = {}
                for i, item in enumerate(current):
                    if isinstance(item, dict):
                        # Try to get name from common fields
                        name = None
                        for name_field in ['name', 'pattern_name', 'field_name', 'title']:
                            if name_field in item:
                                name = item[name_field]
                                break
                        
                        if not name:
                            name = f"item_{i}"
                        
                        items[name] = item
                    else:
                        items[f"item_{i}"] = item
                return items
            else:
                return {}
                
        except Exception:
            return {}
    
    def _identify_conflicts(self, items: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Identify conflicts in initial item set"""
        conflicts = []
        
        # Group items by similar names
        item_groups = defaultdict(list)
        for item_id, item_info in items.items():
            item_name = item_info['original_name'].lower()
            # Simple grouping by exact name match
            item_groups[item_name].append(item_id)
        
        # Check for conflicts within groups
        for group_name, item_ids in item_groups.items():
            if len(item_ids) > 1:
                # Check for type/category conflicts
                types = set()
                categories = set()
                severities = set()
                
                for item_id in item_ids:
                    item_data = items[item_id]['data']
                    
                    # Check various type fields
                    for type_field in ['type', 'data_type', 'attack_type', 'category']:
                        if type_field in item_data and item_data[type_field]:
                            types.add(str(item_data[type_field]).lower())
                    
                    # Check severity/risk levels  
                    for sev_field in ['severity', 'risk_level', 'priority']:
                        if sev_field in item_data and item_data[sev_field]:
                            severities.add(str(item_data[sev_field]).lower())
                
                if len(types) > 1:
                    conflicts.append({
                        'type': 'type_conflict',
                        'item_name': group_name,
                        'item_ids': item_ids,
                        'conflicting_types': list(types)
                    })
                
                if len(severities) > 1:
                    conflicts.append({
                        'type': 'severity_conflict',
                        'item_name': group_name,
                        'item_ids': item_ids,
                        'conflicting_severities': list(severities)
                    })
                
                # Check for framework mapping conflicts (for fields)
                framework_conflicts = defaultdict(set)
                for item_id in item_ids:
                    item_data = items[item_id]['data']
                    for framework in ['OCSF', 'ECS', 'OSSEM']:
                        mapping = item_data.get(framework)
                        if mapping and mapping != 'null':
                            framework_conflicts[framework].add(mapping)
                
                for framework, mappings in framework_conflicts.items():
                    if len(mappings) > 1:
                        conflicts.append({
                            'type': 'framework_conflict',
                            'item_name': group_name,
                            'framework': framework,
                            'item_ids': item_ids,
                            'conflicting_mappings': list(mappings)
                        })
        
        return conflicts
    
    def _generate_possible_actions(self, state: Dict[str, Any]) -> List[MCTSAction]:
        """Generate possible actions from current state"""
        actions = []
        items = state['items']
        conflicts = state['conflicts']
        
        # Actions to resolve conflicts
        for conflict in conflicts:
            if conflict['type'] in ['type_conflict', 'severity_conflict']:
                # Action to merge conflicting items
                action = MCTSAction(
                    action_type=ActionType.MERGE_FIELDS,
                    field_name=conflict['item_name'],
                    parameters={
                        'field_ids': conflict['item_ids'],
                        'strategy': 'type_consensus'
                    },
                    cost=1.0
                )
                actions.append(action)
            
            elif conflict['type'] == 'framework_conflict':
                # Action to align framework mappings
                action = MCTSAction(
                    action_type=ActionType.ALIGN_FRAMEWORKS,
                    field_name=conflict['item_name'],
                    parameters={
                        'framework': conflict['framework'],
                        'field_ids': conflict['item_ids'],
                        'strategy': 'majority_vote'
                    },
                    cost=0.5
                )
                actions.append(action)
        
        # Actions to improve item definitions
        for item_id, item_info in items.items():
            item_data = item_info['data']
            
            # Action to adjust importance/priority scores
            for score_field in ['importance', 'priority', 'risk_level']:
                if score_field in item_data:
                    current_score = item_data.get(score_field, 5)
                    if isinstance(current_score, (int, float)) and current_score < 8:
                        action = MCTSAction(
                            action_type=ActionType.ADJUST_IMPORTANCE,
                            field_name=item_id,
                            parameters={
                                'field': score_field,
                                'new_value': current_score + 1
                            },
                            cost=0.2
                        )
                        actions.append(action)
        
        # Actions to create item groups
        similar_items = self._find_similar_items(items)
        for group in similar_items:
            if len(group) > 1:
                action = MCTSAction(
                    action_type=ActionType.MERGE_FIELDS,
                    field_name=f"group_{hash(tuple(sorted(group))) % 10000}",
                    parameters={
                        'field_ids': group,
                        'strategy': 'semantic_merge'
                    },
                    cost=len(group) * 0.3
                )
                actions.append(action)
        
        return actions[:50]  # Limit action space for performance
    
    def _find_similar_items(self, items: Dict[str, Dict]) -> List[List[str]]:
        """Find groups of semantically similar items"""
        similar_groups = []
        processed = set()
        
        for item_id1, item_info1 in items.items():
            if item_id1 in processed:
                continue
                
            group = [item_id1]
            item_data1 = item_info1['data']
            
            # Get description from various possible fields
            desc1 = ''
            for desc_field in ['description', 'details', 'summary']:
                if desc_field in item_data1 and item_data1[desc_field]:
                    desc1 = str(item_data1[desc_field]).lower()
                    break
            
            for item_id2, item_info2 in items.items():
                if item_id2 == item_id1 or item_id2 in processed:
                    continue
                
                item_data2 = item_info2['data']
                
                # Get description from various possible fields
                desc2 = ''
                for desc_field in ['description', 'details', 'summary']:
                    if desc_field in item_data2 and item_data2[desc_field]:
                        desc2 = str(item_data2[desc_field]).lower()
                        break
                
                # Calculate similarity based on multiple factors
                similarity_score = 0.0
                
                # Description similarity
                if desc1 and desc2:
                    common_words = set(desc1.split()) & set(desc2.split())
                    total_words = set(desc1.split()) | set(desc2.split())
                    
                    if total_words:
                        desc_similarity = len(common_words) / len(total_words)
                        similarity_score += desc_similarity * 0.6
                
                # Type similarity
                type1 = item_data1.get('type', item_data1.get('attack_type', '')).lower()
                type2 = item_data2.get('type', item_data2.get('attack_type', '')).lower()
                
                if type1 and type2 and type1 == type2:
                    similarity_score += 0.3
                
                # Category similarity
                cat1 = item_data1.get('category', '').lower()
                cat2 = item_data2.get('category', '').lower()
                
                if cat1 and cat2 and cat1 == cat2:
                    similarity_score += 0.1
                
                # If overall similarity is high enough, add to group
                if similarity_score > 0.5:
                    group.append(item_id2)
            
            if len(group) > 1:
                similar_groups.append(group)
                processed.update(group)
        
        return similar_groups
    
    def _select_node(self, root: MCTSNode) -> MCTSNode:
        """Select node using UCB1 policy"""
        current = root
        
        while current.children and not current.is_terminal:
            if current.untried_actions:
                # If there are untried actions, return this node for expansion
                return current
            
            # Select child with highest UCB1 score
            best_child = max(current.children, key=lambda child: child.ucb1_score(self.c))
            current = best_child
        
        return current
    
    def _expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a child"""
        if not node.untried_actions:
            return node
        
        # Select random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Apply action to create new state
        new_state = self._apply_action(node.state, action)
        
        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node
        )
        child.actions = node.actions + [action]
        child.untried_actions = self._generate_possible_actions(new_state)
        
        # Check if terminal state
        child.is_terminal = self._is_terminal_state(new_state)
        
        node.children.append(child)
        
        return child
    
    def _apply_action(self, state: Dict[str, Any], action: MCTSAction) -> Dict[str, Any]:
        """Apply action to state and return new state"""
        new_state = copy.deepcopy(state)
        
        if action.action_type == ActionType.MERGE_FIELDS:
            new_state = self._merge_fields_action(new_state, action)
        elif action.action_type == ActionType.ALIGN_FRAMEWORKS:
            new_state = self._align_frameworks_action(new_state, action)
        elif action.action_type == ActionType.ADJUST_IMPORTANCE:
            new_state = self._adjust_importance_action(new_state, action)
        elif action.action_type == ActionType.RESOLVE_CONFLICT:
            new_state = self._resolve_conflict_action(new_state, action)
        
        new_state['applied_actions'].append(action)
        
        # Update conflicts after action
        new_state['conflicts'] = self._identify_conflicts(new_state['items'])
        
        return new_state
    
    def _merge_fields_action(self, state: Dict[str, Any], action: MCTSAction) -> Dict[str, Any]:
        """Apply item merge action"""
        item_ids = action.parameters.get('field_ids', [])
        strategy = action.parameters.get('strategy', 'majority_vote')
        
        if len(item_ids) < 2:
            return state
        
        # Get item data for merging
        items_to_merge = []
        for item_id in item_ids:
            if item_id in state['items']:
                items_to_merge.append(state['items'][item_id]['data'])
        
        if not items_to_merge:
            return state
        
        # Merge item definitions
        merged_item = self._merge_item_definitions(items_to_merge, strategy)
        
        # Create new merged item entry
        merged_item_id = f"merged_{action.field_name}"
        state['items'][merged_item_id] = {
            'provider': 'consensus',
            'original_name': action.field_name,
            'data': merged_item,
            'source_items': item_ids
        }
        
        # Remove original items
        for item_id in item_ids:
            if item_id in state['items']:
                del state['items'][item_id]
        
        return state
    
    def _align_frameworks_action(self, state: Dict[str, Any], action: MCTSAction) -> Dict[str, Any]:
        """Apply framework alignment action"""
        field_ids = action.parameters.get('field_ids', [])
        framework = action.parameters.get('framework')
        strategy = action.parameters.get('strategy', 'majority_vote')
        
        # Collect framework mappings
        mappings = []
        for item_id in field_ids:
            if item_id in state['items']:
                mapping = state['items'][item_id]['data'].get(framework)
                if mapping and mapping != 'null':
                    mappings.append(mapping)
        
        if not mappings:
            return state
        
        # Choose consensus mapping
        if strategy == 'majority_vote':
            mapping_counts = {}
            for mapping in mappings:
                mapping_counts[mapping] = mapping_counts.get(mapping, 0) + 1
            
            consensus_mapping = max(mapping_counts, key=mapping_counts.get)
        else:
            consensus_mapping = mappings[0]  # Take first as default
        
        # Update all items with consensus mapping
        for item_id in field_ids:
            if item_id in state['items']:
                state['items'][item_id]['data'][framework] = consensus_mapping
        
        return state
    
    def _adjust_importance_action(self, state: Dict[str, Any], action: MCTSAction) -> Dict[str, Any]:
        """Apply importance adjustment action"""
        item_id = action.field_name
        field_name = action.parameters.get('field', 'importance')
        new_value = action.parameters.get('new_value', 5)
        
        if item_id in state['items']:
            state['items'][item_id]['data'][field_name] = new_value
        
        return state
    
    def _resolve_conflict_action(self, state: Dict[str, Any], action: MCTSAction) -> Dict[str, Any]:
        """Apply conflict resolution action"""
        # Implementation would depend on specific conflict resolution strategies
        return state
    
    def _merge_field_definitions(self, field_definitions: List[Dict[str, Any]], strategy: str = 'majority_vote') -> Dict[str, Any]:
        """Merge multiple field definitions using specified strategy"""
        if not field_definitions:
            return {}
        
        if len(field_definitions) == 1:
            return field_definitions[0].copy()
        
        merged = {}
        
        # Merge each property
        for key in ['type', 'description', 'importance', 'OCSF', 'ECS', 'OSSEM']:
            values = []
            for field_def in field_definitions:
                if key in field_def and field_def[key] and field_def[key] != 'null':
                    values.append(field_def[key])
            
            if values:
                if strategy == 'majority_vote':
                    # Take most common value
                    value_counts = {}
                    for value in values:
                        value_counts[value] = value_counts.get(value, 0) + 1
                    merged[key] = max(value_counts, key=value_counts.get)
                elif strategy == 'type_consensus':
                    if key == 'type':
                        # For types, choose most specific
                        type_hierarchy = {'string': 1, 'integer': 2, 'float': 2, 'datetime': 3, 'ip': 3}
                        scored_values = [(type_hierarchy.get(v.lower(), 1), v) for v in values]
                        merged[key] = max(scored_values, key=lambda x: x[0])[1]
                    else:
                        merged[key] = values[0]  # Take first for other properties
                elif strategy == 'semantic_merge':
                    if key == 'description':
                        # Combine descriptions
                        merged[key] = ' | '.join(set(values))
                    elif key == 'importance':
                        # Take maximum importance
                        merged[key] = max(values)
                    else:
                        merged[key] = values[0]  # Take first for others
                else:
                    merged[key] = values[0]  # Default to first value
        
        return merged
    
    def _merge_item_definitions(self, item_definitions: List[Dict[str, Any]], strategy: str = 'majority_vote') -> Dict[str, Any]:
        """Merge multiple item definitions using specified strategy"""
        if not item_definitions:
            return {}
        
        if len(item_definitions) == 1:
            return item_definitions[0].copy()
        
        merged = {}
        
        # Common fields to merge across different item types
        merge_fields = [
            'name', 'pattern_name', 'field_name', 'title', 
            'description', 'details', 'summary',
            'type', 'data_type', 'attack_type', 'category',
            'severity', 'risk_level', 'priority', 'importance',
            'indicators', 'tactics', 'techniques', 'procedures',
            'mitigation', 'response',
            'OCSF', 'ECS', 'OSSEM'
        ]
        
        # Merge each property
        for key in merge_fields:
            values = []
            for item_def in item_definitions:
                if key in item_def and item_def[key] and item_def[key] != 'null':
                    values.append(item_def[key])
            
            if values:
                if strategy == 'majority_vote':
                    # Take most common value
                    value_counts = {}
                    for value in values:
                        str_value = str(value)
                        value_counts[str_value] = value_counts.get(str_value, 0) + 1
                    most_common = max(value_counts, key=value_counts.get)
                    
                    # Try to convert back to original type
                    for value in values:
                        if str(value) == most_common:
                            merged[key] = value
                            break
                            
                elif strategy == 'type_consensus':
                    if key in ['type', 'data_type', 'attack_type']:
                        # For types, choose most specific
                        type_hierarchy = {
                            'string': 1, 'text': 1, 'integer': 2, 'number': 2,
                            'float': 2, 'datetime': 3, 'timestamp': 3, 'ip': 3,
                            'malicious': 4, 'injection': 4, 'exploit': 4
                        }
                        scored_values = [(type_hierarchy.get(str(v).lower(), 1), v) for v in values]
                        merged[key] = max(scored_values, key=lambda x: x[0])[1]
                    elif key in ['severity', 'risk_level', 'priority']:
                        # For severity/risk, take highest
                        severity_scores = {
                            'low': 1, 'medium': 2, 'high': 3, 'critical': 4,
                            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10
                        }
                        scored_values = [(severity_scores.get(str(v).lower(), 0), v) for v in values]
                        merged[key] = max(scored_values, key=lambda x: x[0])[1]
                    else:
                        merged[key] = values[0]  # Take first for other properties
                        
                elif strategy == 'semantic_merge':
                    if key in ['description', 'details', 'summary']:
                        # Combine descriptions
                        unique_descs = []
                        seen_words = set()
                        for desc in values:
                            words = set(str(desc).lower().split())
                            if not words.issubset(seen_words):
                                unique_descs.append(str(desc))
                                seen_words.update(words)
                        merged[key] = ' | '.join(unique_descs)
                    elif key in ['indicators', 'tactics', 'techniques']:
                        # Combine lists
                        combined = []
                        for val in values:
                            if isinstance(val, list):
                                combined.extend(val)
                            else:
                                combined.append(val)
                        merged[key] = list(set(str(item) for item in combined))
                    elif key in ['importance', 'priority', 'risk_level']:
                        # Take maximum importance/risk
                        numeric_values = []
                        for val in values:
                            try:
                                numeric_values.append(float(val))
                            except (ValueError, TypeError):
                                pass
                        if numeric_values:
                            merged[key] = max(numeric_values)
                    else:
                        merged[key] = values[0]  # Take first for others
                else:
                    merged[key] = values[0]  # Default to first value
        
        # Add any additional fields that aren't in the standard list
        all_keys = set()
        for item_def in item_definitions:
            all_keys.update(item_def.keys())
        
        for key in all_keys:
            if key not in merged:
                values = [item_def.get(key) for item_def in item_definitions if key in item_def and item_def[key]]
                if values:
                    merged[key] = values[0]
        
        return merged
    
    def _is_terminal_state(self, state: Dict[str, Any]) -> bool:
        """Check if state is terminal (no more conflicts or actions possible)"""
        return len(state['conflicts']) == 0 or len(state['applied_actions']) > 20
    
    def _simulate_rollout(self, node: MCTSNode, objectives: Dict[str, float]) -> float:
        """Perform random rollout from node"""
        current_state = copy.deepcopy(node.state)
        rollout_actions = []
        
        # Perform random actions until terminal state
        max_rollout_depth = 5
        depth = 0
        
        while depth < max_rollout_depth and not self._is_terminal_state(current_state):
            possible_actions = self._generate_possible_actions(current_state)
            
            if not possible_actions:
                break
            
            # Select random action
            action = random.choice(possible_actions)
            current_state = self._apply_action(current_state, action)
            rollout_actions.append(action)
            depth += 1
        
        # Evaluate final state
        reward = self._evaluate_state(current_state, objectives)
        
        return reward
    
    def _evaluate_state(self, state: Dict[str, Any], objectives: Dict[str, float]) -> float:
        """Evaluate quality of state based on objectives"""
        state_key = self._state_to_key(state)
        
        if state_key in self.evaluation_cache:
            return self.evaluation_cache[state_key]
        
        scores = {}
        
        # Completeness: How many items are present
        total_possible_items = 0
        for result in state['provider_results'].values():
            # Count items from various possible paths
            for path in ['fields', 'malicious_patterns', 'behavioral_patterns', 'patterns']:
                if 'parsed_json' in result and path in result['parsed_json']:
                    path_items = result['parsed_json'][path]
                    if isinstance(path_items, dict):
                        total_possible_items += len(path_items)
                    elif isinstance(path_items, list):
                        total_possible_items += len(path_items)
                    break
        
        current_items = len(state['items'])
        scores['completeness'] = min(1.0, current_items / max(1, total_possible_items))
        
        # Consistency: How many conflicts remain
        total_conflicts = len(state['conflicts'])
        scores['consistency'] = 1.0 / (1.0 + total_conflicts)
        
        # Framework alignment: How well frameworks are aligned
        framework_aligned = 0
        total_items = len(state['items'])
        
        for item_info in state['items'].values():
            item_data = item_info['data']
            aligned_frameworks = 0
            for framework in ['OCSF', 'ECS', 'OSSEM']:
                if item_data.get(framework) and item_data[framework] != 'null':
                    aligned_frameworks += 1
            
            framework_aligned += aligned_frameworks / 3  # Normalize by number of frameworks
        
        scores['framework_alignment'] = framework_aligned / max(1, total_items)
        
        # Semantic coherence: Placeholder - would need semantic analysis
        scores['semantic_coherence'] = 0.8  # Default decent score
        
        # Calculate weighted score
        total_score = sum(scores[obj] * weight for obj, weight in objectives.items())
        
        # Apply action cost penalty
        action_cost = sum(action.cost for action in state.get('applied_actions', []))
        final_score = total_score - 0.1 * action_cost
        
        self.evaluation_cache[state_key] = final_score
        return final_score
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to cache key"""
        # Create a simplified state representation for caching
        item_count = len(state.get('items', {}))
        conflict_count = len(state['conflicts'])
        action_count = len(state.get('applied_actions', []))
        
        return f"i{item_count}_c{conflict_count}_a{action_count}"
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree"""
        current = node
        
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent
    
    def _extract_best_solution(self) -> Dict[str, Any]:
        """Extract best solution from MCTS tree"""
        if not self.root or not self.root.children:
            return {'schema': {}, 'actions': [], 'reward': 0.0}
        
        # Find path to best leaf node
        best_node = max(self.root.children, key=lambda child: child.total_reward / max(1, child.visit_count))
        
        # Follow best path to leaf
        path = []
        current = best_node
        
        while current.children:
            path.append(current)
            current = max(current.children, key=lambda child: child.total_reward / max(1, child.visit_count))
        
        path.append(current)  # Add final node
        
        # Extract schema from final state
        final_state = current.state
        schema = {}
        
        for item_id, item_info in final_state['items'].items():
            item_name = item_info['original_name']
            schema[item_name] = item_info['data']
        
        # Extract action sequence
        actions = final_state.get('applied_actions', [])
        
        return {
            'schema': schema,
            'actions': actions,
            'reward': current.total_reward / max(1, current.visit_count),
            'path_length': len(path)
        }
    
    def _get_best_child_reward(self) -> float:
        """Get reward of best child node"""
        if not self.root or not self.root.children:
            return 0.0
        
        best_child = max(self.root.children, key=lambda child: child.total_reward / max(1, child.visit_count))
        return best_child.total_reward / max(1, best_child.visit_count)
    
    def _calculate_tree_depth(self) -> int:
        """Calculate maximum depth of MCTS tree"""
        if not self.root:
            return 0
        
        def get_depth(node: MCTSNode) -> int:
            if not node.children:
                return 0
            return 1 + max(get_depth(child) for child in node.children)
        
        return get_depth(self.root)
    
    def _count_nodes(self) -> int:
        """Count total nodes in MCTS tree"""
        if not self.root:
            return 0
        
        def count_recursive(node: MCTSNode) -> int:
            return 1 + sum(count_recursive(child) for child in node.children)
        
        return count_recursive(self.root)
    
    def _find_convergence_point(self) -> int:
        """Find iteration where search converged"""
        # Simplified convergence detection
        if not self.root or not self.root.children:
            return 0
        
        # Look for when best child stopped changing significantly
        return min(100, self.max_iterations // 2)  # Placeholder implementation