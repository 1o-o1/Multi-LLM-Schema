"""
CSP Framework Alignment for Multi-Agent Schema Consensus
Constraint Satisfaction Problem solver for cross-framework field alignment
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from itertools import combinations, product
import json

logger = logging.getLogger(__name__)

class CSPFrameworkAlignment:
    """Constraint Satisfaction Problem solver for framework alignment"""
    
    def __init__(self):
        """Initialize CSP alignment service"""
        self.variables = {}
        self.domains = {}
        self.constraints = []
        self.solution_cache = {}
        
    def define_alignment_variables(self, provider_results: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """
        Define CSP variables from provider field results
        
        Args:
            provider_results: Dictionary of provider results with field data
            
        Returns:
            Dictionary mapping variable names to their domains
        """
        variables = {}
        
        # Extract all unique field names across providers
        all_fields = set()
        provider_fields = {}
        
        for provider, result in provider_results.items():
            if 'parsed_json' not in result or 'fields' not in result['parsed_json']:
                continue
                
            fields = result['parsed_json']['fields']
            provider_fields[provider] = set(fields.keys())
            all_fields.update(fields.keys())
        
        # Create alignment variables for each field
        for field_name in all_fields:
            # Variable represents which canonical field this maps to
            var_name = f"canonical_{field_name}"
            
            # Domain includes itself and semantically similar fields
            domain = {field_name}
            
            # Add semantically similar fields from other providers
            for other_field in all_fields:
                if other_field != field_name:
                    # Simple similarity heuristic based on name overlap
                    if self._field_name_similarity(field_name, other_field) > 0.6:
                        domain.add(other_field)
            
            variables[var_name] = domain
        
        # Store provider mappings for constraint generation
        self.provider_fields = provider_fields
        self.variables = variables
        self.domains = variables
        
        logger.info(f"Defined {len(variables)} CSP variables from {len(all_fields)} unique fields")
        return variables
    
    def _field_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate simple similarity between field names"""
        name1_words = set(name1.lower().replace('_', ' ').split())
        name2_words = set(name2.lower().replace('_', ' ').split())
        
        if not name1_words or not name2_words:
            return 0.0
        
        intersection = name1_words.intersection(name2_words)
        union = name1_words.union(name2_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def add_framework_consistency_constraints(self, provider_results: Dict[str, Dict]):
        """
        Add constraints to ensure framework mapping consistency
        
        Args:
            provider_results: Dictionary of provider results with field data
        """
        constraints = []
        
        # Extract framework mappings for each field
        framework_mappings = defaultdict(dict)
        
        for provider, result in provider_results.items():
            if 'parsed_json' not in result or 'fields' not in result['parsed_json']:
                continue
                
            fields = result['parsed_json']['fields']
            for field_name, field_data in fields.items():
                for framework in ['OCSF', 'ECS', 'OSSEM']:
                    mapping = field_data.get(framework)
                    if mapping and mapping != 'null':
                        framework_mappings[field_name][framework] = mapping
        
        # Constraint: Fields with same framework mapping should align
        for framework in ['OCSF', 'ECS', 'OSSEM']:
            mapping_groups = defaultdict(list)
            
            for field_name, mappings in framework_mappings.items():
                if framework in mappings:
                    mapping_value = mappings[framework]
                    mapping_groups[mapping_value].append(field_name)
            
            # Add consistency constraints for each mapping group
            for mapping_value, field_list in mapping_groups.items():
                if len(field_list) > 1:
                    constraint = {
                        'type': 'framework_consistency',
                        'framework': framework,
                        'mapping': mapping_value,
                        'fields': field_list,
                        'constraint_func': self._framework_consistency_constraint
                    }
                    constraints.append(constraint)
        
        self.constraints.extend(constraints)
        logger.info(f"Added {len(constraints)} framework consistency constraints")
    
    def add_type_compatibility_constraints(self, provider_results: Dict[str, Dict]):
        """
        Add constraints to ensure data type compatibility
        
        Args:
            provider_results: Dictionary of provider results with field data
        """
        constraints = []
        
        # Extract type information
        field_types = {}
        
        for provider, result in provider_results.items():
            if 'parsed_json' not in result or 'fields' not in result['parsed_json']:
                continue
                
            fields = result['parsed_json']['fields']
            for field_name, field_data in fields.items():
                field_type = field_data.get('type', '').lower()
                if field_type:
                    field_types[field_name] = field_type
        
        # Type compatibility groups
        compatible_types = {
            'numeric': {'integer', 'int', 'float', 'double', 'number'},
            'temporal': {'datetime', 'timestamp', 'date', 'time'},
            'textual': {'string', 'str', 'text'},
            'network': {'ip', 'ipv4', 'ipv6', 'address'},
            'collection': {'array', 'list'}
        }
        
        # Add type compatibility constraints
        for field1, field2 in combinations(field_types.keys(), 2):
            type1 = field_types[field1]
            type2 = field_types[field2]
            
            # Check if types are compatible
            compatible = False
            if type1 == type2:
                compatible = True
            else:
                for type_group in compatible_types.values():
                    if type1 in type_group and type2 in type_group:
                        compatible = True
                        break
            
            if not compatible:
                constraint = {
                    'type': 'type_incompatibility',
                    'field1': field1,
                    'field2': field2,
                    'type1': type1,
                    'type2': type2,
                    'constraint_func': self._type_incompatibility_constraint
                }
                constraints.append(constraint)
        
        self.constraints.extend(constraints)
        logger.info(f"Added {len(constraints)} type compatibility constraints")
    
    def add_semantic_similarity_constraints(self, similarity_matrix: np.ndarray, 
                                          field_names: List[str], 
                                          threshold: float = 0.8):
        """
        Add constraints based on semantic similarity scores
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            field_names: List of field names corresponding to matrix indices
            threshold: Similarity threshold for creating constraints
        """
        constraints = []
        
        for i, field1 in enumerate(field_names):
            for j, field2 in enumerate(field_names):
                if i >= j:  # Avoid duplicate constraints
                    continue
                
                similarity = similarity_matrix[i, j]
                
                if similarity >= threshold:
                    # High similarity fields should align
                    constraint = {
                        'type': 'high_similarity_alignment',
                        'field1': field1,
                        'field2': field2,
                        'similarity': float(similarity),
                        'constraint_func': self._high_similarity_constraint
                    }
                    constraints.append(constraint)
                elif similarity <= 0.2:
                    # Low similarity fields should not align
                    constraint = {
                        'type': 'low_similarity_separation',
                        'field1': field1,
                        'field2': field2,
                        'similarity': float(similarity),
                        'constraint_func': self._low_similarity_constraint
                    }
                    constraints.append(constraint)
        
        self.constraints.extend(constraints)
        logger.info(f"Added {len(constraints)} semantic similarity constraints")
    
    def _framework_consistency_constraint(self, assignment: Dict[str, str], constraint: Dict) -> bool:
        """Check framework consistency constraint"""
        field_list = constraint['fields']
        canonical_assignments = []
        
        for field in field_list:
            var_name = f"canonical_{field}"
            if var_name in assignment:
                canonical_assignments.append(assignment[var_name])
        
        # All fields with same framework mapping should have same canonical assignment
        return len(set(canonical_assignments)) <= 1
    
    def _type_incompatibility_constraint(self, assignment: Dict[str, str], constraint: Dict) -> bool:
        """Check type incompatibility constraint"""
        field1 = constraint['field1']
        field2 = constraint['field2']
        
        var1 = f"canonical_{field1}"
        var2 = f"canonical_{field2}"
        
        if var1 in assignment and var2 in assignment:
            # Incompatible types should not map to the same canonical field
            return assignment[var1] != assignment[var2]
        
        return True  # Constraint satisfied if variables not assigned
    
    def _high_similarity_constraint(self, assignment: Dict[str, str], constraint: Dict) -> bool:
        """Check high similarity alignment constraint"""
        field1 = constraint['field1']
        field2 = constraint['field2']
        
        var1 = f"canonical_{field1}"
        var2 = f"canonical_{field2}"
        
        if var1 in assignment and var2 in assignment:
            # High similarity fields should map to the same canonical field
            return assignment[var1] == assignment[var2]
        
        return True
    
    def _low_similarity_constraint(self, assignment: Dict[str, str], constraint: Dict) -> bool:
        """Check low similarity separation constraint"""
        field1 = constraint['field1']
        field2 = constraint['field2']
        
        var1 = f"canonical_{field1}"
        var2 = f"canonical_{field2}"
        
        if var1 in assignment and var2 in assignment:
            # Low similarity fields should not map to the same canonical field
            return assignment[var1] != assignment[var2]
        
        return True
    
    def solve_csp_backtracking(self) -> Optional[Dict[str, str]]:
        """
        Solve CSP using backtracking algorithm
        
        Returns:
            Solution assignment or None if no solution exists
        """
        try:
            assignment = {}
            solution = self._backtrack_search(assignment)
            
            if solution:
                logger.info(f"Found CSP solution with {len(solution)} variable assignments")
                return solution
            else:
                logger.warning("No solution found for CSP")
                return None
                
        except Exception as e:
            logger.error(f"CSP solving failed: {e}")
            return None
    
    def _backtrack_search(self, assignment: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Backtracking search algorithm"""
        # Check if assignment is complete
        if len(assignment) == len(self.variables):
            return assignment
        
        # Select unassigned variable
        var = self._select_unassigned_variable(assignment)
        
        # Try each value in domain
        for value in self._order_domain_values(var, assignment):
            if self._is_consistent(var, value, assignment):
                # Make assignment
                assignment[var] = value
                
                # Recursively try to complete assignment
                result = self._backtrack_search(assignment)
                if result:
                    return result
                
                # Backtrack
                del assignment[var]
        
        return None
    
    def _select_unassigned_variable(self, assignment: Dict[str, str]) -> str:
        """Select next unassigned variable using MRV heuristic"""
        unassigned = [var for var in self.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Most Remaining Values heuristic - choose variable with smallest domain
        return min(unassigned, key=lambda var: len(self.domains[var]))
    
    def _order_domain_values(self, var: str, assignment: Dict[str, str]) -> List[str]:
        """Order domain values using least constraining value heuristic"""
        domain = list(self.domains[var])
        
        # Simple ordering - could be enhanced with least constraining value heuristic
        return sorted(domain)
    
    def _is_consistent(self, var: str, value: str, assignment: Dict[str, str]) -> bool:
        """Check if assignment is consistent with all constraints"""
        # Create temporary assignment
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        
        # Check all constraints
        for constraint in self.constraints:
            if not constraint['constraint_func'](temp_assignment, constraint):
                return False
        
        return True
    
    def generate_alignment_solution(self, provider_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate complete alignment solution using CSP
        
        Args:
            provider_results: Dictionary of provider results
            
        Returns:
            Dictionary containing alignment solution and metadata
        """
        try:
            # Define variables and constraints
            self.define_alignment_variables(provider_results)
            self.add_framework_consistency_constraints(provider_results)
            self.add_type_compatibility_constraints(provider_results)
            
            # Solve CSP
            solution = self.solve_csp_backtracking()
            
            if solution:
                # Post-process solution into alignment groups
                alignment_groups = self._extract_alignment_groups(solution)
                
                return {
                    'success': True,
                    'csp_solution': solution,
                    'alignment_groups': alignment_groups,
                    'n_variables': len(self.variables),
                    'n_constraints': len(self.constraints),
                    'n_groups': len(alignment_groups)
                }
            else:
                # Try relaxed constraints
                relaxed_solution = self._solve_with_relaxed_constraints()
                
                return {
                    'success': False,
                    'relaxed_solution': relaxed_solution,
                    'n_variables': len(self.variables),
                    'n_constraints': len(self.constraints),
                    'message': 'No solution found with strict constraints, using relaxed approach'
                }
                
        except Exception as e:
            logger.error(f"Alignment solution generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'CSP alignment failed'
            }
    
    def _extract_alignment_groups(self, solution: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract alignment groups from CSP solution"""
        groups = defaultdict(list)
        
        for var, canonical in solution.items():
            # Extract original field name from variable name
            field_name = var.replace('canonical_', '')
            groups[canonical].append(field_name)
        
        # Convert to regular dict and filter single-field groups
        alignment_groups = {}
        for canonical, fields in groups.items():
            if len(fields) > 1:  # Only include multi-field groups
                alignment_groups[canonical] = fields
        
        return dict(alignment_groups)
    
    def _solve_with_relaxed_constraints(self) -> Dict[str, List[str]]:
        """Solve with relaxed constraints when strict CSP fails"""
        # Simple greedy approach based on framework mappings
        groups = defaultdict(list)
        
        for var in self.variables:
            field_name = var.replace('canonical_', '')
            # Use the field name itself as canonical
            groups[field_name].append(field_name)
        
        return dict(groups)
    
    def validate_alignment_quality(self, 
                                 alignment_groups: Dict[str, List[str]], 
                                 provider_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Validate the quality of field alignment
        
        Args:
            alignment_groups: Dictionary of alignment groups
            provider_results: Original provider results
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'framework_consistency': 0.0,
            'type_consistency': 0.0,
            'semantic_coherence': 0.0,
            'coverage_completeness': 0.0
        }
        
        try:
            # Extract field data
            all_field_data = {}
            for provider, result in provider_results.items():
                if 'parsed_json' in result and 'fields' in result['parsed_json']:
                    fields = result['parsed_json']['fields']
                    for field_name, field_data in fields.items():
                        all_field_data[field_name] = field_data
            
            # Calculate framework consistency
            framework_scores = []
            for canonical, fields in alignment_groups.items():
                if len(fields) > 1:
                    # Check if all fields have consistent framework mappings
                    for framework in ['OCSF', 'ECS', 'OSSEM']:
                        mappings = []
                        for field in fields:
                            if field in all_field_data:
                                mapping = all_field_data[field].get(framework)
                                if mapping and mapping != 'null':
                                    mappings.append(mapping)
                        
                        if mappings:
                            unique_mappings = set(mappings)
                            consistency = 1.0 if len(unique_mappings) == 1 else 0.5
                            framework_scores.append(consistency)
            
            if framework_scores:
                metrics['framework_consistency'] = np.mean(framework_scores)
            
            # Calculate type consistency
            type_scores = []
            for canonical, fields in alignment_groups.items():
                if len(fields) > 1:
                    types = []
                    for field in fields:
                        if field in all_field_data:
                            field_type = all_field_data[field].get('type', '').lower()
                            if field_type:
                                types.append(field_type)
                    
                    if types:
                        unique_types = set(types)
                        consistency = 1.0 if len(unique_types) == 1 else 0.5
                        type_scores.append(consistency)
            
            if type_scores:
                metrics['type_consistency'] = np.mean(type_scores)
            
            # Calculate coverage completeness
            total_fields = len(all_field_data)
            aligned_fields = sum(len(fields) for fields in alignment_groups.values())
            metrics['coverage_completeness'] = aligned_fields / total_fields if total_fields > 0 else 0.0
            
            logger.info(f"Alignment quality metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Alignment validation failed: {e}")
            return metrics