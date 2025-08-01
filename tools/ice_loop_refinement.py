"""
ICE Loop Iterative Refinement (Section 8)
Implementation of the Iterative Consensus Ensemble Loop with Human-in-the-Loop integration
"""

import numpy as np
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ICEIterationResult:
    """Result of a single ICE iteration"""
    iteration_number: int
    original_confidence: float
    refined_confidence: float
    new_proposals: Dict[str, Any]
    confidence_improvement: float
    convergence_achieved: bool
    timestamp: datetime

@dataclass
class RefinementQuery:
    """Query object for agent refinement"""
    node_id: str
    original_query: str
    refined_query: str
    context_information: Dict[str, Any]
    target_agents: List[str]
    refinement_strategy: str

class ICELoopRefinement:
    """
    Section 8: The Iterative Consensus Ensemble (ICE) Loop
    
    Automated feedback loop for self-correction and refinement of low-confidence nodes
    with Human-in-the-Loop (HITL) integration for persistent uncertainty
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.6,
                 max_iterations: int = 3,
                 convergence_threshold: float = 0.1,
                 improvement_threshold: float = 0.05,
                 enable_hitl: bool = True,
                 hitl_threshold: float = 0.4):
        """
        Initialize ICE Loop refinement system
        
        Args:
            confidence_threshold: Minimum confidence to avoid refinement (τ in Section 8.1)
            max_iterations: Maximum number of ICE iterations
            convergence_threshold: Threshold for declaring convergence
            improvement_threshold: Minimum improvement required to continue
            enable_hitl: Enable Human-in-the-Loop for persistent low confidence
            hitl_threshold: Threshold below which HITL is triggered
        """
        self.confidence_threshold = confidence_threshold  # τ (tau) from Section 8.1
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.improvement_threshold = improvement_threshold
        self.enable_hitl = enable_hitl
        self.hitl_threshold = hitl_threshold
        
        # ICE Loop state tracking
        self.iteration_history: Dict[str, List[ICEIterationResult]] = defaultdict(list)
        self.refinement_queries: Dict[str, List[RefinementQuery]] = defaultdict(list)
        self.hitl_queue: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_refinements = 0
        self.successful_refinements = 0
        self.hitl_interventions = 0
        
        logger.info(f"Initialized ICE Loop: threshold={confidence_threshold}, max_iter={max_iterations}")
    
    def identify_low_confidence_nodes(self, 
                                    consensus_results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        """
        Section 8.1: Trigger identification
        
        "Following a full consensus round, the Uncertainty Module identifies all nodes
        in the final outline whose confidence score falls below a dynamic threshold, τ."
        
        Args:
            consensus_results: Results from consensus engine with confidence scores
            
        Returns:
            List of (node_id, confidence_score) tuples for nodes requiring refinement
        """
        low_confidence_nodes = []
        
        for node_id, result in consensus_results.items():
            confidence = result.get('final_confidence_score', 0.0)
            
            # Dynamic threshold adjustment based on hierarchy level
            dynamic_threshold = self._calculate_dynamic_threshold(node_id, result)
            
            if confidence < dynamic_threshold:
                low_confidence_nodes.append((node_id, confidence))
        
        logger.info(f"Identified {len(low_confidence_nodes)} nodes for ICE refinement")
        return low_confidence_nodes
    
    def generate_refinement_query(self, 
                                node_id: str, 
                                original_result: Dict[str, Any],
                                context_results: Dict[str, Dict[str, Any]] = None) -> RefinementQuery:
        """
        Section 8.1: Re-query Formulation
        
        "For each identified low-confidence node, a specialized agent formulates a new,
        targeted prompt designed to elicit more specific and detailed information."
        
        Args:
            node_id: Identifier of the low-confidence node
            original_result: Original consensus result for the node
            context_results: Results from related/neighboring nodes
            
        Returns:
            RefinementQuery object with targeted prompt
        """
        # Extract context information
        context_info = self._extract_context_information(node_id, original_result, context_results)
        
        # Determine refinement strategy
        refinement_strategy = self._determine_refinement_strategy(original_result)
        
        # Generate targeted query based on strategy
        refined_query = self._generate_targeted_query(node_id, original_result, refinement_strategy, context_info)
        
        # Select target agents for re-querying
        target_agents = self._select_target_agents(original_result, refinement_strategy)
        
        refinement_query = RefinementQuery(
            node_id=node_id,
            original_query=original_result.get('original_query', ''),
            refined_query=refined_query,
            context_information=context_info,
            target_agents=target_agents,
            refinement_strategy=refinement_strategy
        )
        
        self.refinement_queries[node_id].append(refinement_query)
        
        logger.debug(f"Generated refinement query for {node_id}: {refinement_strategy} strategy")
        return refinement_query
    
    async def execute_ice_iteration(self, 
                                  node_id: str,
                                  refinement_query: RefinementQuery,
                                  consensus_engine,
                                  llm_agents: Dict[str, Any]) -> ICEIterationResult:
        """
        Execute a single ICE loop iteration
        
        Args:
            node_id: Node being refined
            refinement_query: Query to send to agents
            consensus_engine: Consensus engine for re-evaluation
            llm_agents: Available LLM agents
            
        Returns:
            ICEIterationResult with iteration outcomes
        """
        iteration_num = len(self.iteration_history[node_id]) + 1
        logger.info(f"Executing ICE iteration {iteration_num} for node {node_id}")
        
        # Get original confidence for comparison
        original_confidence = 0.0
        if self.iteration_history[node_id]:
            original_confidence = self.iteration_history[node_id][-1].refined_confidence
        
        # Send refined query to target agents
        new_proposals = {}
        for agent_id in refinement_query.target_agents:
            if agent_id in llm_agents:
                try:
                    # In real implementation, this would be actual LLM API call
                    proposal = await self._query_agent_with_refinement(
                        agent_id, refinement_query, llm_agents[agent_id]
                    )
                    new_proposals[agent_id] = proposal
                except Exception as e:
                    logger.error(f"Failed to query agent {agent_id}: {e}")
        
        # Re-run consensus with new proposals
        refined_result = await self._run_focused_consensus(
            node_id, new_proposals, consensus_engine
        )
        
        refined_confidence = refined_result.get('final_confidence_score', 0.0)
        confidence_improvement = refined_confidence - original_confidence
        
        # Check convergence
        convergence_achieved = (
            refined_confidence >= self.confidence_threshold or
            confidence_improvement < self.improvement_threshold
        )
        
        iteration_result = ICEIterationResult(
            iteration_number=iteration_num,
            original_confidence=original_confidence,
            refined_confidence=refined_confidence,
            new_proposals=new_proposals,
            confidence_improvement=confidence_improvement,
            convergence_achieved=convergence_achieved,
            timestamp=datetime.now()
        )
        
        self.iteration_history[node_id].append(iteration_result)
        
        logger.info(f"ICE iteration {iteration_num} complete: {refined_confidence:.3f} confidence (+{confidence_improvement:.3f})")
        return iteration_result
    
    async def run_full_ice_loop(self, 
                              node_id: str,
                              original_result: Dict[str, Any],
                              consensus_engine,
                              llm_agents: Dict[str, Any],
                              context_results: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete ICE loop for a single node
        
        Args:
            node_id: Node to refine
            original_result: Original consensus result
            consensus_engine: Consensus engine instance
            llm_agents: Available LLM agents
            context_results: Context from other nodes
            
        Returns:
            Final refined result with ICE metadata
        """
        self.total_refinements += 1
        
        logger.info(f"Starting ICE loop for node {node_id}")
        
        current_result = original_result.copy()
        ice_metadata = {
            'ice_applied': True,
            'original_confidence': original_result.get('final_confidence_score', 0.0),
            'iterations_performed': 0,
            'final_confidence': 0.0,
            'improvement_achieved': 0.0,
            'convergence_reason': '',
            'hitl_triggered': False
        }
        
        # Run ICE iterations
        for iteration in range(self.max_iterations):
            # Generate refinement query
            refinement_query = self.generate_refinement_query(
                node_id, current_result, context_results
            )
            
            # Execute iteration
            iteration_result = await self.execute_ice_iteration(
                node_id, refinement_query, consensus_engine, llm_agents
            )
            
            # Update current result
            current_result.update({
                'final_confidence_score': iteration_result.refined_confidence,
                'ice_iteration': iteration + 1,
                'new_proposals': iteration_result.new_proposals
            })
            
            ice_metadata['iterations_performed'] = iteration + 1
            ice_metadata['final_confidence'] = iteration_result.refined_confidence
            ice_metadata['improvement_achieved'] = (
                iteration_result.refined_confidence - ice_metadata['original_confidence']
            )
            
            # Check for convergence
            if iteration_result.convergence_achieved:
                ice_metadata['convergence_reason'] = 'threshold_reached'
                self.successful_refinements += 1
                break
                
            if iteration_result.confidence_improvement < self.improvement_threshold:
                ice_metadata['convergence_reason'] = 'no_improvement'
                break
        else:
            ice_metadata['convergence_reason'] = 'max_iterations'
        
        # Check if HITL is needed
        final_confidence = ice_metadata['final_confidence']
        if (self.enable_hitl and 
            final_confidence < self.hitl_threshold and
            ice_metadata['convergence_reason'] != 'threshold_reached'):
            
            hitl_item = self._create_hitl_item(node_id, current_result, ice_metadata)
            self.hitl_queue.append(hitl_item)
            ice_metadata['hitl_triggered'] = True
            self.hitl_interventions += 1
            
            logger.info(f"ICE Loop complete for {node_id}: HITL triggered (confidence: {final_confidence:.3f})")
        else:
            logger.info(f"ICE Loop complete for {node_id}: {ice_metadata['convergence_reason']} (confidence: {final_confidence:.3f})")
        
        current_result['ice_metadata'] = ice_metadata
        return current_result
    
    async def batch_ice_refinement(self, 
                                 consensus_results: Dict[str, Dict[str, Any]],
                                 consensus_engine,
                                 llm_agents: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Apply ICE loop refinement to all low-confidence nodes in batch
        
        Args:
            consensus_results: Complete consensus results
            consensus_engine: Consensus engine instance
            llm_agents: Available LLM agents
            
        Returns:
            Refined consensus results with ICE improvements
        """
        # Identify nodes needing refinement
        low_confidence_nodes = self.identify_low_confidence_nodes(consensus_results)
        
        if not low_confidence_nodes:
            logger.info("No nodes require ICE refinement")
            return consensus_results
        
        refined_results = consensus_results.copy()
        
        # Process nodes in parallel (with controlled concurrency)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent refinements
        
        async def refine_node(node_id: str, original_result: Dict[str, Any]):
            async with semaphore:
                return await self.run_full_ice_loop(
                    node_id, original_result, consensus_engine, llm_agents, consensus_results
                )
        
        # Create tasks for all low-confidence nodes
        refinement_tasks = [
            refine_node(node_id, consensus_results[node_id])
            for node_id, _ in low_confidence_nodes
        ]
        
        # Execute refinements
        refined_node_results = await asyncio.gather(*refinement_tasks, return_exceptions=True)
        
        # Update results
        for i, (node_id, _) in enumerate(low_confidence_nodes):
            if not isinstance(refined_node_results[i], Exception):
                refined_results[node_id] = refined_node_results[i]
            else:
                logger.error(f"ICE refinement failed for {node_id}: {refined_node_results[i]}")
        
        logger.info(f"Batch ICE refinement complete: {len(low_confidence_nodes)} nodes processed")
        return refined_results
    
    def get_hitl_items(self) -> List[Dict[str, Any]]:
        """
        Section 8.1: Human-in-the-Loop (HITL) Adjudication
        
        "If a node's confidence remains below the threshold after a predefined number
        of ICE iterations, it is flagged for human review."
        
        Returns:
            List of items requiring human intervention
        """
        return self.hitl_queue.copy()
    
    def process_hitl_feedback(self, 
                            node_id: str, 
                            human_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process human feedback from HITL review
        
        Args:
            node_id: Node that was reviewed
            human_feedback: Human expert feedback
            
        Returns:
            Updated consensus result incorporating human input
        """
        # Find the HITL item
        hitl_item = None
        for item in self.hitl_queue:
            if item['node_id'] == node_id:
                hitl_item = item
                break
        
        if not hitl_item:
            logger.warning(f"No HITL item found for node {node_id}")
            return {}
        
        # Process human feedback
        updated_result = hitl_item['current_result'].copy()
        
        # Apply human corrections/improvements
        if 'corrected_content' in human_feedback:
            updated_result['human_corrected_content'] = human_feedback['corrected_content']
        
        if 'confidence_override' in human_feedback:
            updated_result['final_confidence_score'] = human_feedback['confidence_override']
        
        if 'expert_notes' in human_feedback:
            updated_result['expert_notes'] = human_feedback['expert_notes']
        
        # Mark as human-reviewed
        updated_result.setdefault('ice_metadata', {})['hitl_resolved'] = True
        updated_result['ice_metadata']['human_intervention_timestamp'] = datetime.now().isoformat()
        
        # Remove from HITL queue
        self.hitl_queue = [item for item in self.hitl_queue if item['node_id'] != node_id]
        
        logger.info(f"HITL feedback processed for node {node_id}")
        return updated_result
    
    def get_ice_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive ICE loop statistics
        
        Returns:
            Dictionary with ICE performance metrics
        """
        total_iterations = sum(len(history) for history in self.iteration_history.values())
        
        confidence_improvements = []
        for history in self.iteration_history.values():
            if history:
                final_result = history[-1]
                confidence_improvements.append(final_result.confidence_improvement)
        
        stats = {
            'total_refinements': self.total_refinements,
            'successful_refinements': self.successful_refinements,
            'success_rate': self.successful_refinements / max(self.total_refinements, 1),
            'total_iterations': total_iterations,
            'avg_iterations_per_node': total_iterations / max(len(self.iteration_history), 1),
            'hitl_interventions': self.hitl_interventions,
            'hitl_queue_size': len(self.hitl_queue),
            'avg_confidence_improvement': np.mean(confidence_improvements) if confidence_improvements else 0.0,
            'nodes_refined': len(self.iteration_history)
        }
        
        return stats
    
    # Helper methods
    
    def _calculate_dynamic_threshold(self, node_id: str, result: Dict[str, Any]) -> float:
        """Calculate dynamic threshold based on hierarchy level (Section 6.4)"""
        hierarchy_level = result.get('hierarchy_level', 0)
        
        # Higher threshold for top-level nodes (more important)
        if hierarchy_level == 0:
            return self.confidence_threshold * 1.2
        elif hierarchy_level == 1:
            return self.confidence_threshold * 1.1
        else:
            return self.confidence_threshold
    
    def _extract_context_information(self, node_id: str, original_result: Dict[str, Any], context_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant context information for refinement"""
        context_info = {
            'node_id': node_id,
            'original_confidence': original_result.get('final_confidence_score', 0.0),
            'voting_agents': original_result.get('voting_agents', []),
            'consensus_reached': original_result.get('consensus_reached', False),
            'related_nodes': []
        }
        
        # Add information from related nodes
        if context_results:
            for related_id, related_result in context_results.items():
                if related_id != node_id and related_result.get('final_confidence_score', 0) > 0.7:
                    context_info['related_nodes'].append({
                        'id': related_id,
                        'confidence': related_result.get('final_confidence_score', 0.0),
                        'content_summary': self._summarize_content(related_result)
                    })
        
        return context_info
    
    def _determine_refinement_strategy(self, original_result: Dict[str, Any]) -> str:
        """Determine appropriate refinement strategy"""
        confidence = original_result.get('final_confidence_score', 0.0)
        voting_analysis = original_result.get('voting_analysis', {})
        
        if confidence < 0.3:
            return 'clarification'  # Very low confidence - need clarification
        elif len(voting_analysis.get('voting_agents', [])) < 3:
            return 'broader_input'  # Few agents participated - get more input
        elif voting_analysis.get('consensus_reached', False) is False:
            return 'conflict_resolution'  # No consensus - resolve conflicts
        else:
            return 'detail_enhancement'  # Moderate confidence - add details
    
    def _generate_targeted_query(self, node_id: str, original_result: Dict[str, Any], strategy: str, context_info: Dict[str, Any]) -> str:
        """Generate targeted refinement query based on strategy"""
        base_content = self._extract_node_content_summary(original_result)
        
        strategy_templates = {
            'clarification': f"Please provide more specific and detailed information about '{base_content}'. Focus on precise definitions, clear explanations, and concrete examples. What specific aspects need clarification?",
            
            'broader_input': f"Regarding '{base_content}', please provide comprehensive analysis from multiple perspectives. Consider different viewpoints, alternative interpretations, and additional relevant factors that might have been overlooked.",
            
            'conflict_resolution': f"There are conflicting views about '{base_content}'. Please help resolve these conflicts by: 1) Identifying the key points of disagreement, 2) Providing evidence or reasoning for each perspective, 3) Suggesting a resolution or synthesis.",
            
            'detail_enhancement': f"Please elaborate on '{base_content}' with more detailed, specific information. Provide: 1) More granular sub-points, 2) Specific examples or evidence, 3) Technical details or implementation considerations."
        }
        
        refined_query = strategy_templates.get(strategy, strategy_templates['clarification'])
        
        # Add context if available
        if context_info.get('related_nodes'):
            related_info = ", ".join([node['id'] for node in context_info['related_nodes'][:3]])
            refined_query += f" Consider the context of related elements: {related_info}."
        
        return refined_query
    
    def _select_target_agents(self, original_result: Dict[str, Any], strategy: str) -> List[str]:
        """Select target agents for re-querying based on strategy"""
        voting_agents = original_result.get('voting_agents', [])
        
        if strategy == 'broader_input':
            # Include agents that didn't participate originally
            # In real implementation, would have access to full agent list
            return voting_agents + ['additional_agent_1', 'additional_agent_2']
        elif strategy == 'conflict_resolution':
            # Focus on agents with highest weights in original voting
            return voting_agents[:3]  # Top 3 agents
        else:
            # Standard re-query to original participants
            return voting_agents
    
    async def _query_agent_with_refinement(self, agent_id: str, refinement_query: RefinementQuery, agent_instance) -> Dict[str, Any]:
        """Query agent with refined prompt (placeholder for actual implementation)"""
        # Placeholder - in real implementation, would call actual LLM API
        await asyncio.sleep(0.1)  # Simulate API call
        
        return {
            'agent_id': agent_id,
            'refined_response': f"Refined response to: {refinement_query.refined_query[:50]}...",
            'confidence': np.random.uniform(0.5, 0.9),
            'refinement_strategy': refinement_query.refinement_strategy,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _run_focused_consensus(self, node_id: str, new_proposals: Dict[str, Any], consensus_engine) -> Dict[str, Any]:
        """Run focused consensus on refined proposals (placeholder)"""
        # Placeholder - in real implementation, would call consensus engine
        await asyncio.sleep(0.1)  # Simulate consensus calculation
        
        # Simulate improved confidence
        base_confidence = np.random.uniform(0.4, 0.6)
        improvement = len(new_proposals) * 0.1  # More proposals = better confidence
        
        return {
            'final_confidence_score': min(base_confidence + improvement, 1.0),
            'consensus_reached': base_confidence + improvement > 0.6,
            'refined_proposals': new_proposals,
            'ice_applied': True
        }
    
    def _create_hitl_item(self, node_id: str, current_result: Dict[str, Any], ice_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create HITL queue item"""
        return {
            'node_id': node_id,
            'current_result': current_result,
            'ice_metadata': ice_metadata,
            'hitl_reason': 'persistent_low_confidence',
            'created_timestamp': datetime.now().isoformat(),
            'priority': 'high' if ice_metadata['final_confidence'] < 0.2 else 'medium',
            'context_summary': self._summarize_content(current_result)
        }
    
    def _summarize_content(self, result: Dict[str, Any]) -> str:
        """Create brief summary of result content"""
        # Extract key information for summary
        summary_parts = []
        
        if 'consensus' in result and result['consensus']:
            content = str(result['consensus'])
            summary_parts.append(content[:100])
        
        if 'voting_analysis' in result:
            agents = result['voting_analysis'].get('total_agents', 0)
            summary_parts.append(f"{agents} agents")
        
        return " | ".join(summary_parts) if summary_parts else "No content summary available"
    
    def _extract_node_content_summary(self, result: Dict[str, Any]) -> str:
        """Extract content summary for query generation"""
        if 'consensus' in result and result['consensus']:
            content = str(result['consensus'])
            return content[:150] + "..." if len(content) > 150 else content
        
        return "unspecified content"