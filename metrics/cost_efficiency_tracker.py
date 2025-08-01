#!/usr/bin/env python3
"""
Cost-Efficiency Tracking (Section 9.2)

This module implements cost and efficiency metrics:
- Processing cost analysis and resource utilization
- Quality-cost trade-off assessment
- ROI calculations for consensus processing
- Resource optimization recommendations
"""

import json
import time
import logging
import statistics
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CostEfficiencyMetrics:
    """Container for cost and efficiency metrics"""
    # Processing Costs
    computational_cost: float = 0.0  # Estimated cost in compute units
    time_cost: float = 0.0  # Time as cost factor
    memory_cost: float = 0.0  # Memory usage cost
    total_processing_cost: float = 0.0
    
    # Quality-Cost Analysis
    quality_per_cost: float = 0.0
    cost_per_consensus_part: float = 0.0
    efficiency_ratio: float = 0.0
    
    # Resource Utilization
    cpu_cost_efficiency: float = 0.0
    memory_cost_efficiency: float = 0.0
    io_cost_efficiency: float = 0.0
    
    # Comparative Analysis
    cost_vs_baseline: float = 0.0  # Compared to simple averaging
    quality_improvement_per_cost: float = 0.0
    
    # ROI Metrics
    return_on_investment: float = 0.0
    cost_benefit_ratio: float = 0.0
    marginal_cost_per_improvement: float = 0.0
    
    # Component Cost Breakdown
    component_costs: Dict[str, float] = field(default_factory=dict)
    cost_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Optimization Metrics
    potential_savings: float = 0.0
    optimization_priority: List[str] = field(default_factory=list)
    cost_reduction_opportunities: Dict[str, float] = field(default_factory=dict)
    
    # Scalability Costs
    scaling_cost_factor: float = 1.0
    cost_per_additional_input: float = 0.0
    
    def __post_init__(self):
        if not self.component_costs:
            self.component_costs = {}
        if not self.cost_distribution:
            self.cost_distribution = {}
        if not self.cost_reduction_opportunities:
            self.cost_reduction_opportunities = {}

class CostEfficiencyTracker:
    """Cost and efficiency tracking system"""
    
    def __init__(self):
        """Initialize cost efficiency tracker"""
        # Cost models (could be configured externally)
        self.cost_models = {
            'cpu_cost_per_second': 0.01,  # $0.01 per CPU-second
            'memory_cost_per_mb_second': 0.0001,  # $0.0001 per MB-second
            'io_cost_per_operation': 0.000001,  # Minimal I/O cost
            'baseline_simple_average_cost': 0.05  # Cost of simple averaging
        }
        
    def evaluate_cost_efficiency(
        self,
        consensus_output: Dict[str, Any],
        processing_metadata: Dict[str, Any],
        original_inputs: List[Dict[str, Any]],
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> CostEfficiencyMetrics:
        """
        Comprehensive cost-efficiency evaluation
        
        Args:
            consensus_output: Output from consensus system
            processing_metadata: Metadata about consensus process
            original_inputs: Original input files
            performance_metrics: Optional performance metrics from PerformanceTracker
            
        Returns:
            CostEfficiencyMetrics with all computed metrics
        """
        logger.info("Starting cost-efficiency evaluation")
        
        metrics = CostEfficiencyMetrics()
        
        try:
            # 1. Calculate Processing Costs
            metrics.computational_cost = self._calculate_computational_cost(processing_metadata, performance_metrics)
            metrics.time_cost = self._calculate_time_cost(processing_metadata)
            metrics.memory_cost = self._calculate_memory_cost(processing_metadata, performance_metrics)
            metrics.total_processing_cost = metrics.computational_cost + metrics.time_cost + metrics.memory_cost
            
            # 2. Quality-Cost Analysis
            overall_quality = float(consensus_output.get('quality_metrics', {}).get('overall_quality', 0.0))
            metrics.quality_per_cost = self._calculate_quality_per_cost(overall_quality, metrics.total_processing_cost)
            metrics.cost_per_consensus_part = self._calculate_cost_per_part(metrics.total_processing_cost, consensus_output)
            metrics.efficiency_ratio = self._calculate_efficiency_ratio(metrics, overall_quality)
            
            # 3. Resource Utilization Efficiency
            metrics.cpu_cost_efficiency = self._calculate_cpu_cost_efficiency(performance_metrics, processing_metadata)
            metrics.memory_cost_efficiency = self._calculate_memory_cost_efficiency(performance_metrics, processing_metadata)
            metrics.io_cost_efficiency = self._calculate_io_cost_efficiency(performance_metrics, processing_metadata)
            
            # 4. Comparative Analysis
            metrics.cost_vs_baseline = self._calculate_cost_vs_baseline(metrics.total_processing_cost, len(original_inputs))
            metrics.quality_improvement_per_cost = self._calculate_quality_improvement_per_cost(overall_quality, metrics.total_processing_cost)
            
            # 5. ROI Metrics
            metrics.return_on_investment = self._calculate_roi(overall_quality, metrics.total_processing_cost)
            metrics.cost_benefit_ratio = self._calculate_cost_benefit_ratio(overall_quality, metrics.total_processing_cost)
            metrics.marginal_cost_per_improvement = self._calculate_marginal_cost_per_improvement(metrics)
            
            # 6. Component Cost Breakdown
            metrics.component_costs = self._analyze_component_costs(processing_metadata, performance_metrics)
            metrics.cost_distribution = self._calculate_cost_distribution(metrics.component_costs, metrics.total_processing_cost)
            
            # 7. Optimization Analysis
            metrics.potential_savings = self._identify_potential_savings(metrics)
            metrics.optimization_priority = self._prioritize_optimizations(metrics)
            metrics.cost_reduction_opportunities = self._identify_cost_reduction_opportunities(metrics)
            
            # 8. Scalability Cost Analysis
            metrics.scaling_cost_factor = self._calculate_scaling_cost_factor(metrics, len(original_inputs))
            metrics.cost_per_additional_input = self._calculate_cost_per_additional_input(metrics, len(original_inputs))
            
            logger.info(f"Cost-efficiency evaluation completed. Total cost: ${metrics.total_processing_cost:.4f}")
            
        except Exception as e:
            logger.error(f"Cost-efficiency evaluation failed: {e}")
        
        return metrics
    
    def _calculate_computational_cost(
        self, 
        processing_metadata: Dict[str, Any], 
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate computational cost based on processing time and complexity"""
        try:
            processing_time = processing_metadata.get('total_processing_time', 0.0)
            
            # Estimate CPU utilization
            cpu_utilization = 1.0  # Default assumption: full CPU usage
            if performance_metrics and 'cpu_utilization' in performance_metrics:
                cpu_util_data = performance_metrics['cpu_utilization']
                if isinstance(cpu_util_data, dict):
                    cpu_utilization = cpu_util_data.get('average_cpu', 100.0) / 100.0
            
            # Calculate cost
            computational_cost = processing_time * cpu_utilization * self.cost_models['cpu_cost_per_second']
            
            return computational_cost
            
        except Exception as e:
            logger.error(f"Error calculating computational cost: {e}")
            return 0.0
    
    def _calculate_time_cost(self, processing_metadata: Dict[str, Any]) -> float:
        """Calculate time cost (opportunity cost of processing time)"""
        try:
            processing_time = processing_metadata.get('total_processing_time', 0.0)
            
            # Time cost is proportional to processing time
            # Using a simple linear model
            time_cost = processing_time * 0.005  # $0.005 per second of processing time
            
            return time_cost
            
        except Exception as e:
            logger.error(f"Error calculating time cost: {e}")
            return 0.0
    
    def _calculate_memory_cost(
        self, 
        processing_metadata: Dict[str, Any], 
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate memory usage cost"""
        try:
            processing_time = processing_metadata.get('total_processing_time', 0.0)
            
            # Estimate memory usage
            memory_usage_mb = 100.0  # Default estimate
            if performance_metrics and 'average_memory_usage' in performance_metrics:
                memory_usage_mb = performance_metrics['average_memory_usage']
            
            # Calculate memory cost
            memory_cost = memory_usage_mb * processing_time * self.cost_models['memory_cost_per_mb_second']
            
            return memory_cost
            
        except Exception as e:
            logger.error(f"Error calculating memory cost: {e}")
            return 0.0
    
    def _calculate_quality_per_cost(self, quality: float, total_cost: float) -> float:
        """Calculate quality achieved per unit cost"""
        if total_cost <= 0:
            return 0.0
        
        return quality / total_cost
    
    def _calculate_cost_per_part(self, total_cost: float, consensus_output: Dict[str, Any]) -> float:
        """Calculate cost per consensus part generated"""
        parts_count = consensus_output.get('consensus_parts_count', 1)
        
        if parts_count <= 0:
            return total_cost
        
        return total_cost / parts_count
    
    def _calculate_efficiency_ratio(self, metrics: CostEfficiencyMetrics, quality: float) -> float:
        """Calculate overall efficiency ratio"""
        try:
            # Efficiency combines quality achievement with cost minimization
            if metrics.total_processing_cost <= 0:
                return 0.0
            
            # Normalize quality and cost for comparison
            # Higher quality is better, lower cost is better
            quality_score = quality  # Already 0-1
            cost_score = 1.0 / (1.0 + metrics.total_processing_cost)  # Inverse of cost + 1
            
            # Weighted combination
            efficiency_ratio = (quality_score * 0.7) + (cost_score * 0.3)
            
            return efficiency_ratio
            
        except Exception as e:
            logger.error(f"Error calculating efficiency ratio: {e}")
            return 0.0
    
    def _calculate_cpu_cost_efficiency(
        self, 
        performance_metrics: Optional[Dict[str, Any]] = None, 
        processing_metadata: Dict[str, Any] = {}
    ) -> float:
        """Calculate CPU cost efficiency"""
        try:
            if not performance_metrics or 'cpu_utilization' not in performance_metrics:
                return 0.5  # Default moderate efficiency
            
            cpu_util_data = performance_metrics['cpu_utilization']
            if not isinstance(cpu_util_data, dict):
                return 0.5
            
            avg_cpu = cpu_util_data.get('average_cpu', 50.0)
            
            # Efficiency is optimal around 70-90% CPU usage
            if 70 <= avg_cpu <= 90:
                efficiency = 1.0
            elif avg_cpu < 70:
                efficiency = avg_cpu / 70.0  # Underutilization penalty
            else:
                efficiency = max(0.1, 1.0 - (avg_cpu - 90) / 100.0)  # Overutilization penalty
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating CPU cost efficiency: {e}")
            return 0.5
    
    def _calculate_memory_cost_efficiency(
        self, 
        performance_metrics: Optional[Dict[str, Any]] = None, 
        processing_metadata: Dict[str, Any] = {}
    ) -> float:
        """Calculate memory cost efficiency"""
        try:
            if not performance_metrics or 'memory_efficiency_score' not in performance_metrics:
                return 0.5  # Default moderate efficiency
            
            # Use the memory efficiency score directly
            memory_efficiency = performance_metrics['memory_efficiency_score']
            return float(memory_efficiency)
            
        except Exception as e:
            logger.error(f"Error calculating memory cost efficiency: {e}")
            return 0.5
    
    def _calculate_io_cost_efficiency(
        self, 
        performance_metrics: Optional[Dict[str, Any]] = None, 
        processing_metadata: Dict[str, Any] = {}
    ) -> float:
        """Calculate I/O cost efficiency"""
        try:
            if not performance_metrics or 'io_operations' not in performance_metrics:
                return 0.8  # Default good efficiency (I/O usually not a bottleneck)
            
            io_ops = performance_metrics['io_operations']
            if not isinstance(io_ops, dict):
                return 0.8
            
            # Simple heuristic: fewer operations per unit time is more efficient
            processing_time = processing_metadata.get('total_processing_time', 1.0)
            total_ops = io_ops.get('read_count', 0) + io_ops.get('write_count', 0)
            
            ops_per_second = total_ops / processing_time if processing_time > 0 else 0
            
            # Efficiency decreases with high I/O rate
            if ops_per_second < 100:
                efficiency = 1.0
            else:
                efficiency = max(0.1, 1.0 - (ops_per_second - 100) / 1000.0)
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating I/O cost efficiency: {e}")
            return 0.8
    
    def _calculate_cost_vs_baseline(self, total_cost: float, input_count: int) -> float:
        """Calculate cost compared to simple baseline (averaging)"""
        try:
            # Baseline cost: simple averaging of inputs
            baseline_cost = input_count * self.cost_models['baseline_simple_average_cost']
            
            if baseline_cost <= 0:
                return 1.0
            
            # Return ratio: >1 means more expensive than baseline
            cost_ratio = total_cost / baseline_cost
            return cost_ratio
            
        except Exception as e:
            logger.error(f"Error calculating cost vs baseline: {e}")
            return 1.0
    
    def _calculate_quality_improvement_per_cost(self, quality: float, total_cost: float) -> float:
        """Calculate quality improvement per cost compared to baseline"""
        try:
            # Assume baseline quality is 0.6 (simple averaging)
            baseline_quality = 0.6
            quality_improvement = quality - baseline_quality
            
            if total_cost <= 0:
                return 0.0
            
            improvement_per_cost = quality_improvement / total_cost
            return max(0.0, improvement_per_cost)  # Only positive improvements count
            
        except Exception as e:
            logger.error(f"Error calculating quality improvement per cost: {e}")
            return 0.0
    
    def _calculate_roi(self, quality: float, total_cost: float) -> float:
        """Calculate return on investment"""
        try:
            # ROI = (Quality Benefit - Cost) / Cost
            # Assume quality benefit in dollar terms (arbitrary scaling)
            quality_benefit = quality * 1.0  # $1 per quality point
            
            if total_cost <= 0:
                return 0.0
            
            roi = (quality_benefit - total_cost) / total_cost
            return roi
            
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return 0.0
    
    def _calculate_cost_benefit_ratio(self, quality: float, total_cost: float) -> float:
        """Calculate cost-benefit ratio"""
        try:
            # Benefit = quality achieved
            # Cost = total processing cost
            if total_cost <= 0:
                return 0.0
            
            cost_benefit_ratio = quality / total_cost
            return cost_benefit_ratio
            
        except Exception as e:
            logger.error(f"Error calculating cost-benefit ratio: {e}")
            return 0.0
    
    def _calculate_marginal_cost_per_improvement(self, metrics: CostEfficiencyMetrics) -> float:
        """Calculate marginal cost per unit of quality improvement"""
        try:
            # This is an approximation - would need historical data for true marginal analysis
            baseline_quality = 0.6
            baseline_cost = self.cost_models['baseline_simple_average_cost']
            
            quality_improvement = metrics.quality_per_cost * metrics.total_processing_cost - baseline_quality
            cost_increase = metrics.total_processing_cost - baseline_cost
            
            if quality_improvement <= 0:
                return float('inf')  # No improvement
            
            marginal_cost = cost_increase / quality_improvement
            return marginal_cost
            
        except Exception as e:
            logger.error(f"Error calculating marginal cost per improvement: {e}")
            return 0.0
    
    def _analyze_component_costs(
        self, 
        processing_metadata: Dict[str, Any], 
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Analyze costs by component"""
        component_costs = {}
        
        try:
            total_time = processing_metadata.get('total_processing_time', 0.0)
            
            if performance_metrics and 'component_processing_times' in performance_metrics:
                component_times = performance_metrics['component_processing_times']
                
                for component, timing_data in component_times.items():
                    if isinstance(timing_data, dict):
                        component_time = timing_data.get('total_time', 0.0)
                    else:
                        component_time = sum(timing_data) if isinstance(timing_data, list) else 0.0
                    
                    # Estimate component cost based on time
                    component_cost = component_time * self.cost_models['cpu_cost_per_second']
                    component_costs[component] = component_cost
            else:
                # Fallback: estimate based on known expensive components
                expensive_components = {
                    'MCTS': 0.6,  # Monte Carlo is expensive
                    'SBERT': 0.2,  # Semantic embeddings
                    'BFT': 0.1,   # Consensus algorithms
                    'ICE': 0.1    # Iterative refinement
                }
                
                total_cost = total_time * self.cost_models['cpu_cost_per_second']
                for component, ratio in expensive_components.items():
                    component_costs[component] = total_cost * ratio
                    
        except Exception as e:
            logger.error(f"Error analyzing component costs: {e}")
        
        return component_costs
    
    def _calculate_cost_distribution(self, component_costs: Dict[str, float], total_cost: float) -> Dict[str, float]:
        """Calculate cost distribution as percentages"""
        if total_cost <= 0:
            return {}
        
        cost_distribution = {}
        for component, cost in component_costs.items():
            percentage = (cost / total_cost) * 100.0
            cost_distribution[component] = percentage
        
        return cost_distribution
    
    def _identify_potential_savings(self, metrics: CostEfficiencyMetrics) -> float:
        """Identify potential cost savings"""
        try:
            potential_savings = 0.0
            
            # Savings from inefficient resource utilization
            if metrics.cpu_cost_efficiency < 0.7:
                cpu_savings = metrics.computational_cost * (0.7 - metrics.cpu_cost_efficiency)
                potential_savings += cpu_savings
            
            if metrics.memory_cost_efficiency < 0.7:
                memory_savings = metrics.memory_cost * (0.7 - metrics.memory_cost_efficiency)
                potential_savings += memory_savings
            
            # Savings from removing expensive components with low impact
            # (This would need more detailed analysis of component necessity)
            if 'MCTS' in metrics.component_costs:
                # Based on our analysis, MCTS provides minimal benefit
                mcts_savings = metrics.component_costs['MCTS'] * 0.9  # 90% savings potential
                potential_savings += mcts_savings
            
            return potential_savings
            
        except Exception as e:
            logger.error(f"Error identifying potential savings: {e}")
            return 0.0
    
    def _prioritize_optimizations(self, metrics: CostEfficiencyMetrics) -> List[str]:
        """Prioritize optimization opportunities"""
        optimizations = []
        
        try:
            # Priority based on cost impact and feasibility
            priority_candidates = []
            
            # Check CPU efficiency
            if metrics.cpu_cost_efficiency < 0.6:
                cpu_impact = metrics.computational_cost * (0.8 - metrics.cpu_cost_efficiency)
                priority_candidates.append(('CPU utilization optimization', cpu_impact))
            
            # Check memory efficiency  
            if metrics.memory_cost_efficiency < 0.6:
                memory_impact = metrics.memory_cost * (0.8 - metrics.memory_cost_efficiency)
                priority_candidates.append(('Memory usage optimization', memory_impact))
            
            # Check expensive components
            for component, cost in metrics.component_costs.items():
                if cost > metrics.total_processing_cost * 0.3:  # >30% of total cost
                    priority_candidates.append((f'{component} component optimization', cost * 0.5))
            
            # Sort by impact and take top priorities
            priority_candidates.sort(key=lambda x: x[1], reverse=True)
            optimizations = [opt[0] for opt in priority_candidates[:5]]
            
        except Exception as e:
            logger.error(f"Error prioritizing optimizations: {e}")
        
        return optimizations
    
    def _identify_cost_reduction_opportunities(self, metrics: CostEfficiencyMetrics) -> Dict[str, float]:
        """Identify specific cost reduction opportunities"""
        opportunities = {}
        
        try:
            # Component removal opportunities
            for component, cost in metrics.component_costs.items():
                if component == 'MCTS':
                    # Based on analysis showing MCTS provides no benefit
                    opportunities[f'Remove {component}'] = cost * 0.95
                elif cost > metrics.total_processing_cost * 0.2:
                    # Expensive components that could be optimized
                    opportunities[f'Optimize {component}'] = cost * 0.3
            
            # Resource utilization improvements
            if metrics.cpu_cost_efficiency < 0.7:
                opportunities['Improve CPU utilization'] = metrics.computational_cost * 0.2
            
            if metrics.memory_cost_efficiency < 0.7:
                opportunities['Optimize memory usage'] = metrics.memory_cost * 0.2
            
            # Algorithmic improvements
            if metrics.cost_vs_baseline > 2.0:
                opportunities['Algorithm simplification'] = metrics.total_processing_cost * 0.3
                
        except Exception as e:
            logger.error(f"Error identifying cost reduction opportunities: {e}")
        
        return opportunities
    
    def _calculate_scaling_cost_factor(self, metrics: CostEfficiencyMetrics, input_count: int) -> float:
        """Calculate how cost scales with input size"""
        try:
            # Estimate scaling based on current cost per input
            if input_count <= 0:
                return 1.0
            
            cost_per_input = metrics.total_processing_cost / input_count
            
            # Compare with expected linear scaling
            expected_cost_per_input = 0.05  # Baseline expectation
            
            scaling_factor = cost_per_input / expected_cost_per_input
            return scaling_factor
            
        except Exception as e:
            logger.error(f"Error calculating scaling cost factor: {e}")
            return 1.0
    
    def _calculate_cost_per_additional_input(self, metrics: CostEfficiencyMetrics, input_count: int) -> float:
        """Calculate marginal cost of processing additional input"""
        try:
            if input_count <= 0:
                return 0.0
            
            # Simple approximation: current cost divided by input count
            # In reality, this would need marginal analysis
            marginal_cost = metrics.total_processing_cost / input_count
            
            return marginal_cost
            
        except Exception as e:
            logger.error(f"Error calculating cost per additional input: {e}")
            return 0.0

def create_cost_efficiency_report(metrics: CostEfficiencyMetrics, output_file: Optional[str] = None) -> str:
    """Create a comprehensive cost-efficiency report"""
    
    report = f"""
# Cost-Efficiency Analysis Report

## Cost Summary

### Processing Costs
- **Computational Cost**: ${metrics.computational_cost:.4f}
- **Time Cost**: ${metrics.time_cost:.4f}
- **Memory Cost**: ${metrics.memory_cost:.4f}
- **Total Processing Cost**: ${metrics.total_processing_cost:.4f}

### Cost Efficiency Metrics
- **Quality per Cost**: {metrics.quality_per_cost:.3f}
- **Cost per Consensus Part**: ${metrics.cost_per_consensus_part:.4f}
- **Overall Efficiency Ratio**: {metrics.efficiency_ratio:.3f}

## Resource Cost Efficiency

### Resource Utilization Efficiency
- **CPU Cost Efficiency**: {metrics.cpu_cost_efficiency:.3f}
- **Memory Cost Efficiency**: {metrics.memory_cost_efficiency:.3f}
- **I/O Cost Efficiency**: {metrics.io_cost_efficiency:.3f}

## Comparative Analysis

### Cost Comparison
- **Cost vs Baseline**: {metrics.cost_vs_baseline:.2f}x
- **Quality Improvement per Cost**: {metrics.quality_improvement_per_cost:.3f}

## Return on Investment

### ROI Metrics
- **Return on Investment**: {metrics.return_on_investment:.3f}
- **Cost-Benefit Ratio**: {metrics.cost_benefit_ratio:.3f}
- **Marginal Cost per Improvement**: ${metrics.marginal_cost_per_improvement:.4f}

## Component Cost Analysis

### Cost Breakdown by Component
{_format_component_costs(metrics.component_costs, metrics.cost_distribution)}

## Scalability Cost Analysis

### Scaling Metrics
- **Scaling Cost Factor**: {metrics.scaling_cost_factor:.2f}
- **Cost per Additional Input**: ${metrics.cost_per_additional_input:.4f}

## Optimization Opportunities

### Potential Savings
- **Total Potential Savings**: ${metrics.potential_savings:.4f}

### Optimization Priorities
{_format_optimization_priorities(metrics.optimization_priority)}

### Cost Reduction Opportunities
{_format_cost_reduction_opportunities(metrics.cost_reduction_opportunities)}

## Cost-Efficiency Assessment

{_generate_cost_efficiency_assessment(metrics)}

## Recommendations

{_generate_cost_efficiency_recommendations(metrics)}
"""
    
    if output_file:
        # Remove Unicode characters for Windows compatibility
        safe_report = report.replace('✓', 'SUCCESS').replace('⚠', 'WARNING').replace('✗', 'FAILED')
        Path(output_file).write_text(safe_report, encoding='utf-8')
        logger.info(f"Cost-efficiency report saved to {output_file}")
    
    return report

def _format_component_costs(component_costs: Dict[str, float], cost_distribution: Dict[str, float]) -> str:
    """Format component cost breakdown"""
    if not component_costs:
        return "No component cost data available"
    
    lines = []
    for component in sorted(component_costs.keys(), key=lambda x: component_costs[x], reverse=True):
        cost = component_costs[component]
        percentage = cost_distribution.get(component, 0.0)
        lines.append(f"- **{component}**: ${cost:.4f} ({percentage:.1f}%)")
    
    return "\n".join(lines)

def _format_optimization_priorities(optimization_priorities: List[str]) -> str:
    """Format optimization priorities"""
    if not optimization_priorities:
        return "No specific optimization priorities identified"
    
    lines = []
    for i, priority in enumerate(optimization_priorities, 1):
        lines.append(f"{i}. {priority}")
    
    return "\n".join(lines)

def _format_cost_reduction_opportunities(cost_reduction_opportunities: Dict[str, float]) -> str:
    """Format cost reduction opportunities"""
    if not cost_reduction_opportunities:
        return "No specific cost reduction opportunities identified"
    
    lines = []
    for opportunity, savings in sorted(cost_reduction_opportunities.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- **{opportunity}**: ${savings:.4f} potential savings")
    
    return "\n".join(lines)

def _generate_cost_efficiency_assessment(metrics: CostEfficiencyMetrics) -> str:
    """Generate overall cost-efficiency assessment"""
    assessment_lines = []
    
    # Overall efficiency assessment
    if metrics.efficiency_ratio >= 0.8:
        assessment_lines.append("✓ EXCELLENT cost efficiency")
    elif metrics.efficiency_ratio >= 0.6:
        assessment_lines.append("✓ GOOD cost efficiency")
    else:
        assessment_lines.append("⚠ POOR cost efficiency")
    
    # Cost vs baseline assessment
    if metrics.cost_vs_baseline <= 1.5:
        assessment_lines.append("✓ REASONABLE cost compared to baseline")
    elif metrics.cost_vs_baseline <= 3.0:
        assessment_lines.append("⚠ MODERATE cost premium over baseline")
    else:
        assessment_lines.append("⚠ HIGH cost premium over baseline")
    
    # ROI assessment
    if metrics.return_on_investment >= 0.5:
        assessment_lines.append("✓ POSITIVE return on investment")
    elif metrics.return_on_investment >= 0.0:
        assessment_lines.append("✓ BREAK-EVEN return on investment")
    else:
        assessment_lines.append("⚠ NEGATIVE return on investment")
    
    # Resource efficiency assessment
    avg_resource_efficiency = (metrics.cpu_cost_efficiency + metrics.memory_cost_efficiency + metrics.io_cost_efficiency) / 3.0
    if avg_resource_efficiency >= 0.8:
        assessment_lines.append("✓ EXCELLENT resource utilization")
    elif avg_resource_efficiency >= 0.6:
        assessment_lines.append("✓ GOOD resource utilization")
    else:
        assessment_lines.append("⚠ POOR resource utilization")
    
    return "\n".join(assessment_lines)

def _generate_cost_efficiency_recommendations(metrics: CostEfficiencyMetrics) -> str:
    """Generate cost-efficiency optimization recommendations"""
    recommendations = []
    
    if metrics.efficiency_ratio < 0.6:
        recommendations.append("- Prioritize overall efficiency improvements to reduce cost per quality unit")
    
    if metrics.cost_vs_baseline > 2.0:
        recommendations.append("- Evaluate if complex consensus is justified compared to simple averaging")
    
    if metrics.return_on_investment < 0.0:
        recommendations.append("- Review system design - current costs exceed benefits")
    
    if metrics.cpu_cost_efficiency < 0.6:
        recommendations.append("- Optimize CPU utilization to reduce computational costs")
    
    if metrics.memory_cost_efficiency < 0.6:
        recommendations.append("- Implement memory optimization strategies")
    
    if metrics.potential_savings > metrics.total_processing_cost * 0.2:
        recommendations.append(f"- Significant cost savings possible (${metrics.potential_savings:.4f}) through optimization")
    
    # Component-specific recommendations
    for component, cost in metrics.component_costs.items():
        if cost > metrics.total_processing_cost * 0.4:  # >40% of total cost
            recommendations.append(f"- Consider optimizing or removing {component} component (high cost impact)")
    
    if metrics.scaling_cost_factor > 2.0:
        recommendations.append("- Improve algorithmic complexity for better scaling")
    
    if not recommendations:
        recommendations.append("- Continue monitoring cost efficiency metrics")
        recommendations.append("- Test with varying input sizes to validate cost model")
    
    return "\n".join(recommendations)