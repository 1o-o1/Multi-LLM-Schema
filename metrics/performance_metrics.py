#!/usr/bin/env python3
"""
Performance Metrics Tracking (Section 9.2)

This module implements performance-specific metrics:
- Processing time breakdown and efficiency analysis
- Memory usage tracking and optimization insights
- Throughput measurements and scalability assessment
- Resource utilization and bottleneck identification
"""

import json
import time
import logging
import statistics
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Try to import psutil, provide fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available, using mock performance tracking")

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Time Analysis
    total_processing_time: float = 0.0
    component_processing_times: Dict[str, float] = field(default_factory=dict)
    initialization_time: float = 0.0
    consensus_time: float = 0.0
    finalization_time: float = 0.0
    
    # Memory Analysis
    peak_memory_usage: float = 0.0  # MB
    average_memory_usage: float = 0.0  # MB
    memory_efficiency_score: float = 0.0
    memory_growth_rate: float = 0.0
    
    # Throughput Analysis
    parts_per_second: float = 0.0
    files_per_second: float = 0.0
    consensus_throughput: float = 0.0
    data_processing_rate: float = 0.0  # MB/s
    
    # Resource Utilization
    cpu_utilization: Dict[str, float] = field(default_factory=dict)
    io_operations: Dict[str, int] = field(default_factory=dict)
    network_usage: Dict[str, float] = field(default_factory=dict)
    
    # Scalability Metrics
    scalability_score: float = 0.0
    performance_degradation: float = 0.0
    efficiency_ratio: float = 0.0
    
    # Bottleneck Analysis
    bottleneck_components: List[str] = field(default_factory=list)
    optimization_potential: Dict[str, float] = field(default_factory=dict)
    
    # Quality-Performance Trade-off
    quality_per_second: float = 0.0
    efficiency_index: float = 0.0
    
    def __post_init__(self):
        if not self.component_processing_times:
            self.component_processing_times = {}
        if not self.cpu_utilization:
            self.cpu_utilization = {}
        if not self.io_operations:
            self.io_operations = {}
        if not self.network_usage:
            self.network_usage = {}
        if not self.optimization_potential:
            self.optimization_potential = {}

class PerformanceTracker:
    """Performance tracking and analysis system"""
    
    def __init__(self):
        """Initialize performance tracker"""
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.component_timings = defaultdict(list)
        self.process = None
        
        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process()
            except Exception as e:
                logger.warning(f"Failed to initialize psutil process: {e}")
                self.process = None
        
    def start_tracking(self) -> None:
        """Start performance tracking"""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.component_timings = defaultdict(list)
        
        # Initial resource snapshot
        self._take_resource_snapshot()
        
    def track_component(self, component_name: str, duration: float) -> None:
        """Track individual component performance"""
        self.component_timings[component_name].append(duration)
        logger.debug(f"Component {component_name} took {duration:.3f}s")
        
    def _take_resource_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current resource usage"""
        try:
            if self.process:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                snapshot = {
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent,
                    'num_threads': self.process.num_threads(),
                    'io_counters': self.process.io_counters()._asdict() if hasattr(self.process, 'io_counters') else {}
                }
                
                self.memory_samples.append(snapshot['memory_mb'])
                self.cpu_samples.append(snapshot['cpu_percent'])
                
                return snapshot
            else:
                # Mock data when psutil is not available
                snapshot = {
                    'timestamp': time.time(),
                    'memory_mb': 150.0,  # Mock memory usage
                    'cpu_percent': 65.0,  # Mock CPU usage
                    'num_threads': 4,
                    'io_counters': {'read_count': 100, 'write_count': 50, 'read_bytes': 1024*1024, 'write_bytes': 512*1024}
                }
                
                self.memory_samples.append(snapshot['memory_mb'])
                self.cpu_samples.append(snapshot['cpu_percent'])
                
                return snapshot
            
        except Exception as e:
            logger.error(f"Error taking resource snapshot: {e}")
            return {}
    
    def evaluate_performance(
        self,
        consensus_output: Dict[str, Any],
        processing_metadata: Dict[str, Any],
        original_inputs: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """
        Comprehensive performance evaluation
        
        Args:
            consensus_output: Output from consensus system
            processing_metadata: Metadata about consensus process
            original_inputs: Original input files
            
        Returns:
            PerformanceMetrics with all computed metrics
        """
        logger.info("Starting performance evaluation")
        
        metrics = PerformanceMetrics()
        
        try:
            # Take final resource snapshot
            final_snapshot = self._take_resource_snapshot()
            
            # 1. Time Analysis
            metrics.total_processing_time = processing_metadata.get('total_processing_time', 0.0)
            metrics.component_processing_times = self._analyze_component_times()
            metrics.initialization_time = self._calculate_initialization_time(processing_metadata)
            metrics.consensus_time = self._calculate_consensus_time(processing_metadata)
            metrics.finalization_time = self._calculate_finalization_time(processing_metadata)
            
            # 2. Memory Analysis
            if self.memory_samples:
                metrics.peak_memory_usage = max(self.memory_samples)
                metrics.average_memory_usage = statistics.mean(self.memory_samples)
                metrics.memory_efficiency_score = self._calculate_memory_efficiency()
                metrics.memory_growth_rate = self._calculate_memory_growth_rate()
            
            # 3. Throughput Analysis
            metrics.parts_per_second = self._calculate_parts_throughput(consensus_output, metrics.total_processing_time)
            metrics.files_per_second = self._calculate_files_throughput(original_inputs, metrics.total_processing_time)
            metrics.consensus_throughput = self._calculate_consensus_throughput(consensus_output, metrics.total_processing_time)
            metrics.data_processing_rate = self._calculate_data_processing_rate(original_inputs, metrics.total_processing_time)
            
            # 4. Resource Utilization
            if self.cpu_samples:
                metrics.cpu_utilization = self._analyze_cpu_utilization()
            metrics.io_operations = self._analyze_io_operations(final_snapshot)
            
            # 5. Scalability Analysis
            metrics.scalability_score = self._calculate_scalability_score(metrics, len(original_inputs))
            metrics.performance_degradation = self._calculate_performance_degradation(metrics)
            metrics.efficiency_ratio = self._calculate_efficiency_ratio(metrics)
            
            # 6. Bottleneck Analysis
            metrics.bottleneck_components = self._identify_bottlenecks()
            metrics.optimization_potential = self._assess_optimization_potential(metrics)
            
            # 7. Quality-Performance Trade-off
            overall_quality = float(consensus_output.get('quality_metrics', {}).get('overall_quality', 0.0))
            metrics.quality_per_second = self._calculate_quality_per_second(overall_quality, metrics.total_processing_time)
            metrics.efficiency_index = self._calculate_efficiency_index(metrics, overall_quality)
            
            logger.info(f"Performance evaluation completed. Efficiency index: {metrics.efficiency_index:.3f}")
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
        
        return metrics
    
    def _analyze_component_times(self) -> Dict[str, float]:
        """Analyze component processing times"""
        component_times = {}
        
        for component, times in self.component_timings.items():
            if times:
                component_times[component] = {
                    'total_time': sum(times),
                    'average_time': statistics.mean(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'call_count': len(times)
                }
        
        return component_times
    
    def _calculate_initialization_time(self, processing_metadata: Dict[str, Any]) -> float:
        """Calculate initialization time"""
        # Look for specific initialization metrics
        init_time = processing_metadata.get('initialization_time', 0.0)
        
        # Fallback: estimate from component times
        if init_time == 0.0 and self.component_timings:
            # First 10% of total time is typically initialization
            total_time = processing_metadata.get('total_processing_time', 0.0)
            init_time = total_time * 0.1
        
        return init_time
    
    def _calculate_consensus_time(self, processing_metadata: Dict[str, Any]) -> float:
        """Calculate core consensus processing time"""
        # Look for consensus-specific timing
        consensus_time = processing_metadata.get('consensus_processing_time', 0.0)
        
        # Fallback: estimate from total time minus initialization and finalization
        if consensus_time == 0.0:
            total_time = processing_metadata.get('total_processing_time', 0.0)
            init_time = self._calculate_initialization_time(processing_metadata)
            final_time = self._calculate_finalization_time(processing_metadata)
            consensus_time = max(0.0, total_time - init_time - final_time)
        
        return consensus_time
    
    def _calculate_finalization_time(self, processing_metadata: Dict[str, Any]) -> float:
        """Calculate finalization time"""
        # Look for specific finalization metrics
        final_time = processing_metadata.get('finalization_time', 0.0)
        
        # Fallback: estimate as small percentage of total time
        if final_time == 0.0:
            total_time = processing_metadata.get('total_processing_time', 0.0)
            final_time = total_time * 0.05  # Last 5% typically finalization
        
        return final_time
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self.memory_samples:
            return 0.0
        
        try:
            peak_memory = max(self.memory_samples)
            avg_memory = statistics.mean(self.memory_samples)
            
            # Efficiency is better when average is close to peak (consistent usage)
            # and when total usage is reasonable
            consistency_score = avg_memory / peak_memory if peak_memory > 0 else 0.0
            
            # Penalize excessive memory usage (> 1GB is concerning for this task)
            usage_penalty = max(0.0, 1.0 - (peak_memory - 1024) / 1024) if peak_memory > 1024 else 1.0
            
            efficiency = consistency_score * usage_penalty
            return min(1.0, max(0.0, efficiency))
            
        except Exception as e:
            logger.error(f"Error calculating memory efficiency: {e}")
            return 0.0
    
    def _calculate_memory_growth_rate(self) -> float:
        """Calculate memory growth rate over time"""
        if len(self.memory_samples) < 2:
            return 0.0
        
        try:
            # Linear regression to find growth trend
            x = np.arange(len(self.memory_samples))
            y = np.array(self.memory_samples)
            
            # Simple linear fit
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize by average memory usage
            avg_memory = statistics.mean(self.memory_samples)
            growth_rate = slope / avg_memory if avg_memory > 0 else 0.0
            
            return growth_rate
            
        except Exception as e:
            logger.error(f"Error calculating memory growth rate: {e}")
            return 0.0
    
    def _calculate_parts_throughput(self, consensus_output: Dict[str, Any], processing_time: float) -> float:
        """Calculate consensus parts processed per second"""
        if processing_time <= 0:
            return 0.0
        
        parts_count = consensus_output.get('consensus_parts_count', 0)
        return parts_count / processing_time
    
    def _calculate_files_throughput(self, original_inputs: List[Dict[str, Any]], processing_time: float) -> float:
        """Calculate files processed per second"""
        if processing_time <= 0:
            return 0.0
        
        return len(original_inputs) / processing_time
    
    def _calculate_consensus_throughput(self, consensus_output: Dict[str, Any], processing_time: float) -> float:
        """Calculate overall consensus throughput"""
        if processing_time <= 0:
            return 0.0
        
        # Combine parts and quality for throughput metric
        parts_count = consensus_output.get('consensus_parts_count', 0)
        quality = float(consensus_output.get('quality_metrics', {}).get('overall_quality', 0.0))
        
        effective_throughput = (parts_count * quality) / processing_time
        return effective_throughput
    
    def _calculate_data_processing_rate(self, original_inputs: List[Dict[str, Any]], processing_time: float) -> float:
        """Calculate data processing rate in MB/s"""
        if processing_time <= 0:
            return 0.0
        
        try:
            total_size_mb = 0.0
            for input_data in original_inputs:
                # Estimate size from JSON string length
                json_str = json.dumps(input_data, default=str)
                size_mb = len(json_str.encode('utf-8')) / 1024 / 1024
                total_size_mb += size_mb
            
            return total_size_mb / processing_time
            
        except Exception as e:
            logger.error(f"Error calculating data processing rate: {e}")
            return 0.0
    
    def _analyze_cpu_utilization(self) -> Dict[str, float]:
        """Analyze CPU utilization patterns"""
        if not self.cpu_samples:
            return {}
        
        return {
            'average_cpu': statistics.mean(self.cpu_samples),
            'peak_cpu': max(self.cpu_samples),
            'cpu_efficiency': statistics.mean(self.cpu_samples) / 100.0,  # Normalize to 0-1
            'cpu_variability': statistics.stdev(self.cpu_samples) if len(self.cpu_samples) > 1 else 0.0
        }
    
    def _analyze_io_operations(self, final_snapshot: Dict[str, Any]) -> Dict[str, int]:
        """Analyze I/O operations"""
        io_counters = final_snapshot.get('io_counters', {})
        
        return {
            'read_count': io_counters.get('read_count', 0),
            'write_count': io_counters.get('write_count', 0),
            'read_bytes': io_counters.get('read_bytes', 0),
            'write_bytes': io_counters.get('write_bytes', 0)
        }
    
    def _calculate_scalability_score(self, metrics: PerformanceMetrics, input_count: int) -> float:
        """Calculate scalability score based on input size"""
        try:
            # Baseline expectation: processing time scales linearly with input count
            expected_time_per_input = 2.0  # 2 seconds per input as baseline
            expected_total_time = input_count * expected_time_per_input
            
            actual_time = metrics.total_processing_time
            
            if expected_total_time <= 0:
                return 1.0
            
            # Scalability is better when actual time is less than expected
            scalability = expected_total_time / actual_time if actual_time > 0 else 0.0
            
            return min(1.0, max(0.0, scalability))
            
        except Exception as e:
            logger.error(f"Error calculating scalability score: {e}")
            return 0.0
    
    def _calculate_performance_degradation(self, metrics: PerformanceMetrics) -> float:
        """Calculate performance degradation indicators"""
        try:
            degradation_factors = []
            
            # Memory growth indicates degradation
            if metrics.memory_growth_rate > 0.1:  # 10% growth
                degradation_factors.append(metrics.memory_growth_rate)
            
            # High CPU variability indicates inefficiency
            cpu_variability = metrics.cpu_utilization.get('cpu_variability', 0.0)
            if cpu_variability > 20.0:  # High variance in CPU usage
                degradation_factors.append(cpu_variability / 100.0)
            
            # Poor memory efficiency indicates problems
            if metrics.memory_efficiency_score < 0.7:
                degradation_factors.append(1.0 - metrics.memory_efficiency_score)
            
            return statistics.mean(degradation_factors) if degradation_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating performance degradation: {e}")
            return 0.0
    
    def _calculate_efficiency_ratio(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall efficiency ratio"""
        try:
            efficiency_factors = []
            
            # Time efficiency (faster is better)
            if metrics.total_processing_time > 0:
                time_efficiency = min(1.0, 10.0 / metrics.total_processing_time)  # 10s as reference
                efficiency_factors.append(time_efficiency)
            
            # Memory efficiency
            efficiency_factors.append(metrics.memory_efficiency_score)
            
            # CPU efficiency
            cpu_efficiency = metrics.cpu_utilization.get('cpu_efficiency', 0.0)
            efficiency_factors.append(cpu_efficiency)
            
            # Throughput efficiency (higher is better)
            if metrics.consensus_throughput > 0:
                throughput_efficiency = min(1.0, metrics.consensus_throughput / 10.0)  # 10 as reference
                efficiency_factors.append(throughput_efficiency)
            
            return statistics.mean(efficiency_factors) if efficiency_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating efficiency ratio: {e}")
            return 0.0
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        try:
            # Analyze component times to find bottlenecks
            for component, timing_data in self.component_timings.items():
                if isinstance(timing_data, dict):
                    avg_time = timing_data.get('average_time', 0.0)
                    if avg_time > 2.0:  # Components taking > 2s on average
                        bottlenecks.append(f"{component} (avg: {avg_time:.2f}s)")
                elif isinstance(timing_data, list) and timing_data:
                    avg_time = statistics.mean(timing_data)
                    if avg_time > 2.0:
                        bottlenecks.append(f"{component} (avg: {avg_time:.2f}s)")
            
            # Check memory bottlenecks
            if self.memory_samples and max(self.memory_samples) > 1024:  # > 1GB
                bottlenecks.append(f"High memory usage ({max(self.memory_samples):.0f}MB)")
            
            # Check CPU bottlenecks
            if self.cpu_samples and max(self.cpu_samples) > 90:  # > 90% CPU
                bottlenecks.append(f"High CPU usage ({max(self.cpu_samples):.1f}%)")
                
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
        
        return bottlenecks
    
    def _assess_optimization_potential(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Assess optimization potential for different aspects"""
        optimization_potential = {}
        
        try:
            # Memory optimization potential
            if metrics.memory_efficiency_score < 0.8:
                optimization_potential['memory'] = 1.0 - metrics.memory_efficiency_score
            
            # CPU optimization potential
            cpu_efficiency = metrics.cpu_utilization.get('cpu_efficiency', 1.0)
            if cpu_efficiency < 0.8:
                optimization_potential['cpu'] = 1.0 - cpu_efficiency
            
            # Time optimization potential
            if metrics.efficiency_ratio < 0.8:
                optimization_potential['time'] = 1.0 - metrics.efficiency_ratio
            
            # Throughput optimization potential
            if metrics.consensus_throughput < 5.0:  # Arbitrary threshold
                optimization_potential['throughput'] = 0.5
                
        except Exception as e:
            logger.error(f"Error assessing optimization potential: {e}")
        
        return optimization_potential
    
    def _calculate_quality_per_second(self, overall_quality: float, processing_time: float) -> float:
        """Calculate quality achieved per second of processing"""
        if processing_time <= 0:
            return 0.0
        
        return overall_quality / processing_time
    
    def _calculate_efficiency_index(self, metrics: PerformanceMetrics, overall_quality: float) -> float:
        """Calculate comprehensive efficiency index"""
        try:
            efficiency_components = []
            
            # Quality efficiency (quality per unit time)
            quality_efficiency = metrics.quality_per_second * 10  # Scale up
            efficiency_components.append(min(1.0, quality_efficiency))
            
            # Resource efficiency
            efficiency_components.append(metrics.efficiency_ratio)
            
            # Scalability efficiency
            efficiency_components.append(metrics.scalability_score)
            
            # Memory efficiency
            efficiency_components.append(metrics.memory_efficiency_score)
            
            # Weighted average with emphasis on quality and time
            weights = [0.4, 0.3, 0.2, 0.1]
            efficiency_index = sum(w * e for w, e in zip(weights, efficiency_components))
            
            return min(1.0, max(0.0, efficiency_index))
            
        except Exception as e:
            logger.error(f"Error calculating efficiency index: {e}")
            return 0.0

def create_performance_report(metrics: PerformanceMetrics, output_file: Optional[str] = None) -> str:
    """Create a comprehensive performance report"""
    
    report = f"""
# Performance Analysis Report

## Overall Performance Summary

### Processing Time Analysis
- **Total Processing Time**: {metrics.total_processing_time:.2f}s
- **Initialization Time**: {metrics.initialization_time:.2f}s
- **Core Consensus Time**: {metrics.consensus_time:.2f}s
- **Finalization Time**: {metrics.finalization_time:.2f}s

### Memory Performance
- **Peak Memory Usage**: {metrics.peak_memory_usage:.1f}MB
- **Average Memory Usage**: {metrics.average_memory_usage:.1f}MB
- **Memory Efficiency Score**: {metrics.memory_efficiency_score:.3f}
- **Memory Growth Rate**: {metrics.memory_growth_rate:.3f}

### Throughput Metrics
- **Parts per Second**: {metrics.parts_per_second:.2f}
- **Files per Second**: {metrics.files_per_second:.2f}
- **Consensus Throughput**: {metrics.consensus_throughput:.2f}
- **Data Processing Rate**: {metrics.data_processing_rate:.2f}MB/s

## Resource Utilization

### CPU Usage
{_format_cpu_utilization(metrics.cpu_utilization)}

### I/O Operations
{_format_io_operations(metrics.io_operations)}

## Scalability and Efficiency

### Performance Scalability
- **Scalability Score**: {metrics.scalability_score:.3f}
- **Performance Degradation**: {metrics.performance_degradation:.3f}
- **Efficiency Ratio**: {metrics.efficiency_ratio:.3f}

### Quality-Performance Trade-off
- **Quality per Second**: {metrics.quality_per_second:.3f}
- **Overall Efficiency Index**: {metrics.efficiency_index:.3f}

## Bottleneck Analysis

### Identified Bottlenecks
{_format_bottlenecks(metrics.bottleneck_components)}

### Optimization Potential
{_format_optimization_potential(metrics.optimization_potential)}

## Component Performance Breakdown
{_format_component_times(metrics.component_processing_times)}

## Performance Assessment

{_generate_performance_assessment(metrics)}

## Optimization Recommendations

{_generate_performance_recommendations(metrics)}
"""
    
    if output_file:
        # Remove Unicode characters for Windows compatibility
        safe_report = report.replace('✓', 'SUCCESS').replace('⚠', 'WARNING').replace('✗', 'FAILED')
        Path(output_file).write_text(safe_report, encoding='utf-8')
        logger.info(f"Performance report saved to {output_file}")
    
    return report

def _format_cpu_utilization(cpu_utilization: Dict[str, float]) -> str:
    """Format CPU utilization data"""
    if not cpu_utilization:
        return "No CPU utilization data available"
    
    lines = []
    for metric, value in cpu_utilization.items():
        if isinstance(value, float):
            lines.append(f"- {metric.replace('_', ' ').title()}: {value:.2f}%")
    
    return "\n".join(lines) if lines else "No CPU data available"

def _format_io_operations(io_operations: Dict[str, int]) -> str:
    """Format I/O operations data"""
    if not io_operations:
        return "No I/O data available"
    
    lines = []
    for operation, count in io_operations.items():
        if 'bytes' in operation:
            # Convert bytes to MB for readability
            mb_value = count / 1024 / 1024
            lines.append(f"- {operation.replace('_', ' ').title()}: {mb_value:.2f}MB")
        else:
            lines.append(f"- {operation.replace('_', ' ').title()}: {count:,}")
    
    return "\n".join(lines)

def _format_bottlenecks(bottlenecks: List[str]) -> str:
    """Format bottleneck information"""
    if not bottlenecks:
        return "No significant bottlenecks identified"
    
    lines = []
    for bottleneck in bottlenecks:
        lines.append(f"- {bottleneck}")
    
    return "\n".join(lines)

def _format_optimization_potential(optimization_potential: Dict[str, float]) -> str:
    """Format optimization potential data"""
    if not optimization_potential:
        return "System appears well-optimized"
    
    lines = []
    for aspect, potential in sorted(optimization_potential.items(), key=lambda x: x[1], reverse=True):
        percentage = potential * 100
        lines.append(f"- {aspect.title()}: {percentage:.1f}% improvement potential")
    
    return "\n".join(lines)

def _format_component_times(component_times: Dict[str, Any]) -> str:
    """Format component timing information"""
    if not component_times:
        return "No component timing data available"
    
    lines = []
    for component, timing_data in sorted(component_times.items(), key=lambda x: x[1].get('total_time', 0) if isinstance(x[1], dict) else 0, reverse=True):
        if isinstance(timing_data, dict):
            total_time = timing_data.get('total_time', 0.0)
            avg_time = timing_data.get('average_time', 0.0)
            call_count = timing_data.get('call_count', 0)
            lines.append(f"- **{component}**: {total_time:.2f}s total, {avg_time:.3f}s avg ({call_count} calls)")
    
    return "\n".join(lines)

def _generate_performance_assessment(metrics: PerformanceMetrics) -> str:
    """Generate overall performance assessment"""
    assessment_lines = []
    
    # Efficiency assessment
    if metrics.efficiency_index >= 0.8:
        assessment_lines.append("✓ EXCELLENT overall efficiency")
    elif metrics.efficiency_index >= 0.6:
        assessment_lines.append("✓ GOOD overall efficiency")
    else:
        assessment_lines.append("⚠ LOW overall efficiency")
    
    # Memory assessment
    if metrics.memory_efficiency_score >= 0.8:
        assessment_lines.append("✓ EXCELLENT memory efficiency")
    elif metrics.memory_efficiency_score >= 0.6:
        assessment_lines.append("✓ ADEQUATE memory efficiency")
    else:
        assessment_lines.append("⚠ POOR memory efficiency")
    
    # Scalability assessment
    if metrics.scalability_score >= 0.8:
        assessment_lines.append("✓ EXCELLENT scalability")
    elif metrics.scalability_score >= 0.6:
        assessment_lines.append("✓ GOOD scalability")
    else:
        assessment_lines.append("⚠ LIMITED scalability")
    
    # Throughput assessment
    if metrics.consensus_throughput >= 10.0:
        assessment_lines.append("✓ HIGH consensus throughput")
    elif metrics.consensus_throughput >= 5.0:
        assessment_lines.append("✓ MODERATE consensus throughput")
    else:
        assessment_lines.append("⚠ LOW consensus throughput")
    
    return "\n".join(assessment_lines)

def _generate_performance_recommendations(metrics: PerformanceMetrics) -> str:
    """Generate performance optimization recommendations"""
    recommendations = []
    
    if metrics.efficiency_index < 0.6:
        recommendations.append("- Prioritize overall system optimization for better efficiency")
    
    if metrics.memory_efficiency_score < 0.7:
        recommendations.append("- Investigate memory usage patterns and optimize memory management")
    
    if metrics.scalability_score < 0.6:
        recommendations.append("- Review algorithms for better scalability with larger inputs")
    
    if metrics.bottleneck_components:
        recommendations.append(f"- Address identified bottlenecks: {', '.join(metrics.bottleneck_components[:3])}")
    
    if metrics.performance_degradation > 0.3:
        recommendations.append("- Investigate sources of performance degradation")
    
    if metrics.consensus_throughput < 5.0:
        recommendations.append("- Optimize consensus algorithms for better throughput")
    
    if metrics.total_processing_time > 30.0:
        recommendations.append("- Consider parallel processing or algorithm optimization for faster execution")
    
    if not recommendations:
        recommendations.append("- Continue monitoring performance metrics")
        recommendations.append("- Consider stress testing with larger datasets")
    
    return "\n".join(recommendations)