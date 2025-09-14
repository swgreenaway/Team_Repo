"""
Metrics orchestration and scoring engine.

This module handles the parallel execution of metrics and assembly of
results into NDJSON format. It provides the core orchestration logic
that coordinates multiple metrics and produces the final NetScore.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter_ns
import json
from typing import Dict, Any, List
from .base import MetricResult, ResourceBundle, Metric

def compute_net_score(results: Dict[str, MetricResult], weights_profile: Dict[str, float]) -> float:
    """
    Compute weighted NetScore from individual metric results.
    
    Implements the NetScore formula from the specification:
    NetScore = (Σ(weight_i × score_i)) / (Σ(weight_i))
    
    Args:
        results: Dictionary mapping metric names to their MetricResult objects
        weights_profile: Dictionary mapping metric names to their weights
        
    Returns:
        Weighted average score in range [0, 1], or 0.0 if no valid weights
        
    Future Enhancement Notes:
    - Add weight validation (ensure weights sum to reasonable value)
    - Support for metric dependencies (some metrics may depend on others)
    - Configurable score normalization strategies
    - Logging for weight mismatches or missing metrics
    """
    if not results or not weights_profile:
        return 0.0
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    # Apply weights to each metric that has both a result and a weight
    for metric_name, result in results.items():
        weight = weights_profile.get(metric_name, 0.0)
        if weight > 0:  # Only include metrics with positive weights
            weighted_sum += result.score * weight
            weight_sum += weight
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0

def assemble_ndjson_row(bundle: ResourceBundle, results: Dict[str, MetricResult], 
                        net_score: float, latency_ms: int) -> str:
    """
    Assemble NDJSON output row for a resource bundle.
    
    Creates the final output format expected by the auto-grader, including
    the model URL, NetScore, timing information, and all individual metric scores.
    
    Args:
        bundle: Resource bundle that was evaluated
        results: Dictionary of metric results
        net_score: Computed weighted average score
        latency_ms: Total processing time for this bundle
        
    Returns:
        JSON string in NDJSON format for output
        
    Future Enhancement Notes:
    - Add JSON schema validation
    - Support for additional metadata fields
    - Configurable precision for score rounding
    - Include metric notes in debug mode
    """
    # Start with required fields per specification
    ndjson_row = {
        "URL": bundle.model_url,
        "NetScore": round(net_score, 3),         # 3 decimal precision per spec
        "NetScore_Latency": latency_ms
    }
    
    # Add individual metric scores and their latencies
    # Sorted by metric name for consistent output ordering
    for metric_name in sorted(results.keys()):
        result = results[metric_name]
        ndjson_row[metric_name] = round(result.score, 3)
        ndjson_row[f"{metric_name}_Latency"] = result.latency_ms
    
    # TODO: In debug mode, could also include:
    # - ndjson_row["_debug_notes"] = {name: result.notes for name, result in results.items()}
    # - ndjson_row["_dataset_urls"] = bundle.dataset_urls
    # - ndjson_row["_code_urls"] = bundle.code_urls
    
    return json.dumps(ndjson_row)

def run_bundle(bundle: ResourceBundle, metrics: List[Metric], 
               weights_profile: Dict[str, float]) -> str:
    """
    Execute all metrics for a resource bundle and return NDJSON result.
    
    This is the main orchestration function that:
    1. Runs all metrics in parallel using ThreadPoolExecutor
    2. Collects results and computes NetScore
    3. Assembles final NDJSON output string
    
    Args:
        bundle: Resource bundle to evaluate (model + datasets + code)
        metrics: List of metric instances to execute
        weights_profile: Weight configuration for NetScore calculation
        
    Returns:
        NDJSON string ready for output
        
    Future Enhancement Notes:
    - Add timeout handling for slow metrics
    - Implement retry logic for transient failures  
    - Add progress reporting for long-running evaluations
    - Support for metric dependency ordering
    - Resource usage monitoring (memory, CPU)
    """
    t0 = perf_counter_ns()
    futures = {}
    
    # Determine optimal thread pool size
    # Cap at 8 to avoid overwhelming APIs, but respect CPU count
    max_workers = min(8, (os.cpu_count() or 4), len(metrics))
    
    # Execute all metrics in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Submit all metric computations
        for metric in metrics:
            future = ex.submit(metric.compute, bundle)
            futures[future] = metric.name
        
        # Collect results as they complete
        results = {}
        for fut in as_completed(futures):
            metric_name = futures[fut]
            try:
                metric_result = fut.result()
                results[metric_name] = metric_result
            except Exception as e:
                # TODO: Add proper logging here
                # For now, create a failed result to avoid breaking the pipeline
                results[metric_name] = MetricResult(
                    score=0.0,
                    latency_ms=0,
                    notes=f"Metric computation failed: {str(e)}"
                )
    
    # Compute final weighted score
    net_score = compute_net_score(results, weights_profile)
    
    # Total processing time for this bundle
    total_latency_ms = int((perf_counter_ns() - t0) / 1_000_000)
    
    return assemble_ndjson_row(bundle, results, net_score, total_latency_ms)
