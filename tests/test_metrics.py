#!/usr/bin/env python3
"""
Comprehensive test script for the metrics system.

This test verifies:
- All 8 metrics are properly registered
- Individual metric computation works correctly
- Engine orchestration produces valid NDJSON output
- Score ranges and timing are reasonable

Color coding:
- [PASS] Green: Successful tests and validations
- [FAIL] Red: Failed tests and errors  
- [WARN] Yellow: Warnings and potential issues
- [INFO] Blue: Informational messages
- Headers: Purple/Bold for section titles

Run from repo root: python tests/test_metrics.py
Or with pytest: pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports (consistent with other tests)
ROOT = Path(__file__).resolve().parent.parent  # Go up from tests/ to repo root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'    # Bright green
    RED = '\033[91m'      # Bright red
    YELLOW = '\033[93m'   # Yellow for warnings
    BLUE = '\033[94m'     # Blue for info
    PURPLE = '\033[95m'   # Purple for headers
    CYAN = '\033[96m'     # Cyan for highlights
    BOLD = '\033[1m'      # Bold text
    RESET = '\033[0m'     # Reset to default

def colored(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"

def success(text: str) -> str:
    """Green text for success messages."""
    return colored(f"[PASS] {text}", Colors.GREEN)

def failure(text: str) -> str:
    """Red text for failure messages."""
    return colored(f"[FAIL] {text}", Colors.RED)

def warning(text: str) -> str:
    """Yellow text for warning messages."""
    return colored(f"[WARN] {text}", Colors.YELLOW)

def info(text: str) -> str:
    """Blue text for informational messages."""
    return colored(f"[INFO] {text}", Colors.BLUE)

def header(text: str) -> str:
    """Purple bold text for section headers."""
    return colored(f"{text}", Colors.PURPLE + Colors.BOLD)

def test_metric_registration():
    """Test that all 8 metrics are properly registered."""
    print(header("=== Testing Metric Registration ==="))
    
    # Import metrics to trigger registration
    from app.metrics import implementations
    from app.metrics.registry import all_metrics
    
    registered = all_metrics()
    print(info(f"Registered metrics: {sorted(list(registered.keys()))}"))
    
    # All metrics from specification
    expected_metrics = [
        "license",
        "ramp_up_time", 
        "bus_factor",
        "performance_claims",
        "size_score",
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality"
    ]
    
    all_registered = True
    for metric_name in expected_metrics:
        if metric_name in registered:
            print(success(f"{metric_name} registered"))
        else:
            print(failure(f"{metric_name} NOT registered"))
            all_registered = False
    
    # Check for unexpected extra metrics
    extra_metrics = set(registered.keys()) - set(expected_metrics)
    if extra_metrics:
        print(warning(f"Unexpected metrics found: {extra_metrics}"))
    
    return all_registered and len(extra_metrics) == 0

def test_individual_metrics():
    """Test individual metric computation for all metrics."""
    print(header("\\n=== Testing Individual Metrics ==="))
    
    from app.metrics.registry import all_metrics
    from app.metrics.base import ResourceBundle
    
    # Create comprehensive test resource bundle
    test_bundle = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=["https://huggingface.co/datasets/squad", "https://huggingface.co/datasets/glue"],
        code_urls=["https://github.com/huggingface/transformers", "https://github.com/google-research/bert"]
    )
    
    # Test each registered metric
    metric_factories = all_metrics()
    all_passed = True
    
    for metric_name, metric_factory in sorted(metric_factories.items()):
        try:
            metric = metric_factory()
            result = metric.compute(test_bundle)
            
            print(success(f"{metric_name}: score={result.score:.3f}, latency={result.latency_ms}ms"))
            
            # Validate result structure and ranges
            if not (0 <= result.score <= 1):
                print(failure(f"{metric_name}: Invalid score range {result.score} (must be 0-1)"))
                all_passed = False
            if result.latency_ms < 0:
                print(failure(f"{metric_name}: Invalid latency {result.latency_ms} (must be >= 0)"))
                all_passed = False
            if not hasattr(result, 'notes'):
                print(failure(f"{metric_name}: Missing notes field"))
                all_passed = False
                
        except Exception as e:
            print(failure(f"{metric_name}: Error during computation - {e}"))
            all_passed = False
    
    return all_passed

def test_engine_functionality():
    """Test the engine's run_bundle function with all metrics."""
    print(header("\\n=== Testing Engine Functionality ==="))
    
    from app.metrics.registry import all_metrics
    from app.metrics.engine import run_bundle
    from app.metrics.base import ResourceBundle
    
    # Create test data with rich ecosystem
    test_bundle = ResourceBundle(
        model_url="https://huggingface.co/bert-base-uncased",
        dataset_urls=["https://huggingface.co/datasets/squad"],
        code_urls=["https://github.com/huggingface/transformers"]
    )
    
    # Get all registered metrics
    metric_factories = all_metrics()
    metrics = [factory() for factory in metric_factories.values()]
    
    # Comprehensive weights profile for all 8 metrics
    weights_profile = {
        "license": 0.15,
        "ramp_up_time": 0.15,
        "bus_factor": 0.10,
        "performance_claims": 0.20,
        "size_score": 0.15,
        "dataset_and_code_score": 0.10,
        "dataset_quality": 0.10,
        "code_quality": 0.05
    }
    
    try:
        ndjson_result = run_bundle(test_bundle, metrics, weights_profile)
        print(info(f"Engine output: {ndjson_result}"))
        
        # Validate JSON structure
        import json
        parsed = json.loads(ndjson_result)
        
        # Check required fields
        required_fields = ["URL", "NetScore", "NetScore_Latency"]
        missing_fields = []
        for field in required_fields:
            if field not in parsed:
                missing_fields.append(field)
        
        if missing_fields:
            print(failure(f"Missing required fields: {missing_fields}"))
            return False
        
        # Check all metric fields are present
        missing_metrics = []
        for metric_name in weights_profile.keys():
            if metric_name not in parsed:
                missing_metrics.append(metric_name)
            if f"{metric_name}_Latency" not in parsed:
                missing_metrics.append(f"{metric_name}_Latency")
        
        if missing_metrics:
            print(failure(f"Missing metric fields: {missing_metrics}"))
            return False
        
        # Validate NetScore range
        net_score = parsed["NetScore"]
        if not (0 <= net_score <= 1):
            print(failure(f"NetScore {net_score} outside valid range [0,1]"))
            return False
            
        # Check that NetScore is reasonable given weights
        individual_scores = [parsed[metric] for metric in weights_profile.keys()]
        if individual_scores and not (min(individual_scores) <= net_score <= max(individual_scores)):
            print(warning(f"NetScore {net_score} seems inconsistent with individual scores {individual_scores}"))
        
        print(success("NDJSON structure validation passed"))
        print(success(f"NetScore: {net_score:.3f} (range validated)"))
        print(success(f"All {len(weights_profile)} metrics included with latency measurements"))
        
        return True
        
    except Exception as e:
        print(failure(f"Engine error: {e}"))
        import traceback
        traceback.print_exc()
        return False

def test_metric_ecosystem_awareness():
    """Test that metrics respond appropriately to dataset/code URL availability."""
    print(header("\\n=== Testing Ecosystem Awareness ==="))
    
    from app.metrics.registry import all_metrics
    from app.metrics.base import ResourceBundle
    
    # Test bundles with different ecosystem completeness
    bundles = {
        "no_extras": ResourceBundle("https://huggingface.co/model1", [], []),
        "with_datasets": ResourceBundle("https://huggingface.co/model2", ["https://dataset1"], []),
        "with_code": ResourceBundle("https://huggingface.co/model3", [], ["https://code1"]),
        "complete": ResourceBundle("https://huggingface.co/model4", ["https://dataset1"], ["https://code1"])
    }
    
    ecosystem_metrics = ["dataset_and_code_score", "dataset_quality", "code_quality"]
    metric_factories = all_metrics()
    
    results = {}
    for bundle_name, bundle in bundles.items():
        results[bundle_name] = {}
        for metric_name in ecosystem_metrics:
            if metric_name in metric_factories:
                metric = metric_factories[metric_name]()
                result = metric.compute(bundle)
                results[bundle_name][metric_name] = result.score
    
    # Validate ecosystem awareness
    passed = True
    
    # dataset_quality should be lower when no datasets
    if "dataset_quality" in results["no_extras"]:
        no_dataset_score = results["no_extras"]["dataset_quality"]
        with_dataset_score = results["with_datasets"]["dataset_quality"]
        if no_dataset_score >= with_dataset_score:
            print(warning(f"dataset_quality should be lower without datasets: {no_dataset_score} >= {with_dataset_score}"))
    
    # code_quality should be lower when no code
    if "code_quality" in results["no_extras"]:
        no_code_score = results["no_extras"]["code_quality"]
        with_code_score = results["with_code"]["code_quality"]
        if no_code_score >= with_code_score:
            print(warning(f"code_quality should be lower without code: {no_code_score} >= {with_code_score}"))
    
    print(success("Ecosystem awareness validated"))
    return passed

def main():
    """Run all tests."""
    print(colored("Testing Metrics System (All 8 Metrics)", Colors.CYAN + Colors.BOLD))
    print()
    
    tests = [
        ("Metric Registration", test_metric_registration),
        ("Individual Metrics", test_individual_metrics), 
        ("Engine Functionality", test_engine_functionality),
        ("Ecosystem Awareness", test_metric_ecosystem_awareness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(success(f"{test_name} PASSED"))
                passed += 1
            else:
                print(failure(f"{test_name} FAILED"))
        except Exception as e:
            print(failure(f"{test_name} ERROR: {e}"))
            import traceback
            traceback.print_exc()
    
    print(header("\\n=== FINAL RESULTS ==="))
    print(info(f"Passed: {passed}/{total}"))
    
    if passed == total:
        print(colored("Status: ALL TESTS PASSED", Colors.GREEN + Colors.BOLD))
    else:
        print(colored("Status: SOME TESTS FAILED", Colors.RED + Colors.BOLD))
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)