import time
import sys
import Url_Parser
import Installer
import Tester
from pathlib import Path
from typing import Dict, Any, List
import json

def install_dependencies() -> int:
    return Installer.run()
    

def run_tests(pytest_args=None) -> int:
    if pytest_args is None:
        try:
            i = sys.argv.index("test")
            pytest_args = sys.argv[i+1:]   # everything after 'test'
        except ValueError:
            pytest_args = []
    return Tester.run(pytest_args=pytest_args)


def process_urls(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Check if line contains comma-separated URLs (grouped evaluation)
                if ',' in line:
                    process_url_group(line)
                else:
                    # Single URL - traditional processing
                    process_url(line)
    return 0

def process_url_group(url_group: str) -> int:
    """
    Run metrics evaluation for a group of related URLs (model, datasets, code).

    Args:
        url_group: Comma-separated URLs representing related resources

    Returns:
        0 on success, non-zero on error
    """
    try:
        # Parse comma-separated URLs
        urls = [url.strip() for url in url_group.split(',') if url.strip()]
        if not urls:
            print("Error: No valid URLs found in group", file=sys.stderr)
            return 1

        # Import URL categorization
        import Url_Parser

        # Categorize URLs
        model_urls = []
        dataset_urls = []
        code_urls = []

        for url in urls:
            category = Url_Parser.categorize_url(url)
            if category == "MODEL":
                model_urls.append(url)
            elif category == "DATASET":
                dataset_urls.append(url)
            elif category == "CODE":
                code_urls.append(url)
            else:
                print(f"Warning: Unknown URL type for {url}", file=sys.stderr)

        # Determine primary model URL
        if not model_urls:
            print("Error: No model URL found in group. At least one HuggingFace model URL is required.", file=sys.stderr)
            return 1

        # Use first model URL as primary (could be enhanced to handle multiple models)
        primary_model = model_urls[0]
        if len(model_urls) > 1:
            print(f"Warning: Multiple model URLs found. Using {primary_model} as primary.", file=sys.stderr)

        # Run grouped evaluation
        result = run_metrics(
            model_url=primary_model,
            dataset_urls=dataset_urls,
            code_urls=code_urls
        )
        print(result)
        return 0

    except Exception as e:
        print(f"Error evaluating URL group {url_group}: {e}", file=sys.stderr)
        return 1

def process_url(url: str) -> int:
    """
    Run metrics evaluation for a single URL.

    Args:
        url: Single model URL to evaluate

    Returns:
        0 on success, non-zero on error
    """
    try:
        # For now, treat the single URL as a model URL
        # In the future, this could be enhanced to detect URL type
        result = run_metrics(model_url=url)
        print(result)
        return 0
    except Exception as e:
        print(f"Error evaluating URL {url}: {e}", file=sys.stderr)
        return 1


def run_metrics(model_url: str, dataset_urls: List[str] = None, code_urls: List[str] = None, weights: Dict[str, float] = None, use_cache: bool = True) -> str:
    """
    Run the metrics system on a given model with optional dataset and code URLs.

    Args:
        model_url: The primary model URL (required)
        dataset_urls: Optional list of dataset URLs
        code_urls: Optional list of code URLs
        weights: Optional custom weights profile for metrics
        use_cache: Whether to use database caching (default: True)

    Returns:
        NDJSON string with metric results
    """
    # Add src to path for imports
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    try:
        # Import metrics system and database
        from app.metrics.registry import all_metrics
        from app.metrics.engine import run_bundle
        from app.metrics.base import ResourceBundle
        from app.database import get_database

        # Set defaults
        if dataset_urls is None:
            dataset_urls = []
        if code_urls is None:
            code_urls = []

        # Check cache first if caching is enabled
        if use_cache:
            db = get_database()
            cached_result = db.get_cached_result(model_url, dataset_urls, code_urls)
            if cached_result:
                return cached_result

        # Default weights profile if not provided
        if weights is None:
            weights = {
                "license": 0.15,
                "ramp_up_time": 0.15,
                "bus_factor": 0.10,
                "performance_claims": 0.20,
                "size_score": 0.15,
                "dataset_and_code_score": 0.10,
                "dataset_quality": 0.10,
                "code_quality": 0.05
            }

        # Extract model_id from URL for metrics that need it
        model_id = model_url.split("/")[-1] if model_url else ""

        # Create resource bundle
        resource_bundle = ResourceBundle(
            model_url=model_url,
            dataset_urls=dataset_urls,
            code_urls=code_urls,
            model_id=model_id
        )

        # Get all available metrics
        metric_factories = all_metrics()
        if not metric_factories:
            raise RuntimeError("No metrics registered")

        metrics = [factory() for factory in metric_factories.values()]

        # Run the metrics engine
        result = run_bundle(resource_bundle, metrics, weights)

        # Cache the result if caching is enabled
        if use_cache:
            db = get_database()
            db.cache_result(model_url, result, dataset_urls, code_urls)

        return result

    except Exception as e:
        # Return error in NDJSON format
        error_result = {
            "URL": model_url,
            "Error": str(e),
            "NetScore": 0.0,
            "NetScore_Latency": 0
        }
        return json.dumps(error_result)

