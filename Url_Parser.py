import argparse
from urllib.parse import urlparse
from pathlib import Path
import json
import time
from typing import Any, Dict

def parse_huggingface_url(url: str) -> dict:
    
    """
    Parses a Hugging Face URL into components.
    Example: https://huggingface.co/datasets/user/dataset_name
    Returns a dictionary with useful parts.
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    return {
        "scheme": parsed.scheme,
        "domain": parsed.netloc,
        "path_parts": parts
    }


def categorize_url(url: str) -> str:
    """Return category for given URL (MODEL, DATASET, CODE)."""
    if "huggingface.co/datasets" in url:
        return "DATASET"
    elif "huggingface.co" in url:
        return "MODEL"
    elif "github.com" in url:
        return "CODE"
    else:
        return "UNKNOWN"


def process_urls(file_path: Path) -> int:
    """Process URLs from file and output NDJSON results for models."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue

            category = categorize_url(url)
            name = url.split("/")[-1]

            if category == "MODEL":
                result = score_model(name)
                print(json.dumps(result))



def score_model(name: str) -> Dict[str, Any]:
    """Generate placeholder scores for a model."""
    start = time.time()

    # Fake scoring values
    scores = {
        "ramp_up_time": 0.7,
        "bus_factor": 0.8,
        "performance_claims": 0.5,
        "license": 1.0,
        "dataset_and_code_score": 0.6,
        "dataset_quality": 0.7,
        "code_quality": 0.8,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics for a Hugging Face model/dataset."
    )
    parser.add_argument("url", help="Hugging Face API URL")
    args = parser.parse_args()

    url_info = parse_huggingface_url(args.url)
    print(f"Parsed URL info: {url_info}")


if __name__ == "__main__":
    main()
