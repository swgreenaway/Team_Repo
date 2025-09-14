import argparse
from urllib.parse import urlparse

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
