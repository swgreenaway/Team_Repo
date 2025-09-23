import re
from typing import Optional
from pathlib import Path

from .base import ResourceBundle, SizeScore
from .registry import register
from .base_metric import BaseMetric

# Import HuggingFace client with error handling
try:
    from app.integrations.huggingface_client import hf_client
    _HF_CLIENT_AVAILABLE = True
except ImportError:
    _HF_CLIENT_AVAILABLE = False
    hf_client = None

# Weight file patterns for identifying model weight files
_WEIGHT_PATTERNS = [
    r"\.safetensors$", r"\.bin$", r"\.gguf$",
    r"pytorch_model-.*\.bin$", r"model-\d+-of-\d+\.safetensors$",
    r"pytorch_model\.bin$", r"model\.safetensors$",
]

def _is_weight_file(path: str) -> bool:
    """Check if a file path matches weight file patterns."""
    return any(re.search(p, path) for p in _WEIGHT_PATTERNS)

@register("size_score")
class SizeScoreMetric(BaseMetric):
    """
    Evaluates model size compatibility by downloading and measuring actual model size.
    """

    name = "size_score"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Download model locally and calculate device compatibility score.
        """
        if not _HF_CLIENT_AVAILABLE:
            return 0.0

        model_size_mb = self._fetch_model_size(resource.model_url)
        if model_size_mb is None:
            return 0.0

        # Simple scoring: smaller models get higher scores
        if model_size_mb < 100:
            return 1.0
        elif model_size_mb < 1000:
            return 0.8
        elif model_size_mb < 5000:
            return 0.6
        elif model_size_mb < 20000:
            return 0.4
        else:
            return 0.2

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        if not _HF_CLIENT_AVAILABLE:
            return f"HuggingFace client not available for {resource.model_url}."

        model_size_mb = self._fetch_model_size(resource.model_url)
        if model_size_mb is not None:
            return f"Downloaded model size: {model_size_mb:.1f}MB"
        return f"Could not download model from {resource.model_url}"

    def _fetch_model_size(self, model_url: str) -> Optional[float]:
        """
        Download model locally and calculate weight file size.
        """
        if not _HF_CLIENT_AVAILABLE:
            return None

        try:
            # Extract model ID from URL (simplified)
            if "huggingface.co" not in model_url:
                return None

            url_parts = model_url.rstrip('/').split('/')
            model_id = '/'.join(url_parts[-2:]) if len(url_parts) >= 2 else url_parts[-1]

            # Download model
            cache_dir = f"models/size_metric_cache/{model_id.replace('/', '_')}"
            download_result = hf_client.download_model(
                model_id=model_id,
                cache_dir=cache_dir,
                ignore_patterns=["*.h5", "*.md", "*.txt"]
            )

            if not download_result['success']:
                return None

            # Calculate weight file sizes
            local_path = Path(download_result['local_path'])
            total_bytes = 0

            if local_path.exists():
                for file_path in local_path.rglob('*'):
                    if file_path.is_file() and _is_weight_file(file_path.name):
                        total_bytes += file_path.stat().st_size

            # Cleanup
            try:
                import shutil
                if local_path.exists():
                    shutil.rmtree(local_path.parent)
            except Exception:
                pass

            return (total_bytes / (1024 * 1024)) if total_bytes > 0 else None

        except Exception:
            return None
