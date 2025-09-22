"""
Model size compatibility metric for different deployment targets.

Evaluates model size compatibility across various hardware platforms
from resource-constrained devices to high-performance servers.
"""

import os
import re
import requests
from typing import Optional
from pathlib import Path

from huggingface_hub import HfApi

# Import from project root
import sys
_root_path = str(Path(__file__).parent.parent.parent.parent)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

try:
    from Url_Parser import parse_huggingface_url
except ImportError:
    # Fallback for when running tests that add src/ to path
    sys.path.insert(0, _root_path)
    from Url_Parser import parse_huggingface_url

from ..base import ResourceBundle, SizeScore
from ..registry import register
from ..base_metric import BaseMetric


# Weight file patterns for identifying model weight files
_WEIGHT_PATTERNS = [
    r"\.safetensors$", r"\.bin$", r"\.gguf$",
    r"pytorch_model-.*\.bin$", r"model-\d+-of-\d+\.safetensors$",
    r"pytorch_model\.bin$", r"model\.safetensors$",
    r"model\.safetensors\.index\.json$",      # ST sharded index
    r"pytorch_model\.bin\.index\.json$",      # BIN sharded index
]


def _is_weight_file(path: str) -> bool:
    """Check if a file path matches weight file patterns."""
    return any(re.search(p, path) for p in _WEIGHT_PATTERNS)


def _auth_token() -> Optional[str]:
    """Get HF token from env, if present."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def _head_size(url: str, token: Optional[str]) -> Optional[int]:
    """Get file size via HEAD request to raw URL (Content-Length)."""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.head(url, headers=headers, timeout=20, allow_redirects=True)
        cl = r.headers.get("Content-Length")
        return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


def _huggingface_raw_url(model_id: str, revision: str, filename: str) -> str:
    """Construct Hugging Face raw file URL."""
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"


@register("size_score")
class SizeScoreMetric(BaseMetric):
    """
    Evaluates model size compatibility across deployment targets.

    This metric assesses model deployability by calculating size compatibility scores for:
    - Raspberry Pi: Resource-constrained edge devices
    - Jetson Nano: AI-focused edge computing platforms
    - Desktop PC: Consumer hardware with moderate resources
    - AWS Server: Cloud instances with high-end resources
    """

    name = "size_score"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate model size compatibility across deployment targets.
        Fetches actual model size from Hugging Face Hub and computes device scores.
        """
        model_size_mb = self._fetch_model_size(resource.model_url)
        if model_size_mb is None:
            return 0.0

        device_scores = self._compute_device_scores(model_size_mb)
        return sum(device_scores.values()) / len(device_scores)

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        model_size_mb = self._fetch_model_size(resource.model_url)
        if model_size_mb is not None:
            return f"Model size: {model_size_mb:.1f}MB. Device compatibility computed from Hub file metadata (with HEAD fallback)."
        return f"Could not fetch model size from {resource.model_url}. Score set to 0.0 (unevaluable)."

    def _fetch_model_size(self, model_url: str) -> Optional[float]:
        """
        Fetch total *weights* size (in MB) from Hugging Face using huggingface_hub.

        Strategy:
          - HfApi.model_info(repo_id, files_metadata=True) to enumerate files with sizes.
          - Sum only weight files (by extension/pattern).
          - If any weight file lacks 'size', HEAD the raw URL to get Content-Length.
        """
        try:
            parts = parse_huggingface_url(model_url)
            path_parts = parts.get("path_parts", [])
            if len(path_parts) < 2:
                return None

            model_id = "/".join(path_parts[:2])
            revision = parts.get("revision") or "main"

            token = _auth_token()
            api = HfApi(token=token)
            info = api.model_info(model_id, files_metadata=True)

            total_bytes = 0
            # info.siblings is a list of RepoSibling objects (attrs: rfilename, size, etc.)
            for sib in info.siblings or []:
                fname = getattr(sib, "rfilename", None)
                if not fname or not _is_weight_file(fname):
                    continue

                size = getattr(sib, "size", None)
                if size is None:
                    # Fallback HEAD to determine size
                    raw = _huggingface_raw_url(model_id, revision, fname)
                    size = _head_size(raw, token)

                if size is not None:
                    total_bytes += int(size)

            return (total_bytes / (1024 * 1024)) if total_bytes > 0 else None

        except Exception:
            return None

    def _compute_device_scores(self, model_size_mb: float) -> SizeScore:
        """
        Compute device-specific compatibility scores based on model size (MB).
        """
        thresholds = {
            "raspberry_pi": {"optimal": 100, "max": 500},
            "jetson_nano": {"optimal": 1000, "max": 4000},
            "desktop_pc": {"optimal": 5000, "max": 20000},
            "aws_server": {"optimal": 50000, "max": 200000},
        }

        scores: SizeScore = {}
        for device, limits in thresholds.items():
            if model_size_mb <= limits["optimal"]:
                scores[device] = 1.0
            elif model_size_mb <= limits["max"]:
                ratio = (model_size_mb - limits["optimal"]) / (limits["max"] - limits["optimal"])
                scores[device] = max(0.0, 1.0 - ratio)
            else:
                scores[device] = 0.0

        return scores
