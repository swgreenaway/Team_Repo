import re
from .base import ResourceBundle
from .registry import register
from .base_metric import BaseMetric
from ..Url_Parser.Url_Parser import *

@register("dataset_quality")
class DatasetQualityMetric(BaseMetric):
    """
    Evaluates dataset quality based on documentation and references.
    """

    name = "dataset_quality"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate dataset quality through simple checks.
        """
        r = requests.get(HF_API_MODEL.format(repo_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()
        readme = data.get("cardData", {}).get("README", "").lower()

        # Base score
        score = 0.2

        # Check for dataset mentions in README
        dataset_keywords = ["dataset", "training data", "evaluation data", "benchmark"]
        has_dataset_mention = any(keyword in readme for keyword in dataset_keywords)
        if has_dataset_mention:
            score += 0.3

        # Check for usage documentation
        usage_keywords = ["usage", "example", "how to use", "getting started"]
        has_usage = any(keyword in readme for keyword in usage_keywords)
        if has_usage:
            score += 0.3

        # Check for license information
        license_info = data.get("cardData", {}).get("license")
        if license_info:
            score += 0.2

        return min(score, 1.0)

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        return f"Dataset quality analysis for {resource.model_url} based on README content and metadata."