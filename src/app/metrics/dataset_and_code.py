import re
import sys
from pathlib import Path
from typing import Optional
from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

# Import HuggingFace client
try:
    from app.integrations.huggingface_client import hf_client
except ImportError:
    hf_client = None

# Import LLM metadata extractor
try:
    from app.metrics.llm_metadata_extractor import metadata_extractor
except ImportError:
    metadata_extractor = None

@register("dataset_and_code_score")
class DatasetAndCodeScoreMetric(BaseMetric):
    """
    Evaluates presence of dataset and code references in model documentation.
    Enhanced with LLM-based metadata extraction for bonus scoring.
    """

    name = "dataset_and_code_score"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Generous approach: Start at 0.5 and provide substantial bonuses for good indicators.
        LLM provides major boost if analysis looked good.1
        """
        # Start with base score of 0.5
        score = 0.5

        # Get README content for analysis
        readme_content = ""
        if hf_client and resource.model_id:
            readme_content = hf_client.get_model_readme(resource.model_id) or ""

        # Direct keyword matching - generous bonus for basic indicators
        if self._has_basic_indicators(readme_content):
            score += 0.2

        # LLM "Vibe Check" - can provide major boost based on overall quality impression
        llm_vibe_boost = self._get_llm_vibe_boost(readme_content, resource.model_id)
        score += llm_vibe_boost  # Up to 0.3+ boost for excellent vibe

        # URL bonuses - generous rewards for linked resources
        if resource.dataset_urls:
            score += 0.15
        if resource.code_urls:
            score += 0.15

        return min(score, 1.0)

    def _has_basic_indicators(self, readme_content: str) -> bool:
        """
        Check for basic dataset and code indicators.
        Simple boolean check for presence of key terms.

        Args:
            readme_content: README content to analyze

        Returns:
            True if basic indicators are found
        """
        if not readme_content:
            return False

        readme_lower = readme_content.lower()

        # Check for dataset mentions
        dataset_keywords = ["dataset", "training data", "evaluation data", "benchmark", "corpus"]
        has_dataset = any(keyword in readme_lower for keyword in dataset_keywords)

        # Check for code/repository links
        code_patterns = ["github.com", "gitlab.com", "code", "repository", "repo"]
        has_code = any(pattern in readme_lower for pattern in code_patterns)

        return has_dataset or has_code


    def _get_llm_vibe_boost(self, readme_content: str, model_id: str) -> float:
        """
        LLM check- provides major boost for good overall documentation impression.
        Not constrained by small weights, can give substantial score increases.

        Args:
            readme_content: README content to analyze
            model_id: Model identifier

        Returns:
            Boost from 0-0.3+ based on LLM's overall positive impression
        """
        if not metadata_extractor or not readme_content:
            return 0.1  # Small default boost

        try:
            extracted_metadata = metadata_extractor.extract_metadata(model_id)

            boost = 0.0

            # High confidence = excellent documentation
            if extracted_metadata.confidence_score > 0.8:
                boost += 0.3  # Major boost for excellent
            elif extracted_metadata.confidence_score > 0.6:
                boost += 0.2  # Good boost for positive
            elif extracted_metadata.confidence_score > 0.3:
                boost += 0.1  # Modest boost for decent

            # Stack additional bonuses for rich content
            if extracted_metadata.training_details or extracted_metadata.training_datasets:
                boost += 0.1  # Bonus for training info

            if extracted_metadata.use_cases:
                boost += 0.1  # Bonus for usage examples

            return min(boost, 0.3)  # Cap the boost but allow stacking

        except Exception:
            # If LLM fails, still give modest boost
            return 0.1


    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        notes = f"Generous scoring for {resource.model_url} with LLM vibe-based boosts."

        # Calculate what bonuses were applied
        bonus_details = []

        if hf_client and resource.model_id:
            readme_content = hf_client.get_model_readme(resource.model_id) or ""
            if readme_content:
                notes += f" README length: {len(readme_content)} characters."

                # Check what basic indicators were found
                if self._has_basic_indicators(readme_content):
                    bonus_details.append("basic indicators (+0.2)")

        # Check URL bonuses
        if resource.dataset_urls:
            bonus_details.append("dataset URLs (+0.15)")
        if resource.code_urls:
            bonus_details.append("code URLs (+0.15)")

        # Add LLM vibe boost info
        if metadata_extractor and resource.model_id:
            try:
                extracted_metadata = metadata_extractor.extract_metadata(resource.model_id)
                if extracted_metadata.confidence_score > 0:
                    vibe_boost = self._get_llm_vibe_boost(hf_client.get_model_readme(resource.model_id) or "", resource.model_id)
                    bonus_details.append(f"LLM vibe boost (+{vibe_boost:.2f})")
                    notes += f" LLM confidence: {extracted_metadata.confidence_score:.2f}."
            except Exception:
                bonus_details.append("LLM bonus points (+0.1 default)")

        if bonus_details:
            notes += f" Bonuses: {', '.join(bonus_details)}."

        return notes