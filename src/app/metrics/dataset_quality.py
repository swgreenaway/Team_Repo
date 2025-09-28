import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from app.metrics.base import ResourceBundle
from app.metrics.registry import register
from app.metrics.base_metric import BaseMetric

# Import HuggingFace client with error handling
try:
    from app.integrations.huggingface_client import hf_client
except ImportError:
    hf_client = None

@register("dataset_quality")
class DatasetQualityMetric(BaseMetric):
    """
    Evaluates dataset reliability, popularity, and quality indicators.

    The simplified scoring rubric focuses on three pillars that are broadly
    available even when dataset cards are sparse:
    - Dataset presence and availability (can the dataset be fetched, is it maintained)
    - Documentation quality (README content, summaries, timestamps)
    - Popularity and community adoption (downloads, likes)

    Scoring Rubric:
    - 1.0: Excellent datasets with comprehensive documentation and clear evidence of adoption
    - 0.8: Good datasets with solid documentation and moderate popularity
    - 0.6: Acceptable datasets with basic documentation or modest usage signals
    - 0.4: Poor datasets with minimal documentation or unclear availability
    - 0.2: Very poor datasets with almost no supporting signals
    - 0.0: No datasets or completely inaccessible datasets
    """

    name = "dataset_quality"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate quality and reliability of associated datasets.

        Analyzes each dataset for:
        1. Popularity metrics (downloads, likes, community usage)
        2. Documentation quality and completeness
        3. Availability signals indicating the dataset can be used and is maintained

        Args:
            resource: Bundle with dataset URLs to analyze

        Returns:
            Score from 0-1 where 1 = high-quality, well-documented, widely-used datasets
        """
        if not resource.dataset_urls:
            return 0.0

        if not hf_client:
            return 0.1  # Minimal score if HF client unavailable

        total_score = 0.0
        analyzed_datasets = 0
        dataset_details = []

        # Analyze each dataset (limit to first 5 to avoid excessive processing)
        for dataset_url in resource.dataset_urls[:5]:
            try:
                dataset_id = self._extract_dataset_id(dataset_url)
                if not dataset_id:
                    continue

                dataset_score, details = self._analyze_single_dataset(dataset_id)
                if dataset_score is not None:
                    total_score += dataset_score
                    analyzed_datasets += 1
                    dataset_details.append(details)

            except Exception:
                continue  # Skip failed datasets

        if analyzed_datasets == 0:
            return 0.0

        # Average score across analyzed datasets
        average_score = total_score / analyzed_datasets

        # Bonus for dataset diversity (multiple quality datasets)
        diversity_bonus = min(0.1, (analyzed_datasets - 1) * 0.05)

        # Store analysis details for notes
        self._analysis_details = {
            'analyzed_count': analyzed_datasets,
            'total_count': len(resource.dataset_urls),
            'dataset_details': dataset_details,
            'average_score': average_score,
            'diversity_bonus': diversity_bonus
        }

        return min(1.0, average_score + diversity_bonus)

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        if not resource.dataset_urls:
            return f"No datasets found for {resource.model_url}. Quality assessment not possible."

        if not hf_client:
            return "HuggingFace client unavailable. Cannot analyze dataset quality."

        details = getattr(self, '_analysis_details', {})
        if not details:
            return f"Failed to analyze {len(resource.dataset_urls)} dataset(s) for {resource.model_url}."

        analyzed = details.get('analyzed_count', 0)
        total = details.get('total_count', 0)
        avg_score = details.get('average_score', 0.0)

        notes = [
            f"Analyzed {analyzed}/{total} datasets for {resource.model_url}.",
            f"Average dataset quality score: {avg_score:.2f}"
        ]

        # Add details about top datasets
        dataset_details = details.get('dataset_details', [])
        if dataset_details:
            top_datasets = sorted(dataset_details, key=lambda x: x['score'], reverse=True)[:3]
            notes.append("Top datasets:")
            for ds in top_datasets:
                notes.append(f"  - {ds['id']}: {ds['score']:.2f} (downloads: {ds['downloads']}, likes: {ds['likes']})")

        return " ".join(notes)

    def _extract_dataset_id(self, dataset_url: str) -> Optional[str]:
        """
        Extract dataset ID from HuggingFace dataset URL.

        Args:
            dataset_url: Full URL to HuggingFace dataset

        Returns:
            Dataset ID string or None if not a valid HF dataset URL
        """
        if not dataset_url:
            return None

        # Handle direct dataset IDs
        if '/' not in dataset_url or not dataset_url.startswith('http'):
            return dataset_url

        # Parse HuggingFace dataset URLs
        # https://huggingface.co/datasets/squad
        # https://huggingface.co/datasets/glue/cola
        parsed = urlparse(dataset_url)
        if 'huggingface.co' not in parsed.netloc:
            return None

        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'datasets':
            # Extract dataset name (may include organization)
            if len(path_parts) == 2:
                return path_parts[1]  # Simple case: datasets/squad
            else:
                # Complex case: datasets/org/dataset or datasets/dataset/subset
                return '/'.join(path_parts[1:3])  # Take first two parts after datasets/

        return None

    def _analyze_single_dataset(self, dataset_id: str) -> Tuple[Optional[float], Dict]:
        """
        Analyze a single dataset for quality metrics.

        Args:
            dataset_id: HuggingFace dataset identifier

        Returns:
            Tuple of (score, details_dict) or (None, {}) on failure
        """
        try:
            # Get dataset info and metadata
            info_result = hf_client.get_dataset_info(dataset_id)
            if 'error' in info_result:
                return None, {}

            card_data = hf_client.get_dataset_card_data(dataset_id)
            if 'error' in card_data:
                card_data = {}

            readme_content = hf_client.get_dataset_readme(dataset_id)

            # Extract basic metrics
            dataset_info = info_result.get('dataset_info')
            downloads = card_data.get('downloads', 0) or 0
            likes = card_data.get('likes', 0) or 0

            # Calculate quality score components with a simplified rubric
            popularity_score = self._calculate_popularity_score(downloads, likes)
            documentation_score = self._calculate_documentation_score(card_data, readme_content)
            presence_score = self._calculate_presence_score(dataset_info, card_data, readme_content)

            # Weighted overall score focused on presence, documentation, and popularity
            overall_score = (
                presence_score * 0.3 +
                documentation_score * 0.35 +
                popularity_score * 0.35
            )

            details = {
                'id': dataset_id,
                'score': overall_score,
                'downloads': downloads,
                'likes': likes,
                'popularity_score': popularity_score,
                'documentation_score': documentation_score,
                'presence_score': presence_score
            }

            return overall_score, details

        except Exception:
            return None, {}

    def _calculate_popularity_score(self, downloads: int, likes: int) -> float:
        """
        Calculate popularity score based on downloads and likes.

        Args:
            downloads: Number of downloads
            likes: Number of likes

        Returns:
            Score from 0-1 based on popularity metrics
        """
        # Normalize download counts (logarithmic scale)
        if downloads > 0:
            # Popular datasets: 100K+ downloads = 1.0, 10K+ = 0.8, 1K+ = 0.6
            download_score = min(1.0, (downloads / 100000) ** 0.5)
        else:
            download_score = 0.0

        # Normalize like counts
        if likes > 0:
            # Popular datasets: 1000+ likes = 1.0, 100+ = 0.8, 10+ = 0.6
            like_score = min(1.0, (likes / 1000) ** 0.5)
        else:
            like_score = 0.0

        # Weighted combination (downloads more important than likes)
        return download_score * 0.8 + like_score * 0.2

    def _calculate_documentation_score(self, card_data: Dict, readme_content: Optional[str]) -> float:
        """
        Calculate documentation quality score.

        Args:
            card_data: Dataset card metadata
            readme_content: README.md content

        Returns:
            Score from 0-1 based on documentation completeness
        """
        score = 0.0

        # Check for README content and award partial credit even if brief
        if readme_content:
            score += 0.4

            # Bonus for comprehensive README (length, sections)
            if len(readme_content) > 1000:
                score += 0.1

            # Check for key documentation sections
            readme_lower = readme_content.lower()
            key_sections = ['dataset', 'description', 'usage', 'citation', 'license']
            found_sections = sum(1 for section in key_sections if section in readme_lower)
            score += (found_sections / len(key_sections)) * 0.3
        else:
            # Minimal credit for having any descriptive metadata
            if card_data.get('card_data') or card_data.get('cardData'):
                score += 0.1

        # Structured metadata and timestamps offer supporting evidence
        if card_data.get('last_modified') or card_data.get('created_at'):
            score += 0.1
        if card_data.get('summary') or card_data.get('description'):
            score += 0.1
        if card_data.get('tags'):
            score += 0.05
        if card_data.get('license'):
            score += 0.05

        return min(1.0, score)

    def _calculate_presence_score(
        self,
        dataset_info: Optional[Dict],
        card_data: Dict,
        readme_content: Optional[str]
    ) -> float:
        """Assess whether the dataset appears usable and maintained."""

        score = 0.0

        # Credit for being able to fetch dataset info, even if sparse
        if dataset_info is not None:
            score += 0.6

        # Credit for having card data or README content
        if card_data:
            score += 0.2
        if readme_content:
            score += 0.1

        # Signals of recent activity or maintenance
        if card_data.get('last_modified') or card_data.get('updatedAt'):
            score += 0.1

        return min(1.0, score)