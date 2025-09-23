"""
Dataset quality and ecosystem metrics.

Evaluates the quality and reliability of datasets associated with models,
including both training datasets and evaluation benchmarks.
"""

import re
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import from project root for URL parsing
import sys
_root_path = str(Path(__file__).parent.parent.parent.parent)
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

try:
    from Url_Parser import parse_huggingface_url
except ImportError:
    try:
        sys.path.insert(0, _root_path)
        from Url_Parser import parse_huggingface_url
    except ImportError:
        def parse_huggingface_url(url: str) -> Dict[str, Any]:
            """Fallback URL parser."""
            return {"type": "unknown", "url": url}

from ..base import ResourceBundle
from ..registry import register
from ..base_metric import BaseMetric, stable_01
from ...integrations.huggingface_client import hf_client


@register("dataset_and_code_score")
class DatasetAndCodeScoreMetric(BaseMetric):
    """
    Evaluates presence and quality of linked dataset and code documentation.
    
    This composite metric assesses the completeness of the model ecosystem by analyzing:
    - Availability and accessibility of training/evaluation datasets
    - Quality and completeness of dataset documentation
    - Presence of associated code repositories
    - Code quality and maintainability indicators
    - Integration between model, data, and code components
    
    Future Implementation Notes:
    - Check Hugging Face model cards for dataset references
    - Validate dataset accessibility and download success
    - Analyze dataset documentation completeness and metadata quality
    - Evaluate associated code repositories for completeness
    - Assess integration quality between model/dataset/code components
    """
    
    name = "dataset_and_code_score"

    def _fetch_hf_model_info(self, model_url: str) -> Optional[Dict[str, Any]]:
        """Fetch model information from Hugging Face API."""
        try:
            parsed = parse_huggingface_url(model_url)
            if not parsed or not parsed.get('path_parts'):
                return None

            # Extract model_id from path_parts
            model_id = '/'.join(parsed['path_parts'])
            if not model_id:
                return None

            # Get model info using the wrapper
            model_info = hf_client.get_model_info(model_id, include_files=True)
            if not model_info or 'error' in model_info:
                return None

            # Get model card data
            card_data = hf_client.get_model_card_data(model_id)

            return {
                'model_info': model_info['model_info'],
                'card_data': card_data,
                'model_id': model_id
            }

        except Exception:
            return None

    def _analyze_dataset_documentation(self, hf_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset references and documentation quality."""
        analysis = {
            'datasets_mentioned': [],
            'dataset_links': [],
            'documentation_quality': 0.0,
            'training_data_described': False,
            'evaluation_data_described': False
        }

        if not hf_info:
            return analysis

        # Check model card for dataset mentions
        card_data = hf_info.get('card_data')
        if card_data and card_data.get('datasets'):
            analysis['datasets_mentioned'] = card_data['datasets']
            analysis['dataset_links'].extend(card_data['datasets'])

        # Check README content for dataset documentation
        readme_content = hf_client.get_model_readme(hf_info['model_id'])

        if readme_content:
            content = readme_content.lower()

            # Look for dataset-related keywords
            dataset_keywords = ['dataset', 'training data', 'evaluation data', 'benchmark', 'corpus']
            training_keywords = ['trained on', 'training set', 'training data', 'fine-tuned on']
            eval_keywords = ['evaluated on', 'evaluation set', 'benchmark', 'test set']

            # Count mentions and assess documentation quality
            dataset_mentions = sum(1 for keyword in dataset_keywords if keyword in content)
            analysis['documentation_quality'] = min(1.0, dataset_mentions / 3.0)

            analysis['training_data_described'] = any(keyword in content for keyword in training_keywords)
            analysis['evaluation_data_described'] = any(keyword in content for keyword in eval_keywords)

            # Extract dataset URLs/references
            dataset_pattern = r'https?://[^\s]+(?:dataset|huggingface\.co/datasets)[^\s]*'
            dataset_urls = re.findall(dataset_pattern, content)
            analysis['dataset_links'].extend(dataset_urls)

        return analysis

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate presence and quality of associated datasets and code.

        Analyzes:
        1. Dataset references in model cards and documentation
        2. Quality of dataset documentation
        3. Presence of associated code repositories
        4. Training and evaluation data descriptions

        Args:
            resource: Bundle containing model, dataset, and code URLs

        Returns:
            Score from 0-1 where 1 = complete ecosystem with high-quality datasets and code
        """
        # Fetch Hugging Face model information
        hf_info = self._fetch_hf_model_info(resource.model_url)

        if not hf_info:
            # Fallback to basic URL analysis if HF API fails
            dataset_factor = 0.8 if resource.dataset_urls else 0.2
            code_factor = 0.8 if resource.code_urls else 0.2
            return stable_01(resource.model_url, "dataset_code") * dataset_factor * code_factor

        # Analyze dataset documentation
        dataset_analysis = self._analyze_dataset_documentation(hf_info)

        # Score components (each 0-1)
        scores = {
            'dataset_urls_provided': 1.0 if resource.dataset_urls else 0.0,
            'code_urls_provided': 1.0 if resource.code_urls else 0.0,
            'datasets_mentioned_in_card': 1.0 if dataset_analysis['datasets_mentioned'] else 0.0,
            'dataset_links_found': 1.0 if dataset_analysis['dataset_links'] else 0.0,
            'documentation_quality': dataset_analysis['documentation_quality'],
            'training_data_described': 1.0 if dataset_analysis['training_data_described'] else 0.0,
            'evaluation_data_described': 1.0 if dataset_analysis['evaluation_data_described'] else 0.0
        }

        # Weighted combination
        weights = {
            'dataset_urls_provided': 0.15,
            'code_urls_provided': 0.10,
            'datasets_mentioned_in_card': 0.20,
            'dataset_links_found': 0.15,
            'documentation_quality': 0.20,
            'training_data_described': 0.15,
            'evaluation_data_described': 0.05
        }

        final_score = sum(scores[component] * weights[component] for component in scores)
        return max(0.0, min(1.0, final_score))
    
    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        """Generate detailed notes about the dataset and code analysis."""
        hf_info = self._fetch_hf_model_info(resource.model_url)

        if not hf_info:
            # Try to get more specific error info
            parsed = parse_huggingface_url(resource.model_url)
            if parsed and parsed.get('path_parts'):
                model_id = '/'.join(parsed['path_parts'])
                model_info = hf_client.get_model_info(model_id)
                if model_info and 'error' in model_info:
                    return f"HuggingFace API error for {resource.model_url}: {model_info['error']}"
            return f"Could not fetch model information from Hugging Face API for {resource.model_url}"

        dataset_analysis = self._analyze_dataset_documentation(hf_info)

        notes = []

        # Basic availability
        dataset_status = f"{len(resource.dataset_urls)} datasets" if resource.dataset_urls else "no datasets"
        code_status = f"{len(resource.code_urls)} code repos" if resource.code_urls else "no code repos"
        notes.append(f"Bundle contains {dataset_status} and {code_status}")

        # Model card analysis
        if dataset_analysis['datasets_mentioned']:
            notes.append(f"Model card mentions {len(dataset_analysis['datasets_mentioned'])} datasets")
        else:
            notes.append("No datasets mentioned in model card")

        # Documentation quality
        if dataset_analysis['documentation_quality'] > 0.7:
            notes.append("High-quality dataset documentation")
        elif dataset_analysis['documentation_quality'] > 0.3:
            notes.append("Moderate dataset documentation quality")
        else:
            notes.append("Limited dataset documentation")

        # Training/evaluation data
        data_descriptions = []
        if dataset_analysis['training_data_described']:
            data_descriptions.append("training data")
        if dataset_analysis['evaluation_data_described']:
            data_descriptions.append("evaluation data")

        if data_descriptions:
            notes.append(f"Describes {' and '.join(data_descriptions)}")
        else:
            notes.append("No clear training/evaluation data descriptions")

        # Dataset links
        if dataset_analysis['dataset_links']:
            notes.append(f"Found {len(dataset_analysis['dataset_links'])} dataset links")

        return "; ".join(notes)