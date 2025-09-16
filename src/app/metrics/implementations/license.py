"""
License evaluation metric.

Evaluates license clarity and LGPLv2.1 compatibility for model repositories.
Analyzes license files, model cards, and repository metadata to determine
commercial use compatibility and legal clarity.
"""

from ..base import ResourceBundle
from ..registry import register
from ..base_metric import BaseMetric, stable_01


@register("license")
class LicenseMetric(BaseMetric):
    """
    Evaluates license clarity and LGPLv2.1 compatibility.
    
    This metric will assess:
    - Presence and clarity of license files (LICENSE, COPYING, etc.)
    - Compatibility with LGPLv2.1 requirements
    - License consistency across repository files
    - Commercial use permissions
    - Hugging Face model card license declarations
    
    Future Implementation Notes:
    - Parse LICENSE files from model repositories via GitHub API
    - Check Hugging Face model card license fields via HF API
    - Cross-reference with SPDX license database for compatibility
    - Evaluate license compatibility matrix against LGPLv2.1
    - Check for license conflicts between model/dataset/code components
    
    Scoring Rubric (Future):
    - 1.0: Clear LGPL-compatible license with proper attribution
    - 0.8: Compatible license but unclear/missing attribution requirements  
    - 0.6: Compatible license with minor compliance concerns
    - 0.4: Restrictive license that limits commercial use
    - 0.2: Unclear/conflicting license information
    - 0.0: No license information or incompatible license
    """
    
    name = "license"
    
    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate license quality and compatibility.
        
        TODO: Replace with real implementation that:
        1. Fetches LICENSE files from model repository (GitHub API)
        2. Parses Hugging Face model card license fields (HF API)
        3. Identifies license type using SPDX database lookups
        4. Checks LGPLv2.1 compatibility via predefined compatibility matrix
        5. Validates license consistency across all repository files
        6. Evaluates attribution and commercial use requirements
        
        Args:
            resource: Bundle containing model and related URLs to analyze
            
        Returns:
            Score from 0-1 where 1 = perfect LGPLv2.1 compatibility
        """
        # PLACEHOLDER: Real implementation will parse actual license files
        # and evaluate against LGPL compatibility requirements
        return stable_01(resource.model_url, "license")
    
    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        return (
            f"Placeholder license evaluation for {resource.model_url}. "
            "Real implementation will parse LICENSE files and check LGPL compatibility "
            "via GitHub API and SPDX database."
        )