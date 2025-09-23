import re
from .base import ResourceBundle
from .registry import register
from .base_metric import BaseMetric
from ..Url_Parser.Url_Parser import *

@register("code_quality")
class CodeQualityMetric(BaseMetric):
    """
    Evaluates code quality based on repository documentation and structure.
    """

    name = "code_quality"

    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate code quality through simple repository checks.
        """
        r = requests.get(HF_API_MODEL.format(repo_id=resource.model_id), timeout=10)
        r.raise_for_status()
        data = r.json()
        readme = data.get("cardData", {}).get("README", "").lower()

        # Base score
        score = 0.2

        # Check for GitHub repository links in README
        github_pattern = r"github\.com/[^\s)]+"
        has_github = bool(re.search(github_pattern, readme))
        if has_github:
            score += 0.4

        # Check for code examples or usage
        code_indicators = ["```", "python", "import", "from ", "def ", "class "]
        has_code = any(indicator in readme for indicator in code_indicators)
        if has_code:
            score += 0.3

        # Check for installation/setup instructions
        setup_keywords = ["install", "pip", "setup", "requirements"]
        has_setup = any(keyword in readme for keyword in setup_keywords)
        if has_setup:
            score += 0.1

        return min(score, 1.0)

    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        return f"Code quality analysis for {resource.model_url} based on README content and repository references."