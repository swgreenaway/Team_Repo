"""
Model size compatibility metric for different deployment targets.

Evaluates model size compatibility across various hardware platforms
from resource-constrained devices to high-performance servers.
"""

from ..base import ResourceBundle, SizeScore
from ..registry import register
from ..base_metric import BaseMetric, stable_01


@register("size_score")
class SizeScoreMetric(BaseMetric):
    """
    Evaluates model size compatibility across deployment targets.
    
    This metric assesses model deployability by calculating size compatibility scores for:
    - Raspberry Pi: Resource-constrained edge devices
    - Jetson Nano: AI-focused edge computing platforms  
    - Desktop PC: Consumer hardware with moderate resources
    - AWS Server: Cloud instances with high-end resources
    
    Future Implementation Notes:
    - Fetch model size information from Hugging Face API
    - Consider both model weights and memory requirements during inference
    - Account for quantization options and their size/performance tradeoffs
    - Evaluate model architecture efficiency (parameters vs performance)
    - Factor in deployment framework overhead (ONNX, TensorRT, etc.)
    - Consider dynamic vs static memory requirements
    
    Size Compatibility Ranges (Future Implementation):
    - Raspberry Pi: < 100MB = 1.0, 100-500MB = linear decay to 0.0
    - Jetson Nano: < 1GB = 1.0, 1-4GB = linear decay to 0.0
    - Desktop PC: < 5GB = 1.0, 5-20GB = linear decay to 0.0
    - AWS Server: < 50GB = 1.0, 50-200GB = linear decay to 0.0
    
    Scoring Approach:
    Overall score = weighted average of device compatibility scores
    """
    
    name = "size_score"
    
    def _compute_score(self, resource: ResourceBundle) -> float:
        """
        Evaluate model size compatibility across deployment targets.
        
        TODO: Replace with real implementation that:
        1. Fetches model size from Hugging Face API (model files, weights)
        2. Estimates runtime memory requirements based on architecture
        3. Calculates compatibility scores for each target device
        4. Considers available quantization options and their impact
        5. Weights device scores based on deployment importance
        6. Accounts for model format optimizations (ONNX, TensorRT)
        
        Args:
            resource: Bundle containing model URL to analyze
            
        Returns:
            Score from 0-1 where 1 = compatible with all deployment targets
        """
        # PLACEHOLDER: Real implementation will fetch actual model size
        # and compute device-specific compatibility scores
        base_score = stable_01(resource.model_url, "size_score")
        
        # Simulate device-specific scoring (future: replace with actual calculations)
        device_scores: SizeScore = {
            "raspberry_pi": base_score * 0.3,  # Typically lowest compatibility
            "jetson_nano": base_score * 0.6,   # Moderate compatibility  
            "desktop_pc": base_score * 0.8,    # Good compatibility
            "aws_server": base_score * 0.95    # Highest compatibility
        }
        
        # Weighted average across devices (equal weighting for now)
        return sum(device_scores.values()) / len(device_scores)
    
    def _get_computation_notes(self, resource: ResourceBundle) -> str:
        return (
            f"Placeholder size analysis for {resource.model_url}. "
            "Real implementation will fetch model size from HuggingFace API "
            "and compute device-specific compatibility scores."
        )
    
    def _compute_device_scores(self, model_size_mb: float) -> SizeScore:
        """
        Compute device-specific compatibility scores based on model size.
        
        This is a helper method for future implementation that will calculate
        compatibility scores for each target device based on actual model size.
        
        Args:
            model_size_mb: Model size in megabytes
            
        Returns:
            SizeScore with compatibility scores for each device type
        """
        # Future implementation will use these thresholds
        thresholds = {
            "raspberry_pi": {"optimal": 100, "max": 500},
            "jetson_nano": {"optimal": 1000, "max": 4000}, 
            "desktop_pc": {"optimal": 5000, "max": 20000},
            "aws_server": {"optimal": 50000, "max": 200000}
        }
        
        scores: SizeScore = {}
        for device, limits in thresholds.items():
            if model_size_mb <= limits["optimal"]:
                scores[device] = 1.0
            elif model_size_mb <= limits["max"]:
                # Linear decay from optimal to max threshold
                ratio = (model_size_mb - limits["optimal"]) / (limits["max"] - limits["optimal"])
                scores[device] = max(0.0, 1.0 - ratio)
            else:
                scores[device] = 0.0
                
        return scores