from typing import Dict, Callable
from .base import Metric

_REGISTRY: Dict[str, Callable[[], Metric]] = {}

def register(name: str):
    def _wrap(factory: Callable[[], Metric]):
        _REGISTRY[name] = factory
        return factory
    return _wrap

def all_metrics() -> Dict[str, Callable[[], Metric]]:
    return dict(_REGISTRY)
