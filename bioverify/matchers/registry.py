"""
Matcher registry and factory helpers.
"""

from typing import Dict, Type

from .base import BaseMatcher, MatcherConfig
from .sift import SIFTMatcher
from .orb import ORBMatcher
from .superglue import SuperGlueMatcher
from .loftr import LoFTRMatcher
from .aspanformer import ASpanFormerMatcher
from .sgmnet import SGMNetMatcher
from .deepdetect import DeepDetectMatcher


MATCHER_REGISTRY: Dict[str, Type[BaseMatcher]] = {
    "sift": SIFTMatcher,
    "orb": ORBMatcher,
    "superglue": SuperGlueMatcher,
    "loftr": LoFTRMatcher,
    "aspanformer": ASpanFormerMatcher,
    "sgmnet": SGMNetMatcher,
    "deepdetect": DeepDetectMatcher,
}


def get_matcher_class(name: str) -> Type[BaseMatcher]:
    """Return matcher class by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in MATCHER_REGISTRY:
        available = ", ".join(sorted(MATCHER_REGISTRY.keys()))
        raise ValueError(f"Unknown matcher '{name}'. Available: {available}")
    return MATCHER_REGISTRY[key]


def create_matcher(name: str, config_dict: dict) -> BaseMatcher:
    """Create a matcher instance from name and config dict."""
    matcher_cls = get_matcher_class(name)
    config = MatcherConfig.from_dict(config_dict or {})
    return matcher_cls(config)
