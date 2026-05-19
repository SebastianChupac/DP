# registry.py - Matcher registry and factory for BioVerify framework.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

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
    """Return matcher class by name (case-insensitive).
    
    Supports versioned matcher names by extracting the base name before "-".
    Examples:
    - "sift" -> SIFTMatcher
    - "sift-v1" -> SIFTMatcher
    - "sift-optimized" -> SIFTMatcher
    """
    # Extract base name before first "-" if present
    base_name = name.strip().lower().split('-')[0]
    
    if base_name not in MATCHER_REGISTRY:
        available = ", ".join(sorted(MATCHER_REGISTRY.keys()))
        raise ValueError(f"Unknown matcher '{name}' (base: '{base_name}'). Available: {available}")
    return MATCHER_REGISTRY[base_name]


def create_matcher(name: str, config_dict: dict) -> BaseMatcher:
    """Create a matcher instance from name and config dict."""
    matcher_cls = get_matcher_class(name)
    config = MatcherConfig.from_dict(config_dict or {})
    return matcher_cls(config)
