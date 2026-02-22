"""
Matcher implementations for biometric verification.

This module provides base classes and concrete implementations for various
feature matching methods used in biometric verification.
"""

from .base import BaseMatcher, MatcherConfig
from .sift import SIFTMatcher
from .orb import ORBMatcher
from .superglue import SuperGlueMatcher
from .registry import create_matcher, get_matcher_class

__all__ = [
	"BaseMatcher",
	"MatcherConfig",
	"SIFTMatcher",
	"ORBMatcher",
	"SuperGlueMatcher",
	"create_matcher",
	"get_matcher_class",
]
