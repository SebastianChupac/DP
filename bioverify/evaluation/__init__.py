# __init__.py - Evaluation module for analyzing experiment results, generating curves, and comparing matchers.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

"""
Evaluation module for analyzing experiment results.

Provides tools for:
- Threshold sweeping and ROC/DET curve generation
- Matcher comparison and analysis
- Parameter sweep analysis
- Result visualization
"""

from .metrics import compute_roc_curve, compute_det_curve, find_eer, compute_threshold_metrics
from .analyzer import ThresholdAnalyzer, MatcherComparator

__all__ = [
    'compute_roc_curve',
    'compute_det_curve', 
    'find_eer',
    'compute_threshold_metrics',
    'ThresholdAnalyzer',
    'MatcherComparator',
]
