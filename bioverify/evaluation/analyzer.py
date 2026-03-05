"""
Result analyzer for threshold sweeping and matcher comparison.

Loads experiment results and provides analysis capabilities:
- Threshold sweeping (ROC, DET, EER)
- Matcher comparison at fixed thresholds
- Score distribution analysis
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .metrics import (
    compute_roc_curve,
    compute_det_curve,
    find_eer,
    compute_threshold_metrics
)


class ThresholdAnalyzer:
    """Analyze a single matcher's threshold effects."""
    
    def __init__(self, results_dict: Dict, matcher_name: str):
        """
        Initialize analyzer for a specific matcher.
        
        Args:
            results_dict: Dictionary with 'results' key containing list of result dicts
            matcher_name: Name of the matcher to analyze (e.g., 'LoFTR_indoor')
        """
        self.matcher_name = matcher_name
        self.results = results_dict.get('results', [])
        
        # Filter by matcher
        self.matcher_results = [
            r for r in self.results 
            if r['method_name'] == matcher_name
        ]
        
        if not self.matcher_results:
            raise ValueError(f"No results found for matcher: {matcher_name}")
        
        # Extract scores and ground truth
        self.scores = np.array([r['verification_confidence'] for r in self.matcher_results])
        self.ground_truth = np.array([
            1 if r['ground_truth'] else 0 
            for r in self.matcher_results
        ])
        
        # Statistics
        self.num_pairs = len(self.matcher_results)
        self.num_genuine = np.sum(self.ground_truth)
        self.num_impostor = self.num_pairs - self.num_genuine
        
        # Score statistics
        self.score_min = self.scores.min()
        self.score_max = self.scores.max()
        self.score_mean = self.scores.mean()
        self.score_std = self.scores.std()
    
    def get_score_statistics(self) -> Dict:
        """Get score distribution statistics."""
        genuine_scores = self.scores[self.ground_truth == 1]
        impostor_scores = self.scores[self.ground_truth == 0]
        
        return {
            'overall': {
                'min': float(self.score_min),
                'max': float(self.score_max),
                'mean': float(self.score_mean),
                'std': float(self.score_std),
            },
            'genuine': {
                'min': float(genuine_scores.min()),
                'max': float(genuine_scores.max()),
                'mean': float(genuine_scores.mean()),
                'std': float(genuine_scores.std()),
                'count': int(self.num_genuine),
            },
            'impostor': {
                'min': float(impostor_scores.min()),
                'max': float(impostor_scores.max()),
                'mean': float(impostor_scores.mean()),
                'std': float(impostor_scores.std()),
                'count': int(self.num_impostor),
            },
        }
    
    def sweep_threshold(self, num_points: int = 1000) -> Dict:
        """
        Sweep across thresholds and compute metrics.
        
        Args:
            num_points: Number of threshold points to evaluate
            
        Returns:
            Analysis results with ROC, DET, EER, and operating points
        """
        # Compute curves
        roc = compute_roc_curve(self.scores, self.ground_truth, num_points)
        det = compute_det_curve(self.scores, self.ground_truth, num_points)
        eer_result = find_eer(self.scores, self.ground_truth, num_points)
        
        # Get current operating point (from experiment config)
        current_threshold = self._extract_threshold_from_config()
        current_metrics = compute_threshold_metrics(self.scores, self.ground_truth, current_threshold)
        
        return {
            'matcher': self.matcher_name,
            'num_pairs': self.num_pairs,
            'num_genuine': self.num_genuine,
            'num_impostor': self.num_impostor,
            'score_statistics': self.get_score_statistics(),
            'roc': roc,
            'det': det,
            'eer': eer_result,
            'current_operating_point': current_metrics,
        }
    
    def _extract_threshold_from_config(self) -> float:
        """
        Extract decision threshold from matcher configuration.
        
        Returns:
            Configured decision threshold, or median score if not found
        """
        if not self.matcher_results:
            return float(self.score_mean)
        
        # Get matcher params from first result (should be same for all)
        matcher_params = self.matcher_results[0].get('matcher_params', {})
        
        # Common threshold parameter names across different matchers
        threshold_keys = [
            'ratio_threshold',      
            '_ratio_threshold', 
        ]
        
        for key in threshold_keys:
            if key in matcher_params:
                return float(matcher_params[key])
        
        # Fallback: Print warning and return median score
        print(f"⚠️  Warning: No threshold parameter found in matcher_params for {self.matcher_name}. Using median score as threshold.")
        return float(self.score_mean)
    
    
    def get_operating_point(self, threshold: float) -> Dict:
        """Get metrics at a specific operating point."""
        return compute_threshold_metrics(self.scores, self.ground_truth, threshold)


class MatcherComparator:
    """Compare multiple matchers at fixed threshold(s)."""
    
    def __init__(self, results_dict: Dict, matcher_names: Optional[List[str]] = None):
        """
        Initialize comparator for multiple matchers.
        
        Args:
            results_dict: Dictionary with 'results' key
            matcher_names: List of matcher names to compare. If None, use all matchers.
        """
        self.results = results_dict.get('results', [])
        
        # Get unique matchers if not specified
        if matcher_names is None:
            matcher_names = sorted(set(r['method_name'] for r in self.results))
        
        self.matcher_names = matcher_names
        self.analyzers = {
            name: ThresholdAnalyzer(results_dict, name)
            for name in matcher_names
        }
    
    def compare_at_threshold(self, threshold: float) -> Dict[str, Dict]:
        """
        Compare all matchers at a specific threshold.
        
        WARNING: This comparison only makes sense for:
        - Comparing the SAME matcher with different parameters (similar score distributions)
        - Academic exercises
        
        For practical matcher comparison, use compare_at_eer() or compare_at_far() instead,
        as different matchers have different score distributions.
        
        Args:
            threshold: Decision threshold to compare at
            
        Returns:
            Dictionary mapping matcher names to their metrics at this threshold
        """
        comparison = {}
        for matcher_name, analyzer in self.analyzers.items():
            comparison[matcher_name] = analyzer.get_operating_point(threshold)
        return comparison
    
    def compare_at_eer(self) -> Dict[str, Dict]:
        """
        Compare all matchers at their respective EER points.
        
        This is the industry-standard way to compare matchers, as each matcher
        operates at its own optimal (balanced) threshold.
        
        Returns:
            Dictionary with:
            - 'matchers': Dict mapping matcher names to their EER metrics
            - 'ranking': List of matchers sorted by EER (best first)
        """
        eer_results = self.get_all_eer()
        
        comparison = {}
        for matcher_name, eer_data in eer_results.items():
            comparison[matcher_name] = {
                'threshold': eer_data['threshold'],
                'eer': eer_data['eer'],
                'tar': 1 - eer_data['frr'],  # TAR at EER point
                'far': eer_data['far'],
                'frr': eer_data['frr'],
                'trr': 1 - eer_data['far'],  # TRR at EER point
            }
        
        # Rank by EER (lower is better)
        ranking = sorted(comparison.items(), key=lambda x: x[1]['eer'])
        
        return {
            'matchers': comparison,
            'ranking': [{'matcher': name, **metrics} for name, metrics in ranking],
        }
    
    def compare_at_far(self, target_far: float) -> Dict[str, Dict]:
        """
        Compare all matchers at a fixed False Acceptance Rate.
        
        This is common in security-critical applications where you want to ensure
        a maximum acceptable impostor acceptance rate (e.g., FAR=0.1% or 0.01%).
        
        For each matcher, finds the threshold that achieves the target FAR,
        then reports TAR/FRR at that point.
        
        Args:
            target_far: Target False Acceptance Rate (e.g., 0.001 for 0.1%)
            
        Returns:
            Dictionary with:
            - 'target_far': The requested FAR
            - 'matchers': Dict mapping matcher names to their metrics at that FAR
            - 'ranking': List of matchers sorted by TAR (best first)
        """
        comparison = {}
        
        for matcher_name, analyzer in self.analyzers.items():
            # Find threshold that gives target FAR
            threshold, metrics = self._find_threshold_for_far(analyzer, target_far)
            comparison[matcher_name] = {
                'threshold': threshold,
                'tar': metrics['tar'],
                'far': metrics['far'],
                'frr': metrics['frr'],
                'trr': metrics['trr'],
            }
        
        # Rank by TAR (higher is better at same FAR)
        ranking = sorted(comparison.items(), key=lambda x: x[1]['tar'], reverse=True)
        
        return {
            'target_far': target_far,
            'matchers': comparison,
            'ranking': [{'matcher': name, **metrics} for name, metrics in ranking],
        }
    
    def compare_at_frr(self, target_frr: float) -> Dict[str, Dict]:
        """
        Compare all matchers at a fixed False Rejection Rate.
        
        This is common in user-convenience applications where you want to minimize
        genuine user rejections (e.g., FRR=1% or 5%).
        
        For each matcher, finds the threshold that achieves the target FRR,
        then reports TAR/FAR at that point.
        
        Args:
            target_frr: Target False Rejection Rate (e.g., 0.01 for 1%)
            
        Returns:
            Dictionary with:
            - 'target_frr': The requested FRR
            - 'matchers': Dict mapping matcher names to their metrics at that FRR
            - 'ranking': List of matchers sorted by TRR (best first)
        """
        comparison = {}
        
        for matcher_name, analyzer in self.analyzers.items():
            # Find threshold that gives target FRR
            threshold, metrics = self._find_threshold_for_frr(analyzer, target_frr)
            comparison[matcher_name] = {
                'threshold': threshold,
                'tar': metrics['tar'],
                'far': metrics['far'],
                'frr': metrics['frr'],
                'trr': metrics['trr'],
            }
        
        # Rank by TRR (higher is better at same FRR)
        ranking = sorted(comparison.items(), key=lambda x: x[1]['trr'], reverse=True)
        
        return {
            'target_frr': target_frr,
            'matchers': comparison,
            'ranking': [{'matcher': name, **metrics} for name, metrics in ranking],
        }
    
    def _find_threshold_for_far(self, analyzer: ThresholdAnalyzer, target_far: float) -> tuple:
        """Find threshold that achieves target FAR for a matcher."""
        # Sweep thresholds
        thresholds = np.linspace(analyzer.score_min, analyzer.score_max, 10000)
        
        best_threshold = analyzer.score_mean
        best_metrics = None
        min_diff = float('inf')
        
        for threshold in thresholds:
            metrics = analyzer.get_operating_point(threshold)
            diff = abs(metrics['far'] - target_far)
            
            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics
    
    def _find_threshold_for_frr(self, analyzer: ThresholdAnalyzer, target_frr: float) -> tuple:
        """Find threshold that achieves target FRR for a matcher."""
        # Sweep thresholds
        thresholds = np.linspace(analyzer.score_min, analyzer.score_max, 10000)
        
        best_threshold = analyzer.score_mean
        best_metrics = None
        min_diff = float('inf')
        
        for threshold in thresholds:
            metrics = analyzer.get_operating_point(threshold)
            diff = abs(metrics['frr'] - target_frr)
            
            if diff < min_diff:
                min_diff = diff
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics
    
    def get_all_eer(self) -> Dict[str, Dict]:
        """Get EER for all matchers."""
        eer_results = {}
        for matcher_name, analyzer in self.analyzers.items():
            result = analyzer.sweep_threshold(num_points=10000)
            eer_results[matcher_name] = result['eer']
        return eer_results
    
    def get_all_thresholds(self) -> Dict[str, Dict]:
        """Get full threshold sweep for all matchers."""
        all_sweeps = {}
        for matcher_name, analyzer in self.analyzers.items():
            all_sweeps[matcher_name] = analyzer.sweep_threshold()
        return all_sweeps
    
    def get_summary(self) -> Dict:
        """Get summary comparison across all matchers."""
        summary = {
            'num_matchers': len(self.matcher_names),
            'matchers': {},
        }
        
        for matcher_name, analyzer in self.analyzers.items():
            summary['matchers'][matcher_name] = {
                'num_pairs': analyzer.num_pairs,
                'num_genuine': analyzer.num_genuine,
                'num_impostor': analyzer.num_impostor,
                'score_statistics': analyzer.get_score_statistics(),
            }
        
        # Add EER comparison
        eer_results = self.get_all_eer()
        summary['eer_comparison'] = {
            name: eer_results[name] for name in self.matcher_names
        }
        
        return summary
