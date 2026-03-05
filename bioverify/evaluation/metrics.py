"""
Metrics computation for biometric verification evaluation.

Provides functions to compute TAR, FAR, FRR, TRR, ROC curves, DET curves,
and Equal Error Rate (EER) from verification results.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def _build_thresholds(scores: np.ndarray, max_points: int) -> np.ndarray:
    """Build robust threshold grid for ROC/DET/EER.

    Uses score-aware thresholds with guaranteed endpoints:
    (max + ε) (predict none) ... unique scores ... (min - ε) (predict all)
    This guarantees ROC starts at (0,0) and reaches (1,1).
    """
    unique_scores = np.unique(scores.astype(float))

    # Limit number of evaluated score thresholds when needed
    if max_points and max_points > 0 and unique_scores.size > max_points:
        idx = np.linspace(0, unique_scores.size - 1, max_points, dtype=int)
        unique_scores = unique_scores[idx]

    # Evaluate from strictest to loosest threshold
    core = unique_scores[::-1]
    max_score = float(unique_scores.max())
    min_score = float(unique_scores.min())

    high_sentinel = float(np.nextafter(max_score, np.inf))
    low_sentinel = float(np.nextafter(min_score, -np.inf))

    return np.concatenate(([high_sentinel], core, [low_sentinel]))


def compute_threshold_metrics(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute verification metrics at a specific threshold.
    
    Args:
        scores: Array of verification confidence scores
        ground_truth: Array of binary ground truth (1=genuine, 0=impostor)
        threshold: Decision threshold
        
    Returns:
        Dictionary with TAR, FAR, FRR, TRR
    """
    # Apply threshold
    predictions = scores >= threshold
    
    # Separate genuine and impostor
    genuine_mask = ground_truth == 1
    impostor_mask = ground_truth == 0
    
    # True positives, false positives, false negatives, true negatives
    tp = np.sum(predictions & genuine_mask)  # Genuine accepted
    fn = np.sum(~predictions & genuine_mask)  # Genuine rejected
    fp = np.sum(predictions & impostor_mask)  # Impostor accepted
    tn = np.sum(~predictions & impostor_mask)  # Impostor rejected
    
    # Metrics
    tar = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Acceptance Rate
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Rejection Rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance Rate
    trr = tn / (fp + tn) if (fp + tn) > 0 else 0.0  # True Rejection Rate
    
    return {
        'threshold': float(threshold),
        'tar': float(tar),
        'frr': float(frr),
        'far': float(far),
        'trr': float(trr),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
    }


def compute_roc_curve(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    num_thresholds: int = 1000
) -> Dict[str, any]:
    """
    Compute ROC curve points (TAR vs FAR).
    
    Args:
        scores: Array of verification confidence scores
        ground_truth: Array of binary ground truth (1=genuine, 0=impostor)
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        Dictionary with thresholds, tar, far, and auc
    """
    thresholds = _build_thresholds(scores, num_thresholds)
    
    tar_list = []
    far_list = []
    
    genuine_mask = ground_truth == 1
    impostor_mask = ground_truth == 0
    
    num_genuine = np.sum(genuine_mask)
    num_impostor = np.sum(impostor_mask)
    
    for threshold in thresholds:
        predictions = scores >= threshold
        
        # TAR: % of genuine accepted
        tar = np.sum(predictions & genuine_mask) / max(1, num_genuine)
        # FAR: % of impostor accepted
        far = np.sum(predictions & impostor_mask) / max(1, num_impostor)
        
        tar_list.append(tar)
        far_list.append(far)
    
    tar_list = np.array(tar_list)
    far_list = np.array(far_list)
    
    # Compute AUC using trapezoidal rule (sort by FAR first)
    sort_idx = np.argsort(far_list)
    far_sorted = far_list[sort_idx]
    tar_sorted = tar_list[sort_idx]
    auc = np.trapz(tar_sorted, far_sorted)
    
    return {
        'thresholds': thresholds.tolist(),
        'tar': tar_list.tolist(),
        'far': far_list.tolist(),
        'auc': float(auc),
    }


def compute_det_curve(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    num_thresholds: int = 1000
) -> Dict[str, any]:
    """
    Compute DET curve points (FRR vs FAR, log scale).
    
    Detection Error Tradeoff curve shows the relationship between
    false rejection rate and false acceptance rate.
    
    Args:
        scores: Array of verification confidence scores
        ground_truth: Array of binary ground truth (1=genuine, 0=impostor)
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        Dictionary with thresholds, frr, far
    """
    thresholds = _build_thresholds(scores, num_thresholds)
    
    frr_list = []
    far_list = []
    
    genuine_mask = ground_truth == 1
    impostor_mask = ground_truth == 0
    
    num_genuine = np.sum(genuine_mask)
    num_impostor = np.sum(impostor_mask)
    
    for threshold in thresholds:
        predictions = scores >= threshold
        
        # FRR: % of genuine rejected
        frr = np.sum(~predictions & genuine_mask) / max(1, num_genuine)
        # FAR: % of impostor accepted
        far = np.sum(predictions & impostor_mask) / max(1, num_impostor)
        
        frr_list.append(frr)
        far_list.append(far)
    
    return {
        'thresholds': thresholds.tolist(),
        'frr': frr_list,
        'far': far_list,
    }


def find_eer(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    num_thresholds: int = 10000
) -> Dict[str, float]:
    """
    Find Equal Error Rate (EER) where FAR = FRR.
    
    EER is a common operating point for biometric systems where
    false acceptance and false rejection rates are equal.
    
    Args:
        scores: Array of verification confidence scores
        ground_truth: Array of binary ground truth (1=genuine, 0=impostor)
        num_thresholds: Number of thresholds to search
        
    Returns:
        Dictionary with eer value, threshold, far, and frr at EER
    """
    thresholds = _build_thresholds(scores, num_thresholds)
    
    genuine_mask = ground_truth == 1
    impostor_mask = ground_truth == 0
    
    num_genuine = np.sum(genuine_mask)
    num_impostor = np.sum(impostor_mask)
    
    min_diff = float('inf')
    eer_threshold = 0.0
    eer_far = 0.0
    eer_frr = 0.0
    
    for threshold in thresholds:
        predictions = scores >= threshold
        
        frr = np.sum(~predictions & genuine_mask) / max(1, num_genuine)
        far = np.sum(predictions & impostor_mask) / max(1, num_impostor)
        
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer_threshold = threshold
            eer_far = far
            eer_frr = frr
    
    eer = (eer_far + eer_frr) / 2.0
    
    return {
        'eer': float(eer),
        'threshold': float(eer_threshold),
        'far': float(eer_far),
        'frr': float(eer_frr),
    }
