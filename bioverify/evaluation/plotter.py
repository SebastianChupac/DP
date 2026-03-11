"""
Visualization utilities for evaluation results.

Provides plotting functions for ROC curves, DET curves, and other metrics.
Requires matplotlib; gracefully degrades if not available.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def plot_roc_curve(
    analysis_results: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot ROC curve (TAR vs FAR).
    
    Args:
        analysis_results: Results from sweep_threshold()
        output_path: Path to save plot (if provided)
        show: Whether to display plot (requires display environment)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping ROC curve plot.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    roc = analysis_results['roc']
    matcher = analysis_results['matcher']
    
    ax.plot(roc['far'], roc['tar'], linewidth=2, label=f'ROC (AUC={roc["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random classifier')
    
    # Mark EER point
    eer = analysis_results['eer']
    ax.plot(eer['far'], 1 - eer['frr'], 'ro', markersize=8, label=f'EER={eer["eer"]:.4f}')
    
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('True Acceptance Rate (TAR)')
    ax.set_title(f'ROC Curve - {matcher}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ ROC curve saved to {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_det_curve(
    analysis_results: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot DET curve (FRR vs FAR, log scale).
    
    Args:
        analysis_results: Results from sweep_threshold()
        output_path: Path to save plot (if provided)
        show: Whether to display plot (requires display environment)
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.special import erfinv
    except ImportError:
        print("Warning: matplotlib/scipy not available. Skipping DET curve plot.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    det = analysis_results['det']
    matcher = analysis_results['matcher']
    
    # Convert to normal inverse scale (pmnorm scale)
    frr = [max(1e-6, min(1-1e-6, f)) for f in det['frr']]
    far = [max(1e-6, min(1-1e-6, f)) for f in det['far']]
    
    ax.semilogy(far, frr, linewidth=2, label=matcher)
    
    # Mark EER point
    eer = analysis_results['eer']
    ax.semilogy(eer['far'], eer['frr'], 'ro', markersize=8, label=f'EER={eer["eer"]:.4f}')
    
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('False Rejection Rate (FRR)')
    ax.set_title(f'DET Curve - {matcher}')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ DET curve saved to {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_roc_comparison(
    analysis_results: Dict[str, Dict],
    output_path: Optional[Path] = None,
    show: bool = False
):
    """
    Plot ROC curves for multiple matchers.
    
    Args:
        analysis_results: Dictionary mapping matcher names to their analysis results
        output_path: Path to save plot
        show: Whether to display plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping comparison plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for matcher_name, result in analysis_results.items():
        roc = result['roc']
        ax.plot(roc['far'], roc['tar'], linewidth=2, label=f'{matcher_name} (AUC={roc["auc"]:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random classifier')
    
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('True Acceptance Rate (TAR)')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {output_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def plot_score_distribution(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    matcher_name: str,
    current_threshold: Optional[float] = None,
    eer_threshold: Optional[float] = None,
    output_path: Optional[Path] = None,
    show: bool = False,
):
    """Plot genuine and impostor score distributions for a matcher.

    Args:
        scores: Array of verification confidence scores
        ground_truth: Binary labels (1=genuine, 0=impostor)
        matcher_name: Name of matcher for plot title
        current_threshold: Currently configured decision threshold
        eer_threshold: Threshold at Equal Error Rate (minimum FAR/FRR gap)
        output_path: Path to save plot
        show: Whether to display plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available. Skipping score distribution plot.")
        return

    scores = np.asarray(scores, dtype=float)
    ground_truth = np.asarray(ground_truth)

    genuine_scores = scores[ground_truth == 1]
    impostor_scores = scores[ground_truth == 0]

    if genuine_scores.size == 0 or impostor_scores.size == 0:
        print(f"Warning: insufficient score classes for distribution plot: {matcher_name}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    bins = np.linspace(score_min, score_max, 40)

    ax.hist(
        impostor_scores,
        bins=bins,
        alpha=0.55,
        density=True,
        color="#d62728",
        label=f"Impostor (n={impostor_scores.size})",
    )
    ax.hist(
        genuine_scores,
        bins=bins,
        alpha=0.55,
        density=True,
        color="#1f77b4",
        label=f"Genuine (n={genuine_scores.size})",
    )

    ax.axvline(np.mean(impostor_scores), color="#d62728", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(genuine_scores), color="#1f77b4", linestyle="--", linewidth=1.5)

    if current_threshold is not None:
        ax.axvline(
            float(current_threshold),
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.8,
            label=f"Current threshold={float(current_threshold):.4f}",
        )

    if eer_threshold is not None:
        ax.axvline(
            float(eer_threshold),
            color="#9467bd",
            linestyle=":",
            linewidth=2.0,
            label=f"EER threshold={float(eer_threshold):.4f}",
        )

    ax.set_title(f"Score Distribution - {matcher_name}")
    ax.set_xlabel("Verification Confidence Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Score distribution plot saved to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def save_analysis_report(analysis_results: Dict, output_path: Path):
    """
    Save analysis results as formatted JSON.
    
    Args:
        analysis_results: Results from analyzer
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, cls=NumpyEncoder)
    print(f"✓ Analysis report saved to {output_path}")
