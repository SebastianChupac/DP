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
