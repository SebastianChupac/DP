"""
CLI commands for experiment evaluation.

Provides commands for threshold sweeping, matcher comparison, and analysis.
"""

import json
from pathlib import Path
from typing import Optional

from .analyzer import ThresholdAnalyzer, MatcherComparator
from .plotter import (
    plot_roc_curve,
    plot_det_curve,
    plot_roc_comparison,
    plot_score_distribution,
    save_analysis_report,
    NumpyEncoder,
)


def threshold_sweep_command(args):
    """
    Execute threshold sweep analysis for an experiment.
    
    Sweeps across decision thresholds to compute ROC/DET/EER metrics.
    
    Args:
        args: Parsed command-line arguments with:
            - experiment_name: Name of the experiment
            - matcher: Optional specific matcher to analyze
            - output: Optional output directory
    """
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("bioverify/results") / args.experiment_name / "evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_path = Path("bioverify/results") / args.experiment_name / "results.json"
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results_dict = json.load(f)
    
    # Get all matchers if not specified
    all_matchers = results_dict.get('statistics', {}).get('matchers', [])
    
    if args.matcher:
        matchers_to_analyze = [args.matcher]
    else:
        matchers_to_analyze = all_matchers
    
    print(f"\n📊 Threshold Sweep Analysis")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Matchers: {len(matchers_to_analyze)}")
    print(f"   Output: {output_dir}\n")
    
    all_analysis = {}
    all_scores = {}
    
    for matcher_name in matchers_to_analyze:
        print(f"   Analyzing {matcher_name}...", end=" ", flush=True)
        
        try:
            analyzer = ThresholdAnalyzer(results_dict, matcher_name)
            analysis = analyzer.sweep_threshold(num_points=1000)
            all_analysis[matcher_name] = analysis
            all_scores[matcher_name] = {
                'scores': analyzer.scores,
                'ground_truth': analyzer.ground_truth,
            }
            
            # Print key metrics
            eer = analysis['eer']
            print(f"✓ (EER={eer['eer']:.4f})")
            
        except ValueError as e:
            print(f"✗ ({e})")
    
    # Save analysis
    print(f"\n💾 Saving results...")
    
    # Save individual matcher analyses
    for matcher_name, analysis in all_analysis.items():
        # Save JSON
        json_path = output_dir / f"{matcher_name}_threshold_analysis.json"
        save_analysis_report(analysis, json_path)
        
        # Save plots
        roc_path = output_dir / f"{matcher_name}_roc.png"
        det_path = output_dir / f"{matcher_name}_det.png"
        dist_path = output_dir / f"{matcher_name}_score_distribution.png"
        
        try:
            plot_roc_curve(analysis, roc_path)
            plot_det_curve(analysis, det_path)
            if matcher_name in all_scores:
                plot_score_distribution(
                    all_scores[matcher_name]['scores'],
                    all_scores[matcher_name]['ground_truth'],
                    matcher_name,
                    analysis['current_operating_point'].get('threshold'),
                    analysis['eer'].get('threshold'),
                    dist_path,
                )
        except Exception as e:
            print(f"⚠ Could not generate plots: {e}")
    
    # Save comparison if multiple matchers
    if len(all_analysis) > 1:
        print(f"\n📈 Generating comparison plots...")
        comp_roc_path = output_dir / "roc_comparison.png"
        try:
            plot_roc_comparison(all_analysis, comp_roc_path)
        except Exception as e:
            print(f"⚠ Could not generate comparison plot: {e}")
        
        # Save EER comparison table
        eer_comparison = {}
        for matcher_name, analysis in all_analysis.items():
            eer = analysis['eer']
            eer_comparison[matcher_name] = {
                'eer': eer['eer'],
                'threshold': eer['threshold'],
                'far': eer['far'],
                'frr': eer['frr'],
            }
        
        eer_path = output_dir / "eer_comparison.json"
        with open(eer_path, 'w') as f:
            json.dump(eer_comparison, f, indent=2, cls=NumpyEncoder)
        print(f"✓ EER comparison saved to {eer_path}")
    
    # Print summary
    print(f"\n✅ Threshold sweep complete!")
    print(f"   Output directory: {output_dir}")
    print(f"\n📊 Summary:")
    
    for matcher_name in sorted(all_analysis.keys()):
        analysis = all_analysis[matcher_name]
        eer = analysis['eer']
        roc = analysis['roc']
        
        print(f"\n   {matcher_name}:")
        print(f"      Pairs: {analysis['num_pairs']} (Genuine: {analysis['num_genuine']}, Impostor: {analysis['num_impostor']})")
        print(f"      EER: {eer['eer']:.4f} @ threshold {eer['threshold']:.4f}")
        print(f"      ROC AUC: {roc['auc']:.4f}")
        print(f"      Current: TAR={analysis['current_operating_point']['tar']:.4f}, FAR={analysis['current_operating_point']['far']:.4f}")


def compare_matchers_command(args):
    """
    Execute matcher comparison.
    
    Supports three comparison modes:
    - 'eer': Compare matchers at their respective EER points (default, recommended)
    - 'far': Compare matchers at a fixed FAR (e.g., 0.001 for 0.1%)
    - 'frr': Compare matchers at a fixed FRR (e.g., 0.01 for 1%)
    - 'threshold': Compare at raw threshold (only for same matcher with different params)
    
    Args:
        args: Parsed command-line arguments with:
            - experiment_name: Name of the experiment
            - mode: Comparison mode ('eer', 'far', 'frr', 'threshold')
            - value: Target value for FAR/FRR/threshold mode
            - output: Optional output directory
    """
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("bioverify/results") / args.experiment_name / "evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_path = Path("bioverify/results") / args.experiment_name / "results.json"
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results_dict = json.load(f)
    
    print(f"\n📊 Matcher Comparison")
    print(f"   Experiment: {args.experiment_name}")
    
    # Create comparator
    comparator = MatcherComparator(results_dict)
    
    # Execute comparison based on mode
    if args.mode == 'eer':
        print(f"   Mode: EER (each matcher at its optimal point)\n")
        comparison = comparator.compare_at_eer()
        _print_eer_comparison(comparison)
        save_path = output_dir / "comparison_at_eer.json"
        
    elif args.mode == 'far':
        target_far = args.value
        print(f"   Mode: Fixed FAR = {target_far:.4f}\n")
        comparison = comparator.compare_at_far(target_far)
        _print_far_comparison(comparison)
        save_path = output_dir / f"comparison_at_far_{target_far:.4f}.json"
        
    elif args.mode == 'frr':
        target_frr = args.value
        print(f"   Mode: Fixed FRR = {target_frr:.4f}\n")
        comparison = comparator.compare_at_frr(target_frr)
        _print_frr_comparison(comparison)
        save_path = output_dir / f"comparison_at_frr_{target_frr:.4f}.json"
        
    elif args.mode == 'threshold':
        threshold = args.value
        print(f"   Mode: Fixed Threshold = {threshold:.4f}")
        print(f"   ⚠ Warning: Only use this for comparing same matcher with different params!\n")
        comparison = comparator.compare_at_threshold(threshold)
        _print_threshold_comparison(comparison, threshold)
        save_path = output_dir / f"comparison_at_threshold_{threshold:.4f}.json"
    
    else:
        print(f"❌ Unknown comparison mode: {args.mode}")
        return
    
    # Save comparison
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ Comparison saved to {save_path}")


def _print_eer_comparison(comparison: dict):
    """Print EER-based comparison results."""
    print(f"{'Rank':<6} {'Matcher':<30} {'EER':<10} {'Threshold':<12} {'TAR':<8} {'FAR':<8}")
    print("-" * 80)
    
    for rank, item in enumerate(comparison['ranking'], 1):
        print(f"{rank:<6} {item['matcher']:<30} {item['eer']:.4f}    {item['threshold']:.6f}    {item['tar']:.4f}  {item['far']:.4f}")


def _print_far_comparison(comparison: dict):
    """Print FAR-based comparison results."""
    target_far = comparison['target_far']
    print(f"All matchers at FAR = {target_far:.4f}:")
    print(f"\n{'Rank':<6} {'Matcher':<30} {'TAR':<10} {'Threshold':<12} {'Actual FAR':<12}")
    print("-" * 80)
    
    for rank, item in enumerate(comparison['ranking'], 1):
        print(f"{rank:<6} {item['matcher']:<30} {item['tar']:.4f}    {item['threshold']:.6f}    {item['far']:.4f}")


def _print_frr_comparison(comparison: dict):
    """Print FRR-based comparison results."""
    target_frr = comparison['target_frr']
    print(f"All matchers at FRR = {target_frr:.4f}:")
    print(f"\n{'Rank':<6} {'Matcher':<30} {'TRR':<10} {'Threshold':<12} {'Actual FRR':<12}")
    print("-" * 80)
    
    for rank, item in enumerate(comparison['ranking'], 1):
        print(f"{rank:<6} {item['matcher']:<30} {item['trr']:.4f}    {item['threshold']:.6f}    {item['frr']:.4f}")


def _print_threshold_comparison(comparison: dict, threshold: float):
    """Print threshold-based comparison results."""
    print(f"{'Matcher':<30} {'TAR':<8} {'FAR':<8} {'FRR':<8} {'TRR':<8}")
    print("-" * 60)
    
    for matcher_name in sorted(comparison.keys()):
        metrics = comparison[matcher_name]
        print(f"{matcher_name:<30} {metrics['tar']:.4f}  {metrics['far']:.4f}  {metrics['frr']:.4f}  {metrics['trr']:.4f}")
