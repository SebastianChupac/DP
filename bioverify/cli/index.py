"""
CLI commands for dataset indexing.

Provides command-line interface for indexing datasets and generating pair manifests.
"""
import argparse
from typing import Optional
import yaml
from pathlib import Path
from datetime import datetime

from ..data.indexer import DatasetIndexer
from ..data.validation import CSVValidator, print_csv_statistics
from ..matchers.registry import create_matcher
from ..experiments.runner import run_experiment
from ..experiments.identification_runner import run_identification_experiment
from ..evaluation.cli import threshold_sweep_command, compare_matchers_command
from ..results import VisualizationResult
from ..visualization import (
    render_match_visualization,
    save_visualization_image,
    show_visualization_image,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_ground_truth(value: str) -> Optional[bool]:
    """Parse ground truth CLI value into a boolean or None."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"same", "true", "1", "yes", "y"}:
        return True
    if normalized in {"different", "diff", "false", "0", "no", "n"}:
        return False
    raise ValueError(
        "Invalid ground truth value. Use: same|different|true|false|1|0."
    )


def index_datasets_command(args):
    """Execute dataset indexing command.
    
    Args:
        args: Parsed command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override output path if specified
    if args.output:
        config['output']['csv_path'] = args.output
    
    # Create indexer
    indexer = DatasetIndexer(
        public_dataset_root=config['public_dataset_root'],
        random_seed=config.get('random_seed', 42)
    )

    identification_generation = config.get('identification_generation')
    if identification_generation:
        rows = indexer.index_and_generate_identification(
            dataset_configs=config['datasets'],
            output_csv=config['output']['csv_path'],
            gallery_samples_per_identity=identification_generation['gallery_samples_per_identity'],
            probes_per_identity=identification_generation['probes_per_identity'],
            number_of_identities=identification_generation.get(
                'number_of_identities',
                identification_generation.get('num_identities', -1)
            ),
            relative_paths=config['output'].get('relative_paths', True),
            require_session_disjoint=identification_generation.get('require_session_disjoint', False),
            identification_filters=(
                identification_generation.get('filters')
                or identification_generation.get('match_constraints')
            ),
        )
        indexer.print_statistics()
        print(f"\nGenerated identification manifest with {len(rows)} rows")
    else:
        # Index and generate verification pairs
        pairs = indexer.index_and_generate(
            dataset_configs=config['datasets'],
            output_csv=config['output']['csv_path'],
            genuine_per_identity=config['pair_generation'].get('genuine_per_identity'),
            max_genuine_pairs=config['pair_generation'].get('max_genuine_pairs'),
            impostor_ratio=config['pair_generation'].get('impostor_ratio', 1.0),
            relative_paths=config['output'].get('relative_paths', True)
        )

        # Print statistics
        indexer.print_statistics(pairs)
    
    print(f"\n✅ Indexing complete!")
    print(f"   Output saved to: {config['output']['csv_path']}")
    
    # Validate output if requested
    if args.validate:
        print(f"\nValidating output CSV...")
        validator = CSVValidator(
            config['output']['csv_path'],
            base_path=config['public_dataset_root']
        )
        result = validator.validate()
        if not result:
            print(f"❌ Validation failed for output CSV: {config['output']['csv_path']}")
        else:
            print(f"✅ Output CSV validation passed!")

def validate_command(args):
    """Execute CSV validation command.
    
    Args:
        args: Parsed command-line arguments
    """
    validator = CSVValidator(
        args.csv,
        base_path=args.base_path
    )
    
    valid = validator.validate()
    
    if args.stats:
        print("\n" + "="*50)
        print_csv_statistics(args.csv, args.base_path)
    
    # Return exit code
    return 0 if valid else 1


def stats_command(args):
    """Execute statistics display command.
    
    Args:
        args: Parsed command-line arguments
    """
    print_csv_statistics(args.csv, args.base_path)


def match_command(args):
    """Execute a single-pair match using a configured matcher.
    
    Args:
        args: Parsed command-line arguments
    """
    config = load_config(args.config)
    matcher_block = config.get("matcher", {})
    matcher_name = args.matcher or matcher_block.get("name")
    if not matcher_name:
        raise ValueError("Matcher name is required (use --matcher or config.matcher.name)")

    matcher_config = matcher_block.get("config", {}).copy()
    if config.get("base_path") and "public_dataset_root" not in matcher_config:
        matcher_config["public_dataset_root"] = config["base_path"]
    matcher = create_matcher(matcher_name, matcher_config)

    if args.viz_output and not args.viz:
        print("⚠ --viz-output is ignored unless --viz is enabled.")

    ground_truth = parse_ground_truth(args.ground_truth)

    result = matcher.match(
        args.image1,
        args.image2,
        modality=args.modality,
        visualize=args.viz,
        ground_truth=ground_truth,
        matcher_name=matcher_name,  # Pass the full matcher name (e.g., "sift-v1")
    )

    if args.viz and isinstance(result, VisualizationResult):
        rendered = render_match_visualization(
            result,
            viz_mode=args.viz_mode,
            image_mode=args.image_mode,
        )
        if args.viz_output:
            output_path = args.viz_output
        else:
            image1_stem = Path(args.image1).stem
            image2_stem = Path(args.image2).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                Path("bioverify/results/single_match_visualizations")
                / f"{matcher_name}_{image1_stem}_vs_{image2_stem}_{timestamp}.png"
            )

        saved_path = save_visualization_image(rendered, output_path)
        print(f"✓ Visualization saved to: {saved_path}")
        show_visualization_image(rendered, title=f"{matcher_name} match visualization")

    if args.full:
        print(result)
    else:
        result.print_summary()


def experiment_command(args):
    """Execute a batch experiment on pairs.
    
    Args:
        args: Parsed command-line arguments
    """
    try:
        results, metrics = run_experiment(args.config)
        
        # Note: Summary metrics are already printed by the runner
        # Just print experiment completion message
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
    
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


def identification_experiment_command(args):
    """Execute a batch closed-set identification experiment.

    Args:
        args: Parsed command-line arguments
    """
    try:
        results, metrics = run_identification_experiment(args.config)

        print("\n" + "=" * 60)
        print("IDENTIFICATION EXPERIMENT COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Identification experiment failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='BioVerify Dataset Indexing and Experimentation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index datasets using config file
  python -m bioverify.cli.index index --config config/indexing/iris.yaml

  # Index and validate in one step
  python -m bioverify.cli.index index --config config/indexing/test_mmu.yaml --validate

  # Validate existing CSV
  python -m bioverify.cli.index validate --csv data/pairs/iris_pairs.csv

  # Show statistics
  python -m bioverify.cli.index stats --csv data/pairs/iris_pairs.csv

  # Run a matcher on a single pair
  python -m bioverify.cli.index match \
      --config config/matching/loftr.yaml \
      --image1 path/to/img1.png --image2 path/to/img2.png

  # Run a batch experiment
  python -m bioverify.cli.index experiment \
      --config config/experiments/exp_loftr_iris.yaml

    # Run a closed-set identification experiment
    python -m bioverify.cli.index identification \
            --config config/experiments/id_iris_casia.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index datasets and generate pairs')
    index_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to YAML configuration file'
    )
    index_parser.add_argument(
        '--output', '-o',
        help='Override output CSV path from config'
    )
    index_parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate output CSV after generation'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate CSV manifest')
    validate_parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV manifest file'
    )
    validate_parser.add_argument(
        '--base-path',
        default='C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset',
        help='Base path for resolving relative paths (default: PublicDataset)'
    )
    validate_parser.add_argument(
        '--stats',
        action='store_true',
        help='Print statistics after validation'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Print CSV statistics')
    stats_parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV manifest file'
    )
    stats_parser.add_argument(
        '--base-path',
        default='C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset',
        help='Base path for resolving relative paths (default: PublicDataset)'
    )

    # Match command
    match_parser = subparsers.add_parser('match', help='Run matcher on a single pair')
    match_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to matcher YAML configuration file'
    )
    match_parser.add_argument(
        '--matcher', '-m',
        help='Override matcher name (e.g., sift)'
    )
    match_parser.add_argument(
        '--image1',
        required=True,
        help='Path to first image'
    )
    match_parser.add_argument(
        '--image2',
        required=True,
        help='Path to second image'
    )
    match_parser.add_argument(
        '--modality',
        help='Optional modality hint (iris, face, hand, fingervein)'
    )
    match_parser.add_argument(
        '--ground-truth',
        help='Optional ground truth label: same|different|true|false|1|0'
    )
    match_parser.add_argument(
        '--full',
        action='store_true',
        help='Print full VerificationResult instead of summary'
    )
    match_parser.add_argument(
        '--viz',
        action='store_true',
        help='Enable single-pair visualization (render, save, and display)'
    )
    match_parser.add_argument(
        '--viz-output',
        help='Output file path for rendered visualization image'
    )
    match_parser.add_argument(
        '--viz-mode',
        choices=['m', 'k', 'b'],
        default='m',
        help='Visualization mode: m=matches (default), k=keypoints, b=both'
    )
    match_parser.add_argument(
        '--image-mode',
        choices=['o', 'p'],
        default='p',
        help='Image source mode: o=original image, p=processed matcher input (default)'
    )

    # Experiment command
    experiment_parser = subparsers.add_parser('experiment', help='Run batch experiment on pairs')
    experiment_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to experiment YAML configuration file'
    )
    experiment_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including tracebacks'
    )

    # Identification experiment command
    identification_parser = subparsers.add_parser(
        'identification',
        help='Run batch closed-set identification experiment'
    )
    identification_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to identification experiment YAML configuration file'
    )
    identification_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including tracebacks'
    )
    
    # Evaluate command (with subcommands)
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate experiment results')
    evaluate_subparsers = evaluate_parser.add_subparsers(dest='evaluate_command', help='Evaluation subcommand')
    
    # Threshold sweep subcommand
    threshold_parser = evaluate_subparsers.add_parser('threshold', help='Perform threshold sweep analysis')
    threshold_parser.add_argument(
        '--experiment', '-e',
        required=True,
        dest='experiment_name',
        help='Name of the experiment (found in bioverify/results/{experiment_name})'
    )
    threshold_parser.add_argument(
        '--matcher', '-m',
        help='Optional: analyze specific matcher only (if not specified, analyzes all)'
    )
    threshold_parser.add_argument(
        '--output', '-o',
        help='Output directory for plots and results (default: {experiment_dir}/evaluation)'
    )
    
    # Matcher comparison subcommand
    compare_parser = evaluate_subparsers.add_parser('compare', help='Compare matchers')
    compare_parser.add_argument(
        '--experiment', '-e',
        required=True,
        dest='experiment_name',
        help='Name of the experiment'
    )
    compare_parser.add_argument(
        '--mode',
        choices=['eer', 'far', 'frr', 'threshold'],
        default='eer',
        help='Comparison mode: eer (default, recommended) | far (fixed FAR) | frr (fixed FRR) | threshold (raw threshold)'
    )
    compare_parser.add_argument(
        '--value',
        type=float,
        help='Target value for mode: FAR value (e.g., 0.001), FRR value (e.g., 0.01), or threshold value (e.g., 0.5). Not used for EER mode.'
    )
    compare_parser.add_argument(
        '--output', '-o',
        help='Output directory for results'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'index':
        index_datasets_command(args)
    elif args.command == 'validate':
        exit(validate_command(args))
    elif args.command == 'stats':
        stats_command(args)
    elif args.command == 'match':
        match_command(args)
    elif args.command == 'experiment':
        experiment_command(args)
    elif args.command == 'identification':
        identification_experiment_command(args)
    elif args.command == 'evaluate':
        if args.evaluate_command == 'threshold':
            threshold_sweep_command(args)
        elif args.evaluate_command == 'compare':
            # Validate required arguments for compare modes
            if args.mode in ['far', 'frr', 'threshold'] and args.value is None:
                print(f"❌ --value is required for mode '{args.mode}'")
                print(f"   Example: --mode {args.mode} --value 0.01")
                compare_parser.print_help()
                exit(1)
            compare_matchers_command(args)
        else:
            evaluate_parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
