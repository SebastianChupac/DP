"""
CLI commands for dataset indexing.

Provides command-line interface for indexing datasets and generating pair manifests.
"""
import argparse
import yaml
from pathlib import Path

from ..data.indexer import DatasetIndexer
from ..data.validation import CSVValidator, print_csv_statistics


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
    
    # Index and generate pairs
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='BioVerify Dataset Indexing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index datasets using config file
  python -m bioverify.cli.index --config config/indexing/iris.yaml

  # Index and validate in one step
  python -m bioverify.cli.index --config config/indexing/test_mmu.yaml --validate

  # Validate existing CSV
  python -m bioverify.cli.index validate --csv dataset_index.csv

  # Show statistics
  python -m bioverify.cli.index stats --csv dataset_index.csv
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'index':
        index_datasets_command(args)
    elif args.command == 'validate':
        exit(validate_command(args))
    elif args.command == 'stats':
        stats_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
