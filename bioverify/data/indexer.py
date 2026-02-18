"""
Dataset indexer for creating CSV manifests from biometric datasets.

Scans dataset directories, parses structures, generates pairs, and creates
CSV files for use in experiments.
"""
import os
import csv
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

from .parsers import get_parser, ImageRecord
from .pairs import PairGenerator, ImagePair


class DatasetIndexer:
    """Main class for indexing datasets and generating pair CSV manifests."""
    
    def __init__(self, public_dataset_root: str, random_seed: int = 42):
        """Initialize dataset indexer.
        
        Args:
            public_dataset_root: Path to PublicDataset directory
            random_seed: Random seed for reproducible pair generation
        """
        self.public_dataset_root = Path(public_dataset_root)
        self.pair_generator = PairGenerator(random_seed=random_seed)
        self.indexed_records: Dict[str, List[ImageRecord]] = {}
    
    def index_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        modality: str,
        parser_name: Optional[str] = None,
        image_type: Optional[str] = None,
        modality_type: Optional[str] = None
    ) -> List[ImageRecord]:
        """Index a single dataset.
        
        Args:
            dataset_path: Relative path from public_dataset_root or absolute path
            dataset_name: Name identifier for the dataset
            modality: Biometric modality (iris, face, hand, fingervein)
            parser_name: Optional specific parser to use
            image_type: Optional image type filter (e.g., "raw", "processed")
            modality_type: Optional modality type filter (e.g., "dorsal", "vein")
        Returns:
            List of ImageRecord objects
        """
        # Resolve path
        if os.path.isabs(dataset_path):
            full_path = dataset_path
        else:
            full_path = str(self.public_dataset_root / dataset_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset path not found: {full_path}")
        
        print(f"Indexing {dataset_name} at {full_path}...")
        
        # Get appropriate parser
        parser = get_parser(full_path, dataset_name, modality, image_type=image_type, modality_type=modality_type)
        print(f"Using parser: {parser.__class__.__name__}")
        
        # Parse dataset
        records = parser.parse()
        
        # Store for later use
        self.indexed_records[dataset_name] = records
        
        # Print summary
        num_identities = len(set(r.identity for r in records))
        print(f"  Found {len(records)} images from {num_identities} identities")
        
        return records
    
    def generate_pairs_from_records(
        self,
        records: List[ImageRecord],
        genuine_per_identity: Optional[int] = None,
        max_genuine_pairs: Optional[int] = None,
        impostor_ratio: float = 1.0,
        match_constraints: Optional[Dict[str, bool]] = None
    ) -> List[ImagePair]:
        """Generate pairs from indexed records.
        
        Args:
            records: List of image records
            genuine_per_identity: Number of genuine pairs per identity. If None, generates all possible.
            max_genuine_pairs: Maximum number of genuine pairs. If set to -1, generate
                all possible genuine pairs up to a safety cap.
            impostor_ratio: Ratio of impostor to genuine pairs (1.0 = equal number)
            match_constraints: Optional constraints for impostor pair matching
            
        Returns:
            List of ImagePair objects
        """
        pairs = self.pair_generator.generate_pairs(
            records=records,
            genuine_per_identity=genuine_per_identity,
            max_genuine_pairs=max_genuine_pairs,
            impostor_ratio=impostor_ratio,
            match_constraints=match_constraints
        )
        
        return pairs
    
    def save_pairs_to_csv(
        self,
        pairs: List[ImagePair],
        output_path: str,
        relative_to: Optional[str] = None
    ):
        """Save pairs to CSV file.
        
        Args:
            pairs: List of ImagePair objects
            output_path: Path to output CSV file
            relative_to: Optional base path to make image paths relative to
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Convert to dictionaries
        pair_dicts = []
        for pair in pairs:
            pair_dict = pair.to_dict()
            
            # Make paths relative if requested
            if relative_to:
                pair_dict['image1_path'] = os.path.relpath(pair_dict['image1_path'], relative_to)
                pair_dict['image2_path'] = os.path.relpath(pair_dict['image2_path'], relative_to)
            
            pair_dicts.append(pair_dict)
        
        # Write CSV
        if pair_dicts:
            fieldnames = pair_dicts[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pair_dicts)
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
    
    def index_and_generate(
        self,
        dataset_configs: List[Dict],
        output_csv: str,
        genuine_per_identity: Optional[int] = None,
        max_genuine_pairs: Optional[int] = None,
        impostor_ratio: float = 1.0,
        relative_paths: bool = True
    ):
        """Complete workflow: index datasets and generate pairs in one call.
        
        Args:
            dataset_configs: List of dicts with keys: dataset_path, dataset_name, modality
            output_csv: Path to output CSV file
            genuine_per_identity: Number of genuine pairs per identity. If None, generates all possible.
            max_genuine_pairs: Maximum number of genuine pairs. If set to -1, generate
                all possible genuine pairs up to a safety cap.
            impostor_ratio: Ratio of impostor to genuine pairs (1.0 = equal number)
            relative_paths: Whether to save relative paths in CSV
        """
        all_pairs = []
        
        for config in dataset_configs:
            # Index dataset
            records = self.index_dataset(
                dataset_path=config['dataset_path'],
                dataset_name=config['dataset_name'],
                modality=config['modality'],
                image_type=config.get('image_type'),
                modality_type=config.get('modality_type')
            )
            
            # Generate pairs
            pairs = self.generate_pairs_from_records(
                records=records,
                genuine_per_identity=genuine_per_identity,
                max_genuine_pairs=max_genuine_pairs,
                impostor_ratio=impostor_ratio,
                match_constraints=config.get('match_constraints')
            )
            
            all_pairs.extend(pairs)
            
            # Print statistics
            genuine_count = sum(1 for p in pairs if p.ground_truth)
            impostor_count = len(pairs) - genuine_count
            
            print(f"  Generated {len(pairs)} pairs:")
            print(f"    - {genuine_count} genuine, {impostor_count} impostor")
        
        # Save all pairs
        relative_to = str(self.public_dataset_root) if relative_paths else None
        self.save_pairs_to_csv(all_pairs, output_csv, relative_to=relative_to)
        
        return all_pairs
    
    def print_statistics(self, pairs: Optional[List[ImagePair]] = None):
        """Print detailed statistics about indexed datasets and pairs.
        
        Args:
            pairs: Optional list of pairs. If None, prints only dataset stats.
        """
        print("\n=== Dataset Index Statistics ===")
        
        for dataset_name, records in self.indexed_records.items():
            num_identities = len(set(r.identity for r in records))
            print(f"\n{dataset_name}:")
            print(f"  Images: {len(records)}")
            print(f"  Identities: {num_identities}")
            print(f"  Modality: {records[0].modality if records else 'N/A'}")
        
        if pairs:
            print("\n=== Pair Statistics ===")
            print(f"Total pairs: {len(pairs)}")
            
            genuine = sum(1 for p in pairs if p.ground_truth)
            impostor = len(pairs) - genuine
            print(f"  Genuine: {genuine} ({genuine/len(pairs)*100:.1f}%)")
            print(f"  Impostor: {impostor} ({impostor/len(pairs)*100:.1f}%)")
            
            by_modality = {}
            for pair in pairs:
                by_modality[pair.modality] = by_modality.get(pair.modality, 0) + 1
            
            print(f"\nBy modality:")
            for modality, count in by_modality.items():
                print(f"  {modality}: {count}")
