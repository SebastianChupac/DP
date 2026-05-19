# validation.py - CSV manifest validation utilities for biometric datasets.
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026

"""
Validation utilities for dataset CSV manifests.

Checks integrity of CSV files, validates paths, and detects common issues.
"""
import os
import csv
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json


class CSVValidator:
    """Validator for dataset CSV manifests."""
    
    def __init__(self, csv_path: str, base_path: Optional[str] = None):
        """Initialize validator.
        
        Args:
            csv_path: Path to CSV manifest file
            base_path: Base path for resolving relative paths (if None, uses CSV directory)
        """
        self.csv_path = csv_path
        self.base_path = base_path or os.path.dirname(csv_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """Run all validation checks.
        
        Returns:
            True if validation passed (no errors), False otherwise
        """
        print(f"Validating {self.csv_path}...")
        
        self.errors = []
        self.warnings = []
        
        # Check if file exists
        if not os.path.exists(self.csv_path):
            print(f"❌ CSV file not found: {self.csv_path}")
            self.errors.append(f"CSV file not found: {self.csv_path}")
            return False
        
        # Load CSV
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Loaded {len(rows)} rows from CSV")
        
        if not rows:
            self.errors.append("CSV file is empty")
            return False
        
        columns = set(rows[0].keys())
        pair_required_cols = {
            'pair_id', 'image1_path', 'image2_path', 'modality', 'ground_truth',
            'identity1', 'identity2', 'dataset_name'
        }
        identification_required_cols = {
            'record_id', 'sample_id', 'image_path', 'identity',
            'modality', 'dataset_name', 'split', 'metadata'
        }

        if pair_required_cols.issubset(columns):
            print("Detected pair-verification CSV schema")
            self._validate_pair_manifest(rows)
        elif identification_required_cols.issubset(columns):
            print("Detected identification CSV schema")
            self._validate_identification_manifest(rows)
        else:
            self.errors.append(
                "CSV schema not recognized. Missing columns for pair or identification manifests."
            )
            self.errors.append(
                f"Available columns: {sorted(columns)}"
            )
            return False
        
        # Print results
        if self.errors:
            print(f"\n❌ Validation FAILED with {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print(f"\n✅ Validation PASSED")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        return len(self.errors) == 0

    def _validate_pair_manifest(self, rows: List[Dict]):
        """Validation routine for pair-verification CSV manifests."""
        print(f"Checking image paths...")
        self._check_pair_image_paths(rows)
        print(f"Checking ground truth consistency...")
        self._check_ground_truth(rows)
        print(f"Checking for duplicate pairs...")
        self._check_duplicate_pairs(rows)
        print(f"Checking class balance...")
        self._check_balance(rows)
        print(f"Checking metadata JSON validity...")
        self._check_metadata(rows)

    def _validate_identification_manifest(self, rows: List[Dict]):
        """Validation routine for closed-set identification CSV manifests."""
        print(f"Checking image paths...")
        self._check_identification_image_paths(rows)
        print(f"Checking split values and uniqueness...")
        self._check_identification_split_and_ids(rows)
        print(f"Checking gallery/probe disjointness...")
        self._check_identification_disjointness(rows)
        print(f"Checking closed-set identity coverage...")
        self._check_identification_closed_set(rows)
        print(f"Checking metadata JSON validity...")
        self._check_metadata(rows)
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative path to absolute."""
        if os.path.isabs(path):
            return path
        return str(Path(self.base_path) / path)
    
    def _check_pair_image_paths(self, rows: List[Dict]):
        """Check if all pair image paths exist."""
        missing_files = []
        for idx, row in enumerate(rows):
            for img_col in ['image1_path', 'image2_path']:
                path = self._resolve_path(row[img_col])
                if not os.path.exists(path):
                    missing_files.append(f"Row {idx}: {path}")
        
        if missing_files:
            self.errors.append(f"Missing {len(missing_files)} image files")
            if len(missing_files) <= 10:
                for f in missing_files:
                    self.errors.append(f"  {f}")

    def _check_identification_image_paths(self, rows: List[Dict]):
        """Check if all identification image paths exist."""
        missing_files = []
        for idx, row in enumerate(rows):
            path = self._resolve_path(row['image_path'])
            if not os.path.exists(path):
                missing_files.append(f"Row {idx}: {path}")

        if missing_files:
            self.errors.append(f"Missing {len(missing_files)} image files")
            if len(missing_files) <= 10:
                for f in missing_files:
                    self.errors.append(f"  {f}")
            else:
                self.errors.append(f"  (showing first 10)")
                for f in missing_files[:10]:
                    self.errors.append(f"  {f}")

    def _check_identification_split_and_ids(self, rows: List[Dict]):
        """Check split domain and id uniqueness for identification manifests."""
        valid_splits = {"gallery", "probe"}
        seen_record_ids: Set[str] = set()
        split_counts = {"gallery": 0, "probe": 0}

        for idx, row in enumerate(rows):
            split = row['split'].strip().lower()
            if split not in valid_splits:
                self.errors.append(f"Row {idx}: Invalid split value: {row['split']}")
            else:
                split_counts[split] += 1

            record_id = row['record_id']
            if record_id in seen_record_ids:
                self.errors.append(f"Row {idx}: Duplicate record_id: {record_id}")
            seen_record_ids.add(record_id)

        if split_counts["gallery"] == 0:
            self.errors.append("No gallery rows found")
        if split_counts["probe"] == 0:
            self.errors.append("No probe rows found")

    def _check_identification_closed_set(self, rows: List[Dict]):
        """Validate closed-set identity membership: probe identities must exist in gallery."""
        gallery_ids: Set[str] = set()
        probe_ids: Set[str] = set()

        for row in rows:
            split = row['split'].strip().lower()
            identity = row['identity']
            if split == 'gallery':
                gallery_ids.add(identity)
            elif split == 'probe':
                probe_ids.add(identity)

        missing_probe_identities = sorted(probe_ids - gallery_ids)
        if missing_probe_identities:
            self.errors.append(
                f"Closed-set violation: {len(missing_probe_identities)} probe identities not present in gallery"
            )
            if len(missing_probe_identities) <= 10:
                for identity in missing_probe_identities:
                    self.errors.append(f"  Missing in gallery: {identity}")

        gallery_without_probe = sorted(gallery_ids - probe_ids)
        if gallery_without_probe:
            self.warnings.append(
                f"{len(gallery_without_probe)} gallery identities have no probes"
            )

    def _check_identification_disjointness(self, rows: List[Dict]):
        """Check that the same sample/image does not appear in both gallery and probe."""
        gallery_sample_ids: Set[str] = set()
        probe_sample_ids: Set[str] = set()
        gallery_image_paths: Set[str] = set()
        probe_image_paths: Set[str] = set()

        for row in rows:
            split = row['split'].strip().lower()
            sample_id = row.get('sample_id', '')
            image_path = row.get('image_path', '')
            if split == 'gallery':
                gallery_sample_ids.add(sample_id)
                gallery_image_paths.add(image_path)
            elif split == 'probe':
                probe_sample_ids.add(sample_id)
                probe_image_paths.add(image_path)

        overlap_sample_ids = sorted(gallery_sample_ids.intersection(probe_sample_ids))
        overlap_image_paths = sorted(gallery_image_paths.intersection(probe_image_paths))

        if overlap_sample_ids:
            self.errors.append(
                f"Disjointness violation: {len(overlap_sample_ids)} sample_id values appear in both gallery and probe"
            )
            if len(overlap_sample_ids) <= 10:
                for sample_id in overlap_sample_ids:
                    self.errors.append(f"  Overlap sample_id: {sample_id}")

        if overlap_image_paths:
            self.errors.append(
                f"Disjointness violation: {len(overlap_image_paths)} image_path values appear in both gallery and probe"
            )
            if len(overlap_image_paths) <= 10:
                for image_path in overlap_image_paths:
                    self.errors.append(f"  Overlap image_path: {image_path}")
    
    def _check_ground_truth(self, rows: List[Dict]):
        """Check ground truth consistency."""
        for idx, row in enumerate(rows):
            gt = row['ground_truth'].lower()
            if gt not in ('true', 'false', '1', '0', 'yes', 'no'):
                self.errors.append(f"Row {idx}: Invalid ground_truth value: {row['ground_truth']}")
            
            # Check consistency: same identity should be genuine
            is_genuine = gt in ('true', '1', 'yes')
            same_identity = row['identity1'] == row['identity2']
            
            if is_genuine and not same_identity:
                self.errors.append(f"Row {idx}: Marked as genuine but identities differ: {row['identity1']} vs {row['identity2']}")
            if not is_genuine and same_identity:
                self.errors.append(f"Row {idx}: Marked as impostor but identities are same: {row['identity1']}")
    
    def _check_duplicate_pairs(self, rows: List[Dict]):
        """Check for duplicate pairs."""
        seen_pairs = set()
        duplicates = []
        
        for idx, row in enumerate(rows):
            # Create normalized pair tuple (order-independent)
            pair = tuple(sorted([row['image1_path'], row['image2_path']]))
            if pair in seen_pairs:
                duplicates.append(f"Row {idx}: {pair[0]} <-> {pair[1]}")
            seen_pairs.add(pair)
        
        if duplicates:
            self.warnings.append(f"Found {len(duplicates)} duplicate pairs")
            # Show first 10 duplicates

            if len(duplicates) <= 10:
                for dup in duplicates:
                    self.warnings.append(f"  {dup}")
            else:
                self.warnings.append(f"  (showing first 10)")
                for dup in duplicates[:10]:
                    self.warnings.append(f"  {dup}")
    
    def _check_balance(self, rows: List[Dict]):
        """Check class balance."""
        total = len(rows)
        genuine = sum(1 for r in rows if r['ground_truth'].lower() in ('true', '1', 'yes'))
        impostor = total - genuine
        
        genuine_ratio = genuine / total if total > 0 else 0
        if genuine_ratio < 0.3 or genuine_ratio > 0.7:
            self.warnings.append(f"Imbalanced classes: {genuine_ratio*100:.1f}% genuine, {(1-genuine_ratio)*100:.1f}% impostor")
    
    def _check_metadata(self, rows: List[Dict]):
        """Check metadata JSON validity."""
        invalid_json = []
        for idx, row in enumerate(rows):
            if 'metadata' in row and row['metadata']:
                try:
                    json.loads(row['metadata'])
                except json.JSONDecodeError:
                    invalid_json.append(f"Row {idx}")
        
        if invalid_json:
            self.warnings.append(f"Invalid JSON metadata in {len(invalid_json)} rows")


def print_csv_statistics(csv_path: str, base_path: Optional[str] = None):
    """Print detailed statistics about a CSV manifest.
    
    Args:
        csv_path: Path to CSV manfest file
        base_path: Base path for resolving relative paths
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV is empty")
        return

    columns = set(rows[0].keys())
    pair_required_cols = {
        'pair_id', 'image1_path', 'image2_path', 'modality', 'ground_truth',
        'identity1', 'identity2', 'dataset_name'
    }
    identification_required_cols = {
        'record_id', 'sample_id', 'image_path', 'identity',
        'modality', 'dataset_name', 'split', 'metadata'
    }

    if pair_required_cols.issubset(columns):
        from .dataset import PairDataset
        dataset = PairDataset(csv_path, base_path=base_path)
        dataset.print_statistics()
        return

    if identification_required_cols.issubset(columns):
        total = len(rows)
        gallery = sum(1 for row in rows if row['split'].strip().lower() == 'gallery')
        probe = sum(1 for row in rows if row['split'].strip().lower() == 'probe')
        identities = set(row['identity'] for row in rows)
        gallery_ids = set(row['identity'] for row in rows if row['split'].strip().lower() == 'gallery')
        probe_ids = set(row['identity'] for row in rows if row['split'].strip().lower() == 'probe')

        by_modality: Dict[str, int] = {}
        by_dataset: Dict[str, int] = {}
        for row in rows:
            by_modality[row['modality']] = by_modality.get(row['modality'], 0) + 1
            by_dataset[row['dataset_name']] = by_dataset.get(row['dataset_name'], 0) + 1

        print("\n=== Identification Dataset Statistics ===")
        print(f"Total rows: {total}")
        print(f"  Gallery: {gallery}")
        print(f"  Probe: {probe}")
        print(f"Unique identities (all): {len(identities)}")
        print(f"Unique identities (gallery): {len(gallery_ids)}")
        print(f"Unique identities (probe): {len(probe_ids)}")

        print("\nBy modality:")
        for modality, count in by_modality.items():
            print(f"  {modality}: {count}")

        print("\nBy dataset:")
        for dataset_name, count in by_dataset.items():
            print(f"  {dataset_name}: {count}")
        return

    print("Unknown CSV schema, cannot print statistics")

