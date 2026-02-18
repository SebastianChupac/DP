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
        
        # Check required columns
        required_cols = ['pair_id', 'image1_path', 'image2_path', 'modality',
                        'ground_truth', 'identity1', 'identity2', 'dataset_name']
        missing_cols = [col for col in required_cols if col not in rows[0]]
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return False
        
        # Run individual checks
        print(f"Checking image paths...")
        self._check_image_paths(rows)
        print(f"Checking ground truth consistency...")
        self._check_ground_truth(rows)
        print(f"Checking for duplicate pairs...")
        self._check_duplicate_pairs(rows)
        print(f"Checking class balance...")
        self._check_balance(rows)
        print(f"Checking metadata JSON validity...")
        self._check_metadata(rows)
        
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
    
    def _resolve_path(self, path: str) -> str:
        """Resolve relative path to absolute."""
        if os.path.isabs(path):
            return path
        return str(Path(self.base_path) / path)
    
    def _check_image_paths(self, rows: List[Dict]):
        """Check if all image paths exist."""
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
            else:
                self.errors.append(f"  (showing first 10)")
                for f in missing_files[:10]:
                    self.errors.append(f"  {f}")
    
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
    from .dataset import PairDataset
    
    dataset = PairDataset(csv_path, base_path=base_path)
    dataset.print_statistics()

