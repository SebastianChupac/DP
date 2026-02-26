"""
Utility to pre-compute and cache masks for all images in a pairs CSV.

This avoids the expensive mask computation during experiments.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Set
import cv2
import numpy as np
from tqdm import tqdm

from ..utils.preprocessing import create_iris_mask, create_face_mask, create_hand_mask


def precompute_masks(
    pairs_csv: str,
    base_path: str,
    cache_dir: str,
    modality: Optional[str] = None,
    force: bool = False
) -> None:
    """Pre-compute masks for all images in a pairs CSV.
    
    Args:
        pairs_csv: Path to pairs CSV file
        base_path: Base path for resolving relative image paths
        cache_dir: Directory to save cached masks
        modality: Optional modality filter (only process this modality)
        force: If True, recompute even if cached mask exists
    """
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Collect unique images
    image_paths: Set[tuple] = set()
    
    print(f"📂 Loading pairs from {pairs_csv}...")
    with open(pairs_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_modality = row['modality']
            
            # Apply modality filter
            if modality and pair_modality != modality:
                continue
            
            # Add both images with their modality
            image_paths.add((row['image1_path'], pair_modality))
            image_paths.add((row['image2_path'], pair_modality))
    
    print(f"✓ Found {len(image_paths)} unique images to process")
    
    # Process each image
    processed = 0
    skipped = 0
    errors = 0
    
    for img_path, img_modality in tqdm(image_paths, desc="Computing masks"):
        # Resolve image path
        full_path = Path(base_path) / img_path
        if not full_path.exists():
            print(f"⚠ Image not found: {full_path}")
            errors += 1
            continue
        
        # Create cache filename
        img_name = full_path.stem
        mask_filename = f"{img_name}_{img_modality}_mask.png"
        mask_path = cache_path / mask_filename
        
        # Skip if already cached and not forcing
        if mask_path.exists() and not force:
            skipped += 1
            continue
        
        # Skip if modality doesn't need masking
        if img_modality == "fingervein":
            skipped += 1
            continue
        
        try:
            # Load image
            img = cv2.imread(str(full_path))
            if img is None:
                print(f"⚠ Failed to load: {full_path}")
                errors += 1
                continue
            
            # Compute mask based on modality
            mask = None
            if img_modality == "iris":
                mask = create_iris_mask(img, exclude_pupil=True)
            elif img_modality == "face":
                mask = create_face_mask(img)
            elif img_modality == "hand":
                mask = create_hand_mask(img)
            
            if mask is not None:
                # Ensure binary (0/255)
                mask = (mask > 0).astype(np.uint8) * 255
                
                # Save mask
                cv2.imwrite(str(mask_path), mask)
                processed += 1
            else:
                errors += 1
        
        except Exception as e:
            print(f"⚠ Error processing {img_path}: {str(e)}")
            errors += 1
    
    print(f"\n✅ Mask pre-computation complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped: {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Cache dir: {cache_dir}")


def main():
    """CLI entry point for mask pre-computation."""
    parser = argparse.ArgumentParser(
        description='Pre-compute masks for images in pairs CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-compute masks for all iris pairs
  python -m bioverify.experiments.precompute_masks \\
      --pairs data/pairs/test001_iris_mmu.csv \\
      --cache-dir cache/masks \\
      --modality iris

  # Pre-compute for all modalities
  python -m bioverify.experiments.precompute_masks \\
      --pairs data/pairs/all_pairs.csv \\
      --cache-dir cache/masks

  # Force recomputation
  python -m bioverify.experiments.precompute_masks \\
      --pairs data/pairs/iris_pairs.csv \\
      --cache-dir cache/masks \\
      --force
        """
    )
    
    parser.add_argument(
        '--pairs',
        required=True,
        help='Path to pairs CSV file'
    )
    parser.add_argument(
        '--base-path',
        default='C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset',
        help='Base path for resolving image paths'
    )
    parser.add_argument(
        '--cache-dir',
        required=True,
        help='Directory to save cached masks'
    )
    parser.add_argument(
        '--modality',
        choices=['iris', 'face', 'hand', 'fingervein'],
        help='Only process images of this modality'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute even if cached mask exists'
    )
    
    args = parser.parse_args()
    
    precompute_masks(
        pairs_csv=args.pairs,
        base_path=args.base_path,
        cache_dir=args.cache_dir,
        modality=args.modality,
        force=args.force
    )


if __name__ == '__main__':
    main()
