"""
Pre-compute masks for all images in dataset folders.

Generates masks with dataset-preserving folder structure:
  PublicDataset/Iris/001-CASIA/S0001L01.jpg
  PublicDataset/_masks/Iris/001-CASIA/S0001L01_mask.png

Run once after downloading new datasets, or when updating masking algorithm.
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from ..utils.preprocessing import create_iris_mask, create_face_mask, create_hand_mask


def _detect_iris_side(img_path: Path) -> str:
    """
    Detect iris eye side from image path.
    
    Handles two patterns:
    1. Side in parent folder: .../L/image.jpg, .../R/image.jpg
    2. Side in filename: Iris_20220818_121752_Left.bmp, Iris_20220818_121752_Right.bmp
    
    Args:
        img_path: Image path
        
    Returns:
        'left' or 'right' (defaults to 'left' if not found)
    """
    img_path_str = str(img_path).lower()
    
    # Check for Left/Right in filename
    if 'left' in img_path_str:
        return 'left'
    if 'right' in img_path_str:
        return 'right'
    
    # Check for L/R in parent folder
    parts = img_path.parts
    for part in parts:
        if part.upper() == 'L':
            return 'left'
        if part.upper() == 'R':
            return 'right'
    
    # Default to left if not found
    return 'left'


def precompute_masks(
    dataset_root: str,
    modality: str,
    force: bool = False,
    iris_exclude_pupil: bool = True
) -> None:
    """Pre-compute masks for a single modality.
    
    Creates folder structure:
      dataset_root/_masks/{modality}/{dataset_path}/{image_stem}_mask.png
    
    This preserves dataset folder structure so masks correspond to correct images
    (important when multiple datasets have same image names in different dirs).
    
    Args:
        dataset_root: Root folder containing Iris/, Face/, HandGeometry/, etc. subdirectories
        modality: Modality to process ('iris', 'face', or 'handGeometry')
        force: If True, recompute even if mask already exists
        iris_exclude_pupil: For iris masks, exclude pupil region
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    
    modality_root = root / modality.capitalize()
    if not modality_root.exists():
        raise FileNotFoundError(f"Modality folder not found: {modality_root}")
    
    _precompute_modality_masks(
        modality_root=modality_root,
        modality=modality,
        dataset_root=root,
        force=force,
        iris_exclude_pupil=iris_exclude_pupil
    )


def _precompute_modality_masks(
    modality_root: Path,
    modality: str,
    dataset_root: Path,
    force: bool,
    iris_exclude_pupil: bool
) -> None:
    """Pre-compute masks for a single modality."""
    # Find all images in modality folder
    image_extensions = ('.jpg', '.png', '.bmp', '.jpeg')
    images: List[Tuple[Path, Path]] = []  # (image_path, relative_path)
    
    for img_file in modality_root.rglob('*'):
        if img_file.suffix.lower() in image_extensions and '__MACOSX' not in img_file.parts:
            # Relative path from modality folder: e.g., "001-CASIA/S0001L01.jpg"
            rel_path = img_file.relative_to(modality_root)
            images.append((img_file, rel_path))
    
    if not images:
        print(f"⚠ No images found for {modality} at {modality_root}")
        return
    
    print(f"\n📊 Processing {modality}: {len(images)} images")
    
    # Create masks folder
    masks_root = dataset_root / '_masks' / modality.capitalize()
    masks_root.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for img_path, rel_path in tqdm(images, desc=f"  {modality}"):
        # Compute mask path with same structure as image
        mask_path = masks_root / rel_path.parent / f"{rel_path.stem}_mask.png"
        
        # Skip if already exists and not forcing
        if mask_path.exists() and not force:
            skipped += 1
            continue
        
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue
            
            # Compute mask based on modality
            mask = None
            if modality == "iris":
                # Detect eye side from image path
                iris_side = _detect_iris_side(img_path)
                mask = create_iris_mask(img, exclude_pupil=iris_exclude_pupil, side=iris_side)
            elif modality == "face":
                mask = create_face_mask(img)
            elif modality == "handGeometry":
                mask = create_hand_mask(img)
            
            if mask is not None:
                # Ensure binary (0/255)
                mask = (mask > 0).astype(np.uint8) * 255
                
                # Create parent directory if needed
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save mask
                cv2.imwrite(str(mask_path), mask)
                processed += 1
            else:
                errors += 1
        
        except Exception as e:
            errors += 1
    
    print(f"   ✓ {processed} processed, ⊘ {skipped} skipped, ✗ {errors} errors")


def main():
    """CLI entry point for mask pre-computation."""
    parser = argparse.ArgumentParser(
        description='Pre-compute masks for a single modality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-compute iris masks
  python -m bioverify.experiments.precompute_masks \\
      --dataset-root C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset \\
      --modality iris

  # Pre-compute face masks
  python -m bioverify.experiments.precompute_masks \\
      --dataset-root C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset \\
      --modality face

  # Force recomputation of hand masks
  python -m bioverify.experiments.precompute_masks \\
      --dataset-root C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset \\
      --modality hand \\
      --force
        """
    )
    
    parser.add_argument(
        '--dataset-root',
        required=True,
        help='Root folder containing Iris/, Face/, HandGeometry/ subdirectories'
    )
    parser.add_argument(
        '--modality',
        required=True,
        choices=['iris', 'face', 'handGeometry'],
        help='Modality to process'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute even if mask already exists'
    )
    
    args = parser.parse_args()
    
    precompute_masks(
        dataset_root=args.dataset_root,
        modality=args.modality,
        force=args.force
    )


if __name__ == '__main__':
    main()
