# path.py - Define paths for the bioverify package
# Author: Sebastian Chupac (xchupa03)
# Date: 19.05.2026


from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PUBLIC_DATASET_ROOT = PACKAGE_ROOT / "PublicDataset"