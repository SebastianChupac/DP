# BioVerify Framework - Foundation Complete ✅

## Implemented steps (Steps 1-2)

### 1. Core Package Structure
- Created `src/bioverify/` package with modular organization:
  - `bioverify/` - Core package
  - `bioverify/data/` - Dataset handling (parsers, pairs, indexer, loader, validation)
  - `bioverify/config/` - Configuration management
  - `bioverify/utils/` - Preprocessing utilities (masking, resizing)
  - `bioverify/cli/` - Command-line interface
  - `bioverify/results.py` - Result data structures (moved from VerificationResult.py)

### 2. Dataset Indexing System
- **Parsers** (`parsers.py`): Handles different dataset structures
  - CASIA-Iris-Thousand
  - MMU-Iris-Database
  - AMF-Iris-Dataset
  - Multi-PIE Face
  - 11k Hands
  - Finger Vein (generic structure)
  - FV-USM Finger Vein Database (with raw/extracted vein variants)
  - THU-FVFDT (dorsal and finger vein with train/test sessions)
  - MMCBNU-6000 Finger Vein (with raw/ROI variants)
  - EEMSC-DBM Finger Vein Database
  - Automatically excludes `__MACOSX` metadata folders
  - Extensible framework for adding new datasets

- **Pair Generator** (`pairs.py`): Creates verification pairs for inference/evaluation
  - Genuine pairs (same identity)
  - Impostor pairs (different identities)
  - Per-identity or maximum-based pair limiting
  - Optional matching constraints (e.g., same side for hand/vein)
  - No train/val/test splitting (for inference-focused workflow)

- **Dataset Indexer** (`indexer.py`): Orchestrates the workflow
  - Scans PublicDataset directory
  - Parses structure
  - Generates pairs
  - Saves CSV manifests

- **CSV Loader** (`dataset.py`): Loads pairs for experiments
  - Filtering by modality/dataset
  - Lazy or eager image loading
  - Batch iteration support

- **Validation** (`validation.py`): Ensures data integrity
  - Checks file existence
  - Validates ground truth consistency
  - Detects duplicate pairs
  - Reports class balance (genuine/impostor ratio)
  - Validates metadata JSON

## Usage Examples

### 1. Index a Dataset

```bash
# Index MMU Iris (small test dataset)
cd src
python -m bioverify index --config ../config/indexing/test_mmu.yaml

# Index all iris datasets
python -m bioverify index --config ../config/indexing/iris.yaml
```

### 2. Validate Generated CSV

```bash
python -m bioverify validate --csv test_mmu_iris.csv \
  --base-path "C:\Users\sebas\Documents\VUT_FIT_MIT\DP\PublicDataset" \
  --stats
```

### 3. Load Pairs in Python

```python
from bioverify.data.dataset import PairDataset

# Load all pairs
ds = PairDataset(
    'test_mmu_iris.csv',
    base_path='C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset',
    filter_modality='iris'  # Optional: filter by modality
)

print(f'Loaded {len(ds)} pairs')

# Get a single pair
pair = ds[0]
print(f'Image 1: {pair["image1_path"]}')
print(f'Same person: {pair["ground_truth"]}')

# Iterate in batches
for batch in ds.iterate_batches(batch_size=32, load_images=True):
    for pair in batch:
        img1 = pair['image1']  # numpy array
        img2 = pair['image2']
        # Run comparison methods...
```

## Configuration Files

Located in `config/indexing/`:

- **`default.yaml`** - Base configuration with all options documented
- **`iris.yaml`** - All iris datasets (CASIA,MMU, AMF)
- **`test_mmu.yaml`** - Small test config (45 identities, 450 images)

### Configuration Structure

```yaml
public_dataset_root: "C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset"
random_seed: 42

pair_generation:
  # Choose one of the following pair limiting strategies:
  genuine_per_identity: 5          # Generate ~5 pairs per identity
  # OR
  max_genuine_pairs: 1000          # Hard limit on total genuine pairs
  
  impostor_ratio: 1.0              # Ratio of impostor to genuine (1.0 = equal)
  max_impostor_pairs: 1000         # Hard limit on total impostor pairs (optional)

output:
  csv_path: "dataset_index.csv"
  relative_paths: true

datasets:
  - dataset_path: "Iris/002-MMU-Iris-Database"
    dataset_name: "MMU-Iris"
    modality: "iris"
  
  - dataset_path: "FingerVein/003-FV-USM"
    dataset_name: "FV-USM"
    modality: "fingervein"
    # For datasets supporting raw/roi split:
    image_type: "extracted"    # or "raw", or "both"
```

## CSV Manifest Format

Generated CSV files contain:

| Column | Description |
|--------|-------------|
| `pair_id` | Unique pair identifier |
| `image1_path` | Path to first image |
| `image2_path` | Path to second image |
| `modality` | Biometric type (iris, face, hand, fingervein) |
| `ground_truth` | true=genuine, false=impostor |
| `identity1` | Identity of first image |
| `identity2` | Identity of second image |
| `dataset_name` | Source dataset name |
| `metadata` | JSON with additional info (image_type, session, etc.) |


## Key Design Decisions

1. **Inference-Focused**: No train/val/test splits - designed for method comparison and evaluation
   - Users can manually create splits if needed for training

2. **Flexible Pair Limiting**: Two ways to control pair generation:
   - Per-identity limiting (`genuine_per_identity=5`) for consistent sampling across identities
   - Absolute limiting (`max_genuine_pairs=1000`) for fixed dataset sizes

3. **Dataset Variants**: Support for raw vs processed images
   - Some datasets (MMCBNU, FV-USM, THU-FVFDT) provide both raw and ROI/extracted versions
   - Use `image_type` parameter to select which variant to use

4. **Extensible Parsers**: Each dataset type has its own parser
   - Easy to add new datasets by implementing a parser class
   - Auto-detects parser from dataset name or modality

5. **Relative Paths**: CSV stores paths relative to `PublicDataset` root for portability

6. **Lazy Loading**: Images aren't loaded until needed, saves memory

7. **YAML Configs**: Easy to create experiment variations without code changes

## Supported Datasets

| Dataset | Modality | Identities | Notes |
|---------|----------|-----------|-------|
| CASIA-Iris-Thousand | Iris | 1000 | CASIAIrisParser |
| MMU-Iris | Iris | 46 | MMUIrisParser  |
| AMF-Iris | Iris | 54 | AMFIrisParser |
| Multi-PIE | Face | 249 | MultiPIEParser |
| 11k-Hands | Hand | 190 | HandsParser |
| Finger Vein | Finger Vein | 106 | FingerVeinParser |
| FV-USM | Finger Vein | 123 | FVUSMParser (raw/extracted variants) |
| THU-FVFDT | Dorsal/Finger Vein | 610 | THUFVFDTParser (raw/roi, train/test sessions) |
| MMCBNU-6000 | Finger Vein | 100 | MMCBNUParser (raw/roi variants) |
| EEMSC-DBM | Finger Vein | 60 | EEMSCDBMParser |

## Next Steps (Steps 3-10)

Now that the foundation is ready, we can proceed with:

3. **Base Matcher Interface** - Abstract class for all methods
4. **Concrete Matchers** - Port existing SIFT, SuperGlue, LoFTR, etc.
5. **Experiment Runner** - Batch processing across methods
6. **Evaluation Module** - Metrics, ROC curves, comparison
7. **Result Persistence** - JSON save/load
8. **CLI Expansion** - Run experiments, compare methods

The dataset indexing foundation ensures we can now:
- ✅ Generate consistent train/val/test splits
- ✅ Control pair generation (permutations, subsets)
- ✅ Load data efficiently with filtering
- ✅ Validate integrity automatically
- ✅ Work with multiple datasets simultaneously

## File Structure

```
src/
├── bioverify/
│   ├── __init__.py
│   ├── __main__.py
│   ├── results.py           # VerificationResult, ImageData, etc.
│   ├── cli/
│   │   ├── __init__.py
│   │   └── index.py         # CLI commands
│   ├── config/
│   │   ├── __init__.py
│   │   └──indexing/
│   │      ├── default.yaml     # Base configuration
│   │      ├── iris.yaml        # All iris datasets
│   │      └── test_mmu.yaml    # Small test config ✅
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parsers.py       # Dataset structure parsers
│   │   ├── pairs.py         # Pair generation
│   │   ├── indexer.py       # Main indexing orchestrator
│   │   ├── dataset.py       # CSV loader
│   │   └── validation.py    # CSV validation
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py  # Image utilities, masking
│
├── test_mmu_iris.csv        # Generated manifest (180 pairs)
└── test_loader.py           # Example usage script

```

---

**Status**: Foundation complete and tested! Ready to build matcher interfaces next. 🚀
