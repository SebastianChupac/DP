# BioVerify Framework - Complete System Documentation

**Status**: ✅ - All 7 matchers ported, unified framework operational

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Implemented Components](#implemented-components)
4. [Matcher Implementations](#matcher-implementations)
5. [Configuration System](#configuration-system)
6. [Usage Guide](#usage-guide)
7. [Results System](#results-system)
8. [Next Steps](#next-steps)

---

## System Overview

BioVerify is a unified framework for biometric verification method comparison across multiple modalities (iris, face, hand, fingervein). The framework provides:

- **7 Feature Matching Methods**: Traditional (SIFT, ORB) + Deep Learning (SuperGlue, LoFTR, ASpanFormer, SGMNet, DeepDetect)
- **Unified Configuration**: Consistent YAML-based config across all matchers
- **Dataset Indexing**: Automated pair generation from 10+ public datasets
- **Modality Support**: Iris, face, hand, finger vein with automatic masking
- **Result Standardization**: Lightweight VerificationResult for experiments, rich VisualizationResult for debugging

### Current Capabilities
✅ Dataset indexing and pair generation  
✅ Single-pair CLI matching  
✅ All 7 matchers operational  
✅ Unified preprocessing pipeline  
✅ Automatic ROI masking  
✅ Homography estimation with RANSAC  
✅ Consistent decision logic patterns  

### Pending Development
❌ Batch experiment runner (process all pairs in CSV)  
❌ Metrics aggregation (accuracy, EER, ROC)  
❌ Multi-matcher comparison tools  
❌ Parameter tuning infrastructure  

---


### Data Flow

```
Dataset → Indexer → Pairs CSV → Matcher → VerificationResult → Metrics
   ↓         ↓          ↓           ↓              ↓              ↓
Parsers   Config   Validation  Preprocessing  Decision Logic  Aggregation
                                   + Masking
```

---

## Implemented Components

### 1. Core Package Structure
**Location**: `src/bioverify/`

- **`matchers/`** - All matcher implementations + embedded models
- **`data/`** - Dataset handling (parsers, pairs, indexer, loader, validation)
- **`config/`** - Configuration management (indexing + matching)
- **`utils/`** - Preprocessing utilities (masking, resizing)
- **`cli/`** - Command-line interface
- **`results.py`** - Result data structures

### 2. Dataset Indexing System

**Parsers** ([parsers.py](bioverify/data/parsers.py)): Handles different dataset structures
- CASIA-Iris-Thousand
- MMU-Iris-Database
- AMF-Iris-Dataset
- Multi-PIE Face
- 11k Hands
- Finger Vein (generic structure)
- FV-USM Finger Vein Database (raw/extracted variants)
- THU-FVFDT (dorsal and finger vein with train/test sessions)
- MMCBNU-6000 Finger Vein (raw/ROI variants)
- EEMSC-DBM Finger Vein Database
- Automatically excludes `__MACOSX` metadata folders
- Extensible framework for adding new datasets

**Pair Generator** ([pairs.py](bioverify/data/pairs.py)): Creates verification pairs
- Genuine pairs (same identity)
- Impostor pairs (different identities)
- Per-identity or maximum-based pair limiting
- Optional matching constraints (e.g., same side for hand/vein)
- No train/val/test splitting (for inference-focused workflow)

**Dataset Indexer** ([indexer.py](bioverify/data/indexer.py)): Orchestrates the workflow
- Scans PublicDataset directory
- Parses structure
- Generates pairs
- Saves CSV manifests

**CSV Loader** ([dataset.py](bioverify/data/dataset.py)): Loads pairs for experiments
- Filtering by modality/dataset
- Lazy or eager image loading
- Batch iteration support

**Validation** ([validation.py](bioverify/data/validation.py)): Ensures data integrity
- Checks file existence
- Validates ground truth consistency
- Detects duplicate pairs
- Reports class balance (genuine/impostor ratio)
- Validates metadata JSON

### 3. Matcher System

**BaseMatcher** ([base.py](bioverify/matchers/base.py)): Abstract base class
- **`match()`**: Orchestration pipeline - load → preprocess → mask → match_impl → homography → result
- **Abstract methods**: 
  - `get_name()` - Returns matcher name
  - `_match_impl()` - Core matching logic
  - `_create_verification_result()` - Decision logic and result generation
- **Utility methods**:
  - `_load_image()` - Load and convert image
  - `_preprocess_image()` - Resize and color conversion (overridable)
  - `_get_or_compute_mask()` - ROI masking for iris/face/hand
  - `estimate_homography()` - RANSAC homography estimation
  - `compute_reprojection_error()` - Geometric error computation

**MatcherConfig** ([base.py](bioverify/matchers/base.py)): Configuration dataclass
- Standard params: `resize_width`, `resize_height`, `ransac_thresh`, `use_masking`, `device`
- `extra_params` dict for matcher-specific settings
- `from_dict()` class method for YAML loading

**Registry** ([registry.py](bioverify/matchers/registry.py)): Matcher factory
- `MATCHER_REGISTRY` - Maps names to classes
- `get_matcher_class(name)` - Case-insensitive lookup
- `create_matcher(name, config_dict)` - Factory function
- Registered: sift, orb, superglue, loftr, aspanformer, sgmnet, deepdetect

---

## Matcher Implementations

All 7 matchers follow this unified pattern:

### Common Structure

```python
class ExampleMatcher(BaseMatcher):
    def __init__(self, config: MatcherConfig):
        super().__init__(config)
        # Extract matcher-specific params from config.extra_params
        # Load model (for deep learning matchers)
    
    def get_name(self) -> str:
        return "MatcherName"
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # Resize based on config
        # Color conversion if needed
        return preprocessed_img
    
    def _match_impl(self, img1, img2, mask1, mask2):
        # Core matching logic
        # Returns (keypoints1, keypoints2, matches)
        return kpts1, kpts2, matches
    
    def _create_verification_result(self, ...):
        # Compute inlier_ratio, reprojection_error
        # Apply decision threshold
        # Return VerificationResult
    
    def _get_matcher_params(self) -> dict:
        # Return params dict for logging
        return {...}
```

### 1. SIFT ([sift.py](bioverify/matchers/sift.py))
**Type**: Traditional feature detector  
**Method**: Scale-invariant feature transform with FLANN matching  
**Decision Logic**: `inlier_ratio >= ratio_threshold`  
**Default Threshold**: 0.3  
**Config Params**: `nfeatures`, `nOctaveLayers`, `contrastThreshold`, `edgeThreshold`, `sigma`

### 2. ORB ([orb.py](bioverify/matchers/orb.py))
**Type**: Binary feature detector (CPU-friendly)  
**Method**: Oriented FAST and Rotated BRIEF  
**Decision Logic**: `inlier_ratio >= ratio_threshold`  
**Default Threshold**: 0.3  
**Config Params**: `nfeatures`, `scaleFactor`, `nlevels`, `edgeThreshold`, `patchSize`

### 3. SuperGlue ([superglue.py](bioverify/matchers/superglue.py))
**Type**: Deep learning graph neural network  
**Method**: SuperPoint keypoint detection + SuperGlue matching  
**Decision Logic**: `inlier_ratio >= ratio_threshold`  
**Default Threshold**: 0.4  
**Config Params**: `sinkhorn_iterations`, `match_threshold`, `weights` (indoor/outdoor)  
**Models**: Embedded in `matchers/superglue_models/`

### 4. LoFTR ([loftr.py](bioverify/matchers/loftr.py))
**Type**: Transformer-based dense matcher  
**Method**: Local Feature Transformer  
**Decision Logic**: `inlier_ratio >= ratio_threshold`  
**Default Threshold**: 0.3  
**Config Params**: `pretrained_weights` (indoor/outdoor)  
**Models**: Loaded from Kornia library

### 5. ASpanFormer ([aspanformer.py](bioverify/matchers/aspanformer.py))
**Type**: Attention-based dense matcher  
**Method**: Adaptive Span Transformer  
**Decision Logic**: `(inlier_ratio >= ratio_threshold) AND (reproj_error <= max_error)`  
**Default Thresholds**: ratio=0.45, max_error=5.0  
**Config Params**: `model_checkpoint`, `match_threshold`, `thr`  
**Models**: Embedded in `matchers/aspanformer_models/` (includes demo_utils)

### 6. SGMNet ([sgmnet.py](bioverify/matchers/sgmnet.py))
**Type**: Graph matching network  
**Method**: Semantic graph matching with learnable graph construction  
**Decision Logic**: `(inlier_ratio >= ratio_threshold) AND (reproj_error <= max_error)`  
**Default Thresholds**: ratio=0.3, max_error=5.0  
**Config Params**: `extractor_config`, `matcher_config` (embedded YAML)  
**Models**: Embedded in `matchers/sgmnet_models/` (extractor + matcher + components)

### 7. DeepDetect ([deepdetect.py](bioverify/matchers/deepdetect.py))
**Type**: CNN keypoint detection  
**Method**: ESPNet CNN for keypoint heatmap + SIFT descriptors  
**Decision Logic**: `(inlier_ratio >= ratio_threshold) AND (reproj_error <= max_error)`  
**Default Thresholds**: ratio=0.3, max_error=5.0  
**Config Params**: `model_path`, `min_distance`, `threshold_abs`, `num_points`  
**Models**: Embedded in `matchers/deepdetect_models/`  
**Note**: Applies ROI masking before CNN prediction, processes at 320x320 resolution

### Decision Logic Patterns

**Simple Threshold** (SIFT, ORB, SuperGlue, LoFTR):
```python
is_match = inlier_ratio >= ratio_threshold
```

**Dual Threshold** (ASpanFormer, SGMNet, DeepDetect):
```python
is_match = (inlier_ratio >= ratio_threshold) AND (reproj_error <= max_reprojection_error)
```

---

## Configuration System

### Configuration Structure

All matchers use flat YAML configuration:

```yaml
matcher:
  name: "loftr"
  config:
    # Standard parameters (all matchers)
    resize_width: 640
    resize_height: 480
    ransac_thresh: 3.0
    use_masking: true
    device: "cuda"
    
    # Matcher-specific parameters
    pretrained_weights: "indoor"
    ratio_threshold: 0.3
```

### Configuration Files

**Location**: `config/matching/`

- [sift_config.yaml](bioverify/config/matching/sift_config.yaml)
- [orb_config.yaml](bioverify/config/matching/orb_config.yaml)
- [superglue_config.yaml](bioverify/config/matching/superglue_config.yaml)
- [loftr_config.yaml](bioverify/config/matching/loftr_config.yaml)
- [aspanformer_config.yaml](bioverify/config/matching/aspanformer_config.yaml)
- [sgmnet_config.yaml](bioverify/config/matching/sgmnet_config.yaml)
- [deepdetect_config.yaml](bioverify/config/matching/deepdetect_config.yaml)

### Dataset Indexing Configuration

**Location**: `config/indexing/`

- **`default.yaml`** - Base configuration with all options documented
- **`iris.yaml`** - All iris datasets (CASIA, MMU, AMF)
- **`test_mmu.yaml`** - Small test config (45 identities, 450 images)

**Example**:
```yaml
public_dataset_root: "C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset"
random_seed: 42

pair_generation:
  genuine_per_identity: 5      # ~5 pairs per identity
  impostor_ratio: 1.0          # Equal genuine/impostor
  max_impostor_pairs: 1000     # Hard limit

output:
  csv_path: "dataset_index.csv"
  relative_paths: true

datasets:
  - dataset_path: "Iris/002-MMU-Iris-Database"
    dataset_name: "MMU-Iris"
    modality: "iris"
```

---

## Usage Guide

### 1. Dataset Indexing

Generate pairs CSV from datasets:

```bash
cd src

# Index small test dataset
python -m bioverify index --config ../config/indexing/test_mmu.yaml

# Index all iris datasets
python -m bioverify index --config ../config/indexing/iris.yaml
```

### 2. Validate Pairs CSV

Check integrity of generated pairs:

```bash
python -m bioverify validate --csv test_mmu_iris.csv \
  --base-path "C:\Users\sebas\Documents\VUT_FIT_MIT\DP\PublicDataset" \
  --stats
```

### 3. View Dataset Statistics

```bash
python -m bioverify stats --csv data/pairs/iris_pairs.csv
```

### 4. Single-Pair Matching (CLI)

```bash
# Match with SIFT
python -m bioverify match \
  --image1 "path/to/image1.jpg" \
  --image2 "path/to/image2.jpg" \
  --matcher sift \
  --config config/matching/sift_config.yaml \
  --visualize

# Match with LoFTR
python -m bioverify match \
  --image1 "path/to/image1.jpg" \
  --image2 "path/to/image2.jpg" \
  --matcher loftr \
  --config config/matching/loftr_config.yaml
```

### 5. Programmatic Usage

#### Load and Process Pairs

```python
from bioverify.data.dataset import PairDataset

# Load pairs from CSV
ds = PairDataset(
    'data/pairs/iris_pairs.csv',
    base_path='C:/Users/sebas/Documents/VUT_FIT_MIT/DP/PublicDataset',
    filter_modality='iris'
)

print(f'Loaded {len(ds)} pairs')

# Iterate in batches
for batch in ds.iterate_batches(batch_size=32, load_images=False):
    for pair in batch:
        print(f"Pair {pair['pair_id']}: {pair['ground_truth']}")
```

#### Use a Matcher

```python
from bioverify.matchers.registry import create_matcher
from bioverify.matchers.base import MatcherConfig
import yaml

# Load config
with open('config/matching/loftr_config.yaml') as f:
    config = yaml.safe_load(f)

# Create matcher
matcher = create_matcher(
    config['matcher']['name'],
    config['matcher']['config']
)

# Match images
result = matcher.match(
    image1_path='path/to/img1.jpg',
    image2_path='path/to/img2.jpg',
    ground_truth=True  # optional
)

# Access results
print(f"Predicted same person: {result.is_same_person_pred}")
print(f"Inlier ratio: {result.inlier_ratio:.3f}")
print(f"Num matches: {result.num_matches}")
```

---

## Results System

### VerificationResult ([results.py](bioverify/results.py))

**Purpose**: Lightweight result storage for experiments (no image data stored)

**Fields**:
```python
@dataclass
class VerificationResult:
    method_name: str                    # Matcher name
    is_same_person_pred: bool          # Prediction: same/different
    verification_confidence: float      # Confidence score (0-1)
    ground_truth: Optional[bool]       # True label (if available)
    
    # Matching metrics
    num_matches: int                   # Total feature matches
    num_inliers: int                   # Geometric consistent matches
    inlier_ratio: float                # num_inliers / num_matches
    reprojection_error: float          # Average geometric error
    
    # Auto-computed
    is_correct: Optional[bool]         # prediction == ground_truth
```

### VisualizationResult ([results.py](bioverify/results.py))

**Purpose**: Rich result for debugging and visualization (stores images, keypoints, matches)

**Fields**: All VerificationResult fields plus:
- `image1`, `image2` - Original images
- `keypoints1`, `keypoints2` - Detected keypoints
- `matches` - Match pairs
- `homography` - Estimated transformation matrix
- `matcher_params` - Config parameters used

### CSV Manifest Format

Generated pairs CSV structure:

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
| `metadata` | JSON with additional info (image_type, session, etc.) |---

## Supported Datasets

| Dataset | Modality | Identities | Parser | Notes |
|---------|----------|-----------|--------|-------|
| CASIA-Iris-Thousand | Iris | 1000 | CASIAIrisParser | Large-scale iris dataset |
| MMU-Iris | Iris | 46 | MMUIrisParser | Small test dataset |
| AMF-Iris | Iris | 54 | AMFIrisParser | Iris with metadata |
| Multi-PIE | Face | 249 | MultiPIEParser | Multi-pose face dataset |
| 11k-Hands | Hand | 190 | HandsParser | Dorsal hand images |
| Finger Vein | Finger Vein | 106 | FingerVeinParser | Generic structure |
| FV-USM | Finger Vein | 123 | FVUSMParser | Raw/extracted variants |
| THU-FVFDT | Dorsal/Finger Vein | 610 | THUFVFDTParser | Raw/ROI, train/test sessions |
| MMCBNU-6000 | Finger Vein | 100 | MMCBNUParser | Raw/ROI variants |
| EEMSC-DBM | Finger Vein | 60 | EEMSCDBMParser | Finger vein database |

---

## Key Design Decisions

### 1. Inference-Focused Workflow
- No train/val/test splits by default - designed for method comparison
- Users can manually create splits if needed for training experiments

### 2. Flexible Pair Limiting
Two strategies for controlling pair generation:
- **Per-identity**: `genuine_per_identity=5` ensures consistent sampling across identities
- **Absolute**: `max_genuine_pairs=1000` limits total dataset size

### 3. Dataset Variant Support
Some datasets provide multiple image types:
- **Raw vs Processed**: MMCBNU, FV-USM, THU-FVFDT offer both raw and ROI/extracted versions
- **Config Control**: Use `image_type` parameter to select variant

### 4. Extensible Parser System
- Each dataset type has dedicated parser class
- Easy to add new datasets by implementing parser interface
- Auto-detection from dataset name or modality

### 5. Path Portability
- CSV stores relative paths from `PublicDataset` root
- Works across different machines/environments

### 6. Memory Efficiency
- **Lazy Loading**: Images loaded only when needed
- **Lightweight Results**: VerificationResult excludes image data
- **Batch Processing**: Iterate over large datasets without loading all into memory

### 7. Unified Preprocessing
- Framework handles resize + masking consistently
- Matchers focus on core algorithm logic
- Ensures fair comparison across methods

### 8. Self-Contained Models
- All deep learning models embedded in framework directories
- No external config file dependencies
- Isolated from original implementation repositories

---

## Implementation Details

### Masking Pipeline

**Supported Modalities**: Iris, Face, Hand  
**Method**: Automatic ROI mask generation using segmentation models  
**Location**: [bioverify/utils/preprocessing.py](bioverify/utils/preprocessing.py)

**Usage**:
```python
mask = matcher._get_or_compute_mask(image_path, modality='iris')
```

**Iris Masking**:
- Uses MediaPipe Iris solution
- Segments iris region from eye image
- Binary mask isolating circular iris region

**Face Masking**:
- TFLite segmentation model (`face_segmentation/`)
- Removes background, isolates face region

**Hand Masking**:
- TFLite hand segmentation model
- Isolates dorsal hand area

**Finger Vein**: No masking (already ROI-cropped in datasets)

### Homography Estimation

**Method**: RANSAC-based homography from matched keypoints  
**Parameters**: `ransac_thresh` (default: 3.0 pixels)  
**Location**: [BaseMatcher.estimate_homography()](bioverify/matchers/base.py#L150)

**Inlier Computation**:
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacThreshold)
num_inliers = mask.sum()
inlier_ratio = num_inliers / len(matches)
```

### Reprojection Error

**Definition**: Average geometric error of inlier matches after homography transformation  
**Location**: [BaseMatcher.compute_reprojection_error()](bioverify/matchers/base.py#L170)

**Computation**:
```python
# Transform keypoints1 to image2 space
projected = cv2.perspectiveTransform(kpts1, homography)
errors = np.linalg.norm(projected - kpts2, axis=1)
reproj_error = errors.mean()
```

---

## Current Status

### Completed ✅

1. **Core Framework**
   - BaseMatcher abstract class with unified pipeline
   - MatcherConfig dataclass for YAML loading
   - Registry system for matcher factory
   - VerificationResult + VisualizationResult dataclasses

2. **All 7 Matchers Ported**
   - SIFT (traditional)
   - ORB (binary features)
   - SuperGlue (graph neural network)
   - LoFTR (transformer-based)
   - ASpanFormer (attention-based)
   - SGMNet (semantic graph matching)
   - DeepDetect (CNN keypoints)

3. **Configuration System**
   - Flat YAML structure across all matchers
   - Standardized parameters (resize, masking, RANSAC, device)
   - Matcher-specific params via extra_params
   - 7 config files in `config/matching/`

4. **Dataset Infrastructure**
   - 10 dataset parsers implemented
   - Pair generation with genuine/impostor balancing
   - CSV manifest format with metadata
   - Validation and statistics tools
   - 10 pre-generated pairs CSV files in `data/pairs/`

5. **CLI Tools**
   - `index` - Generate pairs from datasets
   - `validate` - Check CSV integrity
   - `stats` - Display pair statistics
   - `match` - Single-pair matching with visualization

6. **Preprocessing Pipeline**
   - Automatic ROI masking (iris, face, hand)
   - Consistent image resizing
   - Homography estimation
   - Reprojection error computation

### Validated ✅

- ✅ All matchers instantiate successfully from config
- ✅ Config format consistent across all matchers
- ✅ Decision logic implemented per matcher specification
- ✅ 10 pairs CSV files generated and validated
- ✅ Single-pair matching works via CLI
- ✅ All matchers registered in factory
- ✅ Masking pipeline operational for all modalities

### Pending ❌

1. **Batch Experiment Runner**
   - Process all pairs in CSV file
   - Save results to JSON/CSV
   - Progress tracking and resumption
   - Multi-matcher parallel processing

2. **Metrics Module**
   - Accuracy, precision, recall, F1
   - Equal Error Rate (EER)
   - ROC curves and AUC
   - Confusion matrices
   - Per-dataset and per-modality aggregation

3. **Comparison Tools**
   - Side-by-side matcher performance
   - Statistical significance testing
   - Performance visualization
   - Result export for papers/presentations

4. **Parameter Tuning**
   - Grid search over thresholds
   - Cross-validation infrastructure
   - Optimization for each modality
   - Best parameter reporting

5. **Visualization**
   - Match visualization for debugging
   - ROC curve plotting
   - Error analysis tools
   - Dataset distribution visualizations

---

## Next Steps

### Immediate Priorities

1. **Batch Experiment Runner** (Priority 1)
   - Add CLI command: `python -m bioverify experiment --pairs data/pairs/iris_pairs.csv --matcher loftr`
   - Process all pairs in CSV file
   - Save VerificationResult for each pair to JSON
   - Generate summary statistics

2. **Metrics Module** (Priority 2)
   - Implement `bioverify/metrics/evaluation.py`
   - Compute accuracy, EER, ROC from VerificationResult list
   - Export results to CSV/JSON for analysis

3. **Multi-Matcher Comparison** (Priority 3)
   - Run multiple matchers on same dataset
   - Generate comparison tables
   - Statistical significance tests

### Future Enhancements

4. **Optimization Infrastructure**
   - Parameter grid search
   - Cross-validation support
   - Per-modality threshold tuning

5. **Advanced Visualization**
   - Interactive match visualizations
   - ROC curve plotting
   - Confusion matrix heatmaps

6. **Performance Optimization**
   - Cache preprocessed images
   - Parallel batch processing
   - GPU memory optimization

7. **Cross-Dataset Validation**
   - Train on one dataset, test on another
   - Domain adaptation experiments
   - Transfer learning support

---

## Known Issues and Limitations

### Method-Specific Notes

**DeepDetect**:
- Processes at 320x320 resolution (framework standardization)
- Original implementation used high-res for SIFT matching
- May affect keypoint detection quality compared to original

**LoFTR**:
- Requires sequential match indices after filtering
- May produce fewer matches than other methods on low-texture images

**ASpanFormer**:
- Embedded demo_utils may differ from latest upstream
- Indoor model checkpoint required for optimal performance

**SGMNet**:
- Complex config with extractor + matcher pipeline
- Longer inference time than simpler methods

### Framework Limitations

1. **No Batch Processing**: Currently only single-pair matching via CLI
2. **Manual Threshold Tuning**: Default thresholds not optimized per dataset
3. **Limited Metrics**: No aggregated performance metrics yet
4. **No Caching**: Preprocessed images recomputed on each run
5. **Sequential Processing**: No parallel matching for large experiments

### Dataset Considerations

- **Imbalanced Pairs**: Some datasets have more impostor than genuine pairs
- **Image Quality**: Varies significantly across datasets
- **ROI Consistency**: Some datasets provide pre-cropped ROI, others don't
- **Metadata Completeness**: Not all datasets have session/capture info

---

## File Structure Summary

```
src/bioverify/
├── __init__.py
├── __main__.py
├── FOUNDATION_README.md         # This file
├── results.py                   # VerificationResult, VisualizationResult
│
├── matchers/
│   ├── __init__.py
│   ├── base.py                  # BaseMatcher, MatcherConfig
│   ├── registry.py              # Matcher factory
│   ├── sift.py                  # SIFT implementation
│   ├── orb.py                   # ORB implementation
│   ├── superglue.py             # SuperGlue implementation
│   ├── loftr.py                 # LoFTR implementation
│   ├── aspanformer.py           # ASpanFormer implementation
│   ├── sgmnet.py                # SGMNet implementation
│   ├── deepdetect.py            # DeepDetect implementation
│   ├── superglue_models/        # Embedded SuperPoint + SuperGlue
│   ├── aspanformer_models/      # Embedded ASpanFormer + demo_utils
│   ├── sgmnet_models/           # Embedded SGMNet components
│   └── deepdetect_models/       # Embedded ESPNet CNN
│
├── data/
│   ├── __init__.py
│   ├── parsers.py               # Dataset structure parsers
│   ├── pairs.py                 # Pair generation + PairsValidator
│   ├── indexer.py               # Dataset indexing orchestrator
│   ├── dataset.py               # PairDataset CSV loader
│   └── validation.py            # CSV validation tools
│
├── config/
│   ├── indexing/
│   │   ├── default.yaml         # Base indexing config
│   │   ├── iris.yaml            # All iris datasets
│   │   └── test_mmu.yaml        # Small test config
│   └── matching/
│       ├── sift_config.yaml
│       ├── orb_config.yaml
│       ├── superglue_config.yaml
│       ├── loftr_config.yaml
│       ├── aspanformer_config.yaml
│       ├── sgmnet_config.yaml
│       └── deepdetect_config.yaml
│
├── cli/
│   ├── __init__.py
│   └── index.py                 # CLI: index, validate, stats, match
│
└── utils/
    ├── __init__.py
    └── preprocessing.py         # Masking, resizing utilities

data/pairs/                      # Generated pairs CSV files (10 files)
├── face_pairs.csv
├── fingervein_pairs.csv
├── hand_pairs.csv
├── iris_pairs.csv
└── ... (6 more)
```

---

## Contribution Guidelines

### Adding a New Matcher

1. **Create matcher file**: `bioverify/matchers/newmatcher.py`
2. **Inherit from BaseMatcher**: Implement required abstract methods
3. **Add config file**: `config/matching/newmatcher_config.yaml`
4. **Register matcher**: Add to `MATCHER_REGISTRY` in `registry.py`
5. **Test**: Verify instantiation and single-pair matching

**Required Methods**:
```python
def get_name(self) -> str
def _match_impl(self, img1, img2, mask1, mask2)
def _create_verification_result(self, ...)
```

### Adding a New Dataset Parser

1. **Create parser class** in `bioverify/data/parsers.py`
2. **Implement methods**: `parse()`, `get_image_paths()`, `get_identity()`
3. **Register parser**: Add to parser selection logic in `indexer.py`
4. **Create config**: Add dataset entry to indexing config YAML
5. **Test**: Run indexing and validate CSV output

