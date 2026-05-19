# BioVerify

Author: Sebastian Chupac (xchupa03)

Date: 19.5.2026

BioVerify is a framework for experimenting with homography based methods on biometric identification and verification tasks.
It includes dataset indexing, pair, and gallery+probe generation from datasets, homogrphy based methods implementations, verification matchers, closed-set identification,
mask precomputation, and evaluation utilities.

## Installation

These instructions assume your terminal is in the `src` directory of the repository.

1. Install Python 3.10.
2. Create and activate a virtual environment.

PowerShell

```powershell
python -m venv .venv
# If PowerShell blocks scripts, enable for this session only:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

```

or Command Prompt (cmd.exe):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install the dependencies. This also pulls the CUDA-enabled PyTorch wheel from the official PyTorch index, because BioVerify relies on CUDA and PyTorch.

```powershell
python -m pip install -r requirements.txt
```

4. If `torch.cuda.is_available()` still prints `False`, first confirm that this venv is active and that the NVIDIA driver is installed.

Important: this does not install the NVIDIA GPU driver or the full CUDA Toolkit on Windows. It only installs the CUDA runtime bundled with PyTorch. For `torch.cuda.is_available()` to return `True`, you need:

- an NVIDIA GPU
- the current NVIDIA driver installed on Windows
- a PyTorch wheel built for a CUDA version your driver supports

Verify CUDA from Python:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

5. Verify that the CLI starts.

```powershell
python -m bioverify --help
```

## Data Layout

The framework expects the bundled dataset root to live inside the package at:

`bioverify/PublicDataset`

It is recommended to set full path to this folder in configuration files for data indexing and experiments. Set it into public_dataset_root and base_path parameters.


The directory should contain the modality folders used by the project, such as:

- `Face`
- `Iris`
- `HandGeometry`
- `FingerVein`

Mask caches are stored under `bioverify/PublicDataset/_masks`.

The package also bundles model assets that the matchers use at runtime, including the face segmentation TFLite models and the matcher weights under `bioverify/matchers/*`.

## Usage

Run all commands from the `src` directory unless you change the paths in your config files.

Check the correct config file formats in bioverify/configs/... Set your actual paths to datasets.

### 1. Index datasets and generate pair/identification manifests

Index one or more datasets described in a YAML config and write a CSV manifest. The config can request either verification pair generation or identification manifest generation.

Flags:
- `--config, -c` (required): Path to YAML configuration file
- `--output, -o`: Override output CSV path from config
- `--validate, -v`: Validate the generated CSV after creation

Example:

```powershell
python -m bioverify index --config config/indexing/iris.yaml --validate
```

### 2. Validate a CSV manifest

Validate an existing CSV manifest and optionally print statistics.

Flags:
- `--csv` (required): Path to CSV manifest file
- `--base-path`: Base path for resolving relative image paths (default: PublicDataset)
- `--stats`: Print CSV statistics after validation

Example:

```powershell
python -m bioverify validate --csv data/pairs/iris_pairs.csv --stats
```

### 3. Print CSV statistics

Print summary statistics for a CSV manifest.

Flags:
- `--csv` (required): Path to CSV manifest file
- `--base-path`: Base path for resolving relative image paths (default: PublicDataset)

Example:

```powershell
python -m bioverify stats --csv data/pairs/iris_pairs.csv
```

### 4. Precompute masks

 Precompute segmentation masks (iris, face, hand) for all images in a modality and cache them under `bioverify/PublicDataset/_masks`.

 This is important if you want to use masks, because online mask generation during matching or experiments is not supported.

Flags:
- `--dataset-root`: Root folder containing modality subfolders (default: `bioverify/PublicDataset`)
- `--modality`: `iris` | `face` | `handGeometry`
- `--iris-exclude-pupil`: For iris masks, exclude the pupil (boolean)

Example:

```powershell
python -m bioverify.experiments.precompute_masks --dataset-root bioverify/PublicDataset --modality iris
```

### 5. Run a single-pair matcher (interactive / debug use) with optional vizualization

Run a single matcher on two images.

Flags:
- `--config, -c` (required): Matcher YAML config file
- `--matcher, -m`: Optional matcher name override (e.g., `sift`)
- `--image1` (required), `--image2` (required): Paths to the two images
- `--modality`: Modality hint (`iris`, `face`, `hand`, `fingervein`)
- `--ground-truth`: `same|different|true|false|1|0` (optional)
- `--full`: Print full `VerificationResult` object instead of summary
- `--viz`: Enable visualization (render, save, and display)
- `--viz-output`: Output path for rendered visualization image
- `--viz-mode`: `m`=matches (default), `k`=keypoints, `b`=both
- `--image-mode`: `o`=original image, `p`=processed matcher input (default)

Example:

```powershell
python -m bioverify match --config config/matching/loftr.yaml --image1 path\to\img1.png --image2 path\to\img2.png --viz --viz-mode b
```

### 6. Run a batch experiment (matching on a pairs CSV)

Execute a verification experiment defined in the config file. 

Flags:
- `--config, -c` (required): Path to experiment YAML config file
- `--verbose, -v`: Print verbose output including tracebacks

Example:

```powershell
python -m bioverify experiment --config config/experiments/exp_loftr_iris.yaml
```

### 7. Run a closed-set identification experiment

Run identification experiments (gallery + probes) using a config file.

Flags:
- `--config, -c` (required): Path to identification YAML config file
- `--verbose, -v`: Print verbose output including tracebacks

Example:

```powershell
python -m bioverify identification --config config/experiments/identification/identification_iris_casia_50_1_3_left.yaml
```

### 8. Evaluate verification experiment results

Verification experiments support addional evaluation. Use subcommands `threshold` (threshold sweep) and `compare` (matcher comparison).

Subcommand `threshold` flags:
- `--experiment, -e` (required): Experiment name (folder under `bioverify/results`)
- `--matcher, -m`: Optional matcher to analyze
- `--output, -o`: Output directory for plots and results

Example:

```powershell
python -m bioverify evaluate threshold --experiment my_experiment
```

Subcommand `compare` flags:
- `--experiment, -e` (required): Experiment name
- `--mode`: `eer` (default) | `far` | `frr` | `threshold`
- `--value`: Numeric value required when `--mode` is `far`, `frr`, or `threshold`
- `--output, -o`: Output directory for results

Example:

```powershell
python -m bioverify evaluate compare --experiment my_experiment --mode eer
```



## Dependencies

The full dependency list is stored in `requirements.txt`.

