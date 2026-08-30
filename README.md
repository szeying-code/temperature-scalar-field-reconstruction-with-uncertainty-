# Scalar Field Reconstruction Code

This repository contains the final code used for the dissertation experiments on spherical harmonic scalar-field reconstruction and adaptive geodesic conformal uncertainty quantification.

## Prior Work

The spherical harmonic scalar-field reconstruction builds on the work of **Will Thomas (2025)**. The reconstruction framework and results from this work provide the starting point for the uncertainty quantification developed in this repository.

Will Thomas's dissertation code is available here:

https://github.com/Wthomas443/Dissertation

This repository extends the scalar-field reconstruction with an adaptive geodesic conformal prediction framework for uncertainty quantification.

## Files

* `scalar_field_reconstruction.py`: main training, reconstruction, conformal prediction, robustness, and plotting workflow.
* `check.py`: supporting check script for selected city/data diagnostics.
* `requirements.txt`: Python packages required to run the scripts.

## Data

The raw temperature dataset is not included because it is large. Place the dataset in the repository root with this exact name:

```text
GlobalLandTemperaturesByCity.csv
```

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main script:

```bash
python scalar_field_reconstruction.py
```

The script writes generated models and figures to `models/` and `plots/`.

## Main Experimental Configuration

The primary experiment uses the following configuration:

* Spherical harmonic degree: `L = 26`
* Regularisation parameter: `λ = 0.0001`
* Sobolev penalty power: `p = 2`
* Number of nearest neighbours for local difficulty estimation: `k = 20`
* Number of cross-validation folds: `5`
* Conformal miscoverage level: `α = 0.1`
* Target coverage: `90%`

### Workflow

The main experimental pipeline consists of:

1. Data preprocessing
2. Hyperparameter selection using cross-validation
3. Spherical harmonic reconstruction
4. Local difficulty estimation using geodesic k-nearest neighbours
5. Adaptive geodesic conformal calibration
6. Evaluation on a held-out test set
7. Degree-robustness analysis

