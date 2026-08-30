# Scalar Field Reconstruction Code

This repository contains the final code used for the dissertation experiments on spherical harmonic scalar-field reconstruction and adaptive geodesic conformal uncertainty quantification.

## Files

- `scalar_field_reconstruction.py`: main training, reconstruction, conformal prediction, robustness, and plotting workflow.
- `check.py`: supporting check script for selected city/data diagnostics.
- `requirements.txt`: Python packages required to run the scripts.

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

