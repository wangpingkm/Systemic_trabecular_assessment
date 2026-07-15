# HRR-calibrated vertebral trabecular analysis

This repository contains the core analysis code used for selected-slice vertebral trabecular assessment across clinical CT, micro-PCCT, and VMI reconstructions.

Registration is treated as preprocessing. The code assumes that each selected slice has already been aligned to the reference grid, or that a registered image path is supplied in the manifest.

## Inputs

Use either a CSV manifest or a DICOM root directory.

Required manifest columns:

```text
case_id,slice_id,modality,image_path
```

Optional columns:

```text
mask_path,registered_image_path,resolution_um,pixel_spacing_row_mm,pixel_spacing_col_mm
```

The high-resolution reference label defaults to `HRR_52um`. Change it with `--hrr-label`.

## Run

Demo:

```bash
python core_trabecular_analysis.py --demo --outdir demo_outputs
```

Manifest input:

```bash
python core_trabecular_analysis.py \
  --manifest manifest.csv \
  --hrr-label MicroPCCT_052p53um \
  --outdir outputs
```

DICOM directory input:

```bash
python core_trabecular_analysis.py \
  --input-dir "/path/to/Selected slice DICOM" \
  --hrr-label MicroPCCT_052p53um \
  --outdir outputs
```

## Outputs

```text
per_image_metrics.csv
hrr_gaps.csv
modality_summary.csv
bridge_metric_screening.csv
analysis_config.json
```

The main metric families are primary trabecular morphometry, local structural descriptors, image-quality descriptors, skeleton/edge descriptors, texture features, and local bone-fraction grids.

