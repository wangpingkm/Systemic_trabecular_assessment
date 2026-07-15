#!/usr/bin/env python3
"""
Core selected-slice trabecular analysis for HRR-calibrated CT comparison.

Input manifest columns:
case_id,slice_id,modality,image_path

Optional columns:
mask_path,registered_image_path,reader_score,energy_kev,resolution_um
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import exposure, filters, io, measure, morphology, util
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

try:
    import pydicom
except ImportError:
    pydicom = None


PRIMARY_METRICS = ["bone_area_fraction", "tb_th_mm", "tb_sp_mm", "tb_n_per_mm", "edge_density"]


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required for DICOM input")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = io.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def read_mask(path: Optional[str | Path], shape: Tuple[int, int]) -> np.ndarray:
    if path is None or (isinstance(path, float) and math.isnan(path)):
        return np.ones(shape, dtype=bool)
    mask = read_image(path)
    if mask.shape != shape:
        mask = ndi.zoom(mask, np.array(shape) / np.array(mask.shape), order=0)
    return mask > 0


def maybe_registered_image(row: pd.Series) -> np.ndarray:
    path = row.get("registered_image_path")
    if isinstance(path, str) and path.strip():
        return read_image(path)
    return read_image(row["image_path"])


def registration_note() -> str:
    return (
        "Registration is treated as preprocessing. The analysis expects each image "
        "to be already aligned to the selected-slice reference grid; provide "
        "`registered_image_path` in the manifest when available."
    )


def robust_normalize(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = img[mask & np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def threshold_image(img01: np.ndarray, mask: np.ndarray, method: str, fixed: Optional[float]) -> np.ndarray:
    vals = img01[mask & np.isfinite(img01)]
    if vals.size == 0:
        return np.zeros_like(mask, dtype=bool)
    if method == "fixed":
        thr = 0.5 if fixed is None else fixed
    elif method == "yen":
        thr = filters.threshold_yen(vals)
    elif method == "li":
        thr = filters.threshold_li(vals)
    else:
        thr = filters.threshold_otsu(vals)
    return (img01 >= thr) & mask


def distance_mean(binary: np.ndarray, spacing_mm: float) -> float:
    if not np.any(binary):
        return np.nan
    dist = ndi.distance_transform_edt(binary) * spacing_mm
    return float(2.0 * np.mean(dist[binary]))


def skeleton_metrics(bone: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    clean = morphology.remove_small_objects(bone.astype(bool), min_size=4)
    skel = morphology.skeletonize(clean)
    k = np.ones((3, 3), dtype=int)
    neighbors = ndi.convolve(skel.astype(int), k, mode="constant", cval=0) - skel.astype(int)
    endpoints = skel & (neighbors == 1)
    branches = skel & (neighbors >= 3)
    labels = measure.label(skel, connectivity=2)
    props = measure.regionprops(labels)
    lengths = [p.area for p in props]
    roi_area = float(np.count_nonzero(roi))
    return {
        "skeleton_pixel_count": float(np.count_nonzero(skel)),
        "skeleton_density": float(np.count_nonzero(skel) / roi_area) if roi_area else np.nan,
        "skeleton_endpoint_count": float(np.count_nonzero(endpoints)),
        "skeleton_branch_count": float(np.count_nonzero(branches)),
        "skeleton_component_count": float(len(lengths)),
        "skeleton_mean_component_length_px": float(np.mean(lengths)) if lengths else np.nan,
    }


def edge_metrics(img01: np.ndarray, bone: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    gx = ndi.sobel(img01, axis=1)
    gy = ndi.sobel(img01, axis=0)
    grad = np.hypot(gx, gy)
    lap = np.abs(ndi.laplace(img01))
    edge = morphology.binary_dilation(bone) ^ morphology.binary_erosion(bone)
    denom = float(np.count_nonzero(roi))
    return {
        "edge_density": float(np.count_nonzero(edge & roi) / denom) if denom else np.nan,
        "gradient_sharpness": masked_mean(grad, edge & roi),
        "laplacian_sharpness": masked_mean(lap, edge & roi),
        "gradient_mean_roi": masked_mean(grad, roi),
        "laplacian_mean_roi": masked_mean(lap, roi),
    }


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask & np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size else np.nan


def masked_std(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask & np.isfinite(arr)]
    return float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan


def entropy_metric(img01: np.ndarray, roi: np.ndarray) -> float:
    vals = img01[roi & np.isfinite(img01)]
    if vals.size == 0:
        return np.nan
    hist, _ = np.histogram(vals, bins=64, range=(0, 1), density=False)
    p = hist.astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def glcm_metrics(img01: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    if not np.any(roi):
        return {f"glcm_{k}": np.nan for k in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"]}
    crop = crop_to_mask(img01, roi)
    crop_roi = crop_to_mask(roi.astype(np.uint8), roi).astype(bool)
    fill = masked_mean(crop, crop_roi)
    patch = crop.copy()
    patch[~crop_roi] = fill
    patch = util.img_as_ubyte(np.clip(patch, 0, 1))
    glcm = graycomatrix(patch, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True, normed=True)
    out = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        name = "asm" if prop == "ASM" else prop
        out[f"glcm_{name}"] = float(np.nanmean(graycoprops(glcm, prop)))
    return out


def crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return arr
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def local_contrast_metrics(img01: np.ndarray, bone: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    marrow = roi & ~bone
    bone_vals = img01[bone]
    marrow_vals = img01[marrow]
    if bone_vals.size == 0 or marrow_vals.size == 0:
        contrast = np.nan
        cnr = np.nan
    else:
        contrast = float(np.mean(bone_vals) - np.mean(marrow_vals))
        noise = np.sqrt(np.var(bone_vals) + np.var(marrow_vals))
        cnr = float(abs(contrast) / noise) if noise > 0 else np.nan
    local = ndi.uniform_filter(img01, size=9)
    local_sq = ndi.uniform_filter(img01 * img01, size=9)
    local_sd = np.sqrt(np.maximum(local_sq - local * local, 0))
    return {
        "bone_intensity_mean": masked_mean(img01, bone),
        "marrow_intensity_mean": masked_mean(img01, marrow),
        "local_bone_contrast_mean": contrast,
        "local_bone_contrast_sd": masked_std(local_sd, roi),
        "intensity_mean": masked_mean(img01, roi),
        "intensity_sd": masked_std(img01, roi),
        "estimated_noise": float(np.median(np.abs(img01[roi] - ndi.median_filter(img01, size=3)[roi])) / 0.6745) if np.any(roi) else np.nan,
        "snr": snr(img01, roi),
        "cnr": cnr,
        "rms_contrast": rms_contrast(img01, roi),
        "entropy": entropy_metric(img01, roi),
    }


def snr(img01: np.ndarray, roi: np.ndarray) -> float:
    vals = img01[roi & np.isfinite(img01)]
    if vals.size < 2:
        return np.nan
    sd = np.std(vals, ddof=1)
    return float(np.mean(vals) / sd) if sd > 0 else np.nan


def rms_contrast(img01: np.ndarray, roi: np.ndarray) -> float:
    vals = img01[roi & np.isfinite(img01)]
    if vals.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((vals - np.mean(vals)) ** 2)))


def grid_metrics(bone: np.ndarray, roi: np.ndarray, grids: Iterable[int] = (4, 8)) -> Dict[str, float]:
    out = {}
    h, w = roi.shape
    for g in grids:
        vals = []
        for y0 in np.linspace(0, h, g + 1, dtype=int)[:-1]:
            y1 = min(h, y0 + math.ceil(h / g))
            for x0 in np.linspace(0, w, g + 1, dtype=int)[:-1]:
                x1 = min(w, x0 + math.ceil(w / g))
                r = roi[y0:y1, x0:x1]
                if np.count_nonzero(r) == 0:
                    continue
                b = bone[y0:y1, x0:x1]
                vals.append(np.count_nonzero(b & r) / np.count_nonzero(r))
        vals = np.asarray(vals, dtype=float)
        out[f"grid_{g}x{g}_bone_fraction_mean"] = float(np.mean(vals)) if vals.size else np.nan
        out[f"grid_{g}x{g}_bone_fraction_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
        out[f"grid_{g}x{g}_bone_fraction_cv"] = float(np.std(vals, ddof=1) / np.mean(vals)) if vals.size > 1 and np.mean(vals) else np.nan
    return out


def trabecular_morphometry(bone: np.ndarray, roi: np.ndarray, spacing_mm: float) -> Dict[str, float]:
    bone = bone & roi
    marrow = roi & ~bone
    roi_area = float(np.count_nonzero(roi))
    baf = float(np.count_nonzero(bone) / roi_area) if roi_area else np.nan
    tb_th = distance_mean(bone, spacing_mm)
    tb_sp = distance_mean(marrow, spacing_mm)
    tb_n = float(baf / tb_th) if tb_th and not np.isnan(tb_th) else np.nan
    return {
        "bone_area_fraction": baf,
        "tb_th_mm": tb_th,
        "tb_sp_mm": tb_sp,
        "tb_n_per_mm": tb_n,
        "bone_area_mm2": float(np.count_nonzero(bone) * spacing_mm * spacing_mm),
        "roi_area_mm2": float(roi_area * spacing_mm * spacing_mm),
        "connected_component_count": float(measure.label(bone, connectivity=2).max()),
    }


def compute_metrics(img: np.ndarray, roi: np.ndarray, spacing_mm: float, threshold: str, fixed_threshold: Optional[float]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    img01 = robust_normalize(img, roi)
    bone = threshold_image(img01, roi, threshold, fixed_threshold)
    metrics = {}
    metrics.update(trabecular_morphometry(bone, roi, spacing_mm))
    metrics.update(edge_metrics(img01, bone, roi))
    metrics.update(local_contrast_metrics(img01, bone, roi))
    metrics.update(skeleton_metrics(bone, roi))
    metrics.update(grid_metrics(bone, roi))
    metrics.update(glcm_metrics(img01, roi))
    return metrics, img01, bone


def spacing_from_row(row: pd.Series, default_um: float) -> float:
    row_spacing = row.get("pixel_spacing_row_mm")
    col_spacing = row.get("pixel_spacing_col_mm")
    if pd.notna(row_spacing) and pd.notna(col_spacing):
        return float(row_spacing + col_spacing) / 2.0
    value = row.get("resolution_um", default_um)
    if pd.isna(value):
        value = default_um
    return float(value) / 1000.0


def dicom_spacing(path: str | Path) -> Tuple[float, float]:
    if pydicom is None:
        return np.nan, np.nan
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        ps = getattr(ds, "PixelSpacing", [np.nan, np.nan])
        return float(ps[0]), float(ps[1])
    except Exception:
        return np.nan, np.nan


def parse_v9_filename(path: Path) -> Dict[str, object]:
    parts = path.name.replace(".dcm", "").split("__")
    if len(parts) >= 4:
        case, modality, plane, detail = parts[:4]
    else:
        case = path.parents[1].name if len(path.parents) > 1 else "case"
        plane = path.parent.name
        modality = path.stem
        detail = ""
    m = re.search(r"axis(\d+)_slice(\d+)of(\d+)", detail)
    return {
        "case_id": case,
        "slice_id": plane,
        "modality": modality,
        "axis": int(m.group(1)) if m else np.nan,
        "slice_index": int(m.group(2)) if m else np.nan,
        "slice_total": int(m.group(3)) if m else np.nan,
    }


def build_manifest_from_dicom_dir(input_dir: Path, out_csv: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("**/*.dcm")):
        row = parse_v9_filename(path)
        row["image_path"] = str(path)
        row["mask_path"] = ""
        row["pixel_spacing_row_mm"], row["pixel_spacing_col_mm"] = dicom_spacing(path)
        rows.append(row)
    if not rows:
        raise ValueError(f"No DICOM files found under {input_dir}")
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


def build_common_roi(group: pd.DataFrame, images: Dict[int, np.ndarray]) -> np.ndarray:
    common = None
    for idx, row in group.iterrows():
        mask = read_mask(row.get("mask_path"), images[idx].shape)
        valid = mask & np.isfinite(images[idx])
        common = valid if common is None else (common & valid)
    return common.astype(bool)


def fidelity_metrics(img01: np.ndarray, ref01: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    if not np.any(roi):
        return {"mae_vs_hrr": np.nan, "mse_vs_hrr": np.nan, "psnr_vs_hrr": np.nan, "ssim_vs_hrr": np.nan}
    a = img01.copy()
    b = ref01.copy()
    fill_a = masked_mean(a, roi)
    fill_b = masked_mean(b, roi)
    a[~roi] = fill_a
    b[~roi] = fill_b
    mae = float(np.mean(np.abs(a[roi] - b[roi])))
    mse = float(mean_squared_error(b[roi], a[roi]))
    if mse == 0:
        return {"mae_vs_hrr": mae, "mse_vs_hrr": mse, "psnr_vs_hrr": float("inf"), "ssim_vs_hrr": 1.0}
    try:
        psnr = float(peak_signal_noise_ratio(b, a, data_range=1.0))
        ssim = float(structural_similarity(b, a, data_range=1.0, gaussian_weights=True))
    except ValueError:
        psnr, ssim = np.nan, np.nan
    return {"mae_vs_hrr": mae, "mse_vs_hrr": mse, "psnr_vs_hrr": psnr, "ssim_vs_hrr": ssim}


def analyze_manifest(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.input_dir is not None:
        manifest = build_manifest_from_dicom_dir(args.input_dir, args.outdir / "generated_manifest.csv")
    else:
        manifest = pd.read_csv(args.manifest)
    required = {"case_id", "slice_id", "modality", "image_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    rows = []
    normalized_cache: Dict[int, np.ndarray] = {}
    metric_cache: Dict[int, Dict[str, float]] = {}

    for (case_id, slice_id), group in manifest.groupby(["case_id", "slice_id"], sort=False):
        images = {idx: maybe_registered_image(row) for idx, row in group.iterrows()}
        shape = next(iter(images.values())).shape
        for idx, img in images.items():
            if img.shape != shape:
                images[idx] = ndi.zoom(img, np.array(shape) / np.array(img.shape), order=1)
        common_roi = build_common_roi(group, images)

        hrr_rows = group[group["modality"].astype(str) == args.hrr_label]
        if hrr_rows.empty:
            raise ValueError(f"No HRR row for case={case_id}, slice={slice_id}")
        hrr_idx = hrr_rows.index[0]

        for idx, row in group.iterrows():
            metrics, img01, _ = compute_metrics(
                images[idx],
                common_roi,
                spacing_from_row(row, args.default_resolution_um),
                args.threshold,
                args.fixed_threshold,
            )
            normalized_cache[idx] = img01
            metric_cache[idx] = metrics
            rec = row.to_dict()
            rec.update(metrics)
            rec["common_roi_fraction"] = float(np.count_nonzero(common_roi) / common_roi.size)
            rows.append(rec)

        ref01 = normalized_cache[hrr_idx]
        for idx, row in group.iterrows():
            rows[-len(group) + list(group.index).index(idx)].update(fidelity_metrics(normalized_cache[idx], ref01, common_roi))

    per_image = pd.DataFrame(rows)
    gaps = compute_hrr_gaps(per_image, args.hrr_label)
    summary = summarize_modalities(per_image, gaps, args.hrr_label)
    bridges = screen_bridge_metrics(per_image, gaps, args.hrr_label)
    return per_image, gaps, summary, bridges


def compute_hrr_gaps(per_image: pd.DataFrame, hrr_label: str) -> pd.DataFrame:
    records = []
    id_cols = ["case_id", "slice_id"]
    hrr = per_image[per_image["modality"].astype(str) == hrr_label]
    hrr = hrr.set_index(id_cols)
    for _, row in per_image[per_image["modality"].astype(str) != hrr_label].iterrows():
        key = (row["case_id"], row["slice_id"])
        if key not in hrr.index:
            continue
        ref = hrr.loc[key]
        rec = {c: row[c] for c in id_cols + ["modality"] if c in row}
        abs_primary = []
        for metric in PRIMARY_METRICS:
            if metric not in row or metric not in ref:
                continue
            diff = row[metric] - ref[metric]
            rec[f"{metric}_diff"] = float(diff)
            rec[f"{metric}_abs_diff"] = float(abs(diff))
            rec[f"{metric}_pct_diff"] = float(100 * diff / ref[metric]) if ref[metric] not in [0, np.nan] and not pd.isna(ref[metric]) else np.nan
            if not pd.isna(diff):
                abs_primary.append(abs(diff))
        rec["mean_abs_primary_hrr_gap"] = float(np.mean(abs_primary)) if abs_primary else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def summarize_modalities(per_image: pd.DataFrame, gaps: pd.DataFrame, hrr_label: str) -> pd.DataFrame:
    metric_cols = [c for c in per_image.columns if c not in {"image_path", "mask_path", "registered_image_path"}]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(per_image[c])]
    base = per_image.groupby("modality")[metric_cols].agg(["mean", "std", "median"]).reset_index()
    base.columns = ["_".join([str(x) for x in c if x]) for c in base.columns]
    if gaps.empty:
        return base
    gap_cols = [c for c in gaps.columns if c.endswith("_pct_diff") or c == "mean_abs_primary_hrr_gap"]
    gap_summary = gaps.groupby("modality")[gap_cols].agg(["mean", "std", "median"]).reset_index()
    gap_summary.columns = ["_".join([str(x) for x in c if x]) for c in gap_summary.columns]
    return base.merge(gap_summary, on="modality", how="left")


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    ok = np.isfinite(pvals)
    p = pvals[ok]
    if p.size == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * p.size / (np.arange(p.size) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty_like(q)
    tmp[order] = np.clip(q, 0, 1)
    out[ok] = tmp
    return out


def screen_bridge_metrics(per_image: pd.DataFrame, gaps: pd.DataFrame, hrr_label: str) -> pd.DataFrame:
    if gaps.empty:
        return pd.DataFrame()
    ids = ["case_id", "slice_id", "modality"]
    merged = per_image[per_image["modality"].astype(str) != hrr_label].merge(gaps[ids + ["mean_abs_primary_hrr_gap"]], on=ids, how="inner")
    exclude = set(ids + ["mean_abs_primary_hrr_gap"])
    exclude.update(c for c in merged.columns if c.endswith("_diff") or c.endswith("_abs_diff") or c.endswith("_pct_diff"))
    candidates = [c for c in merged.columns if c not in exclude and pd.api.types.is_numeric_dtype(merged[c])]
    records = []
    y = merged["mean_abs_primary_hrr_gap"]
    for c in candidates:
        x = merged[c]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 or x[ok].nunique() < 3:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        if np.isfinite(rho):
            records.append({"metric": c, "spearman_rho": float(rho), "p_value": float(p), "n": int(ok.sum())})
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["q_value"] = fdr_bh(out["p_value"].to_numpy())
    out["abs_rho"] = out["spearman_rho"].abs()
    return out.sort_values(["abs_rho", "q_value"], ascending=[False, True])


def write_outputs(outdir: Path, per_image: pd.DataFrame, gaps: pd.DataFrame, summary: pd.DataFrame, bridges: pd.DataFrame, args: argparse.Namespace) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(outdir / "per_image_metrics.csv", index=False)
    gaps.to_csv(outdir / "hrr_gaps.csv", index=False)
    summary.to_csv(outdir / "modality_summary.csv", index=False)
    bridges.to_csv(outdir / "bridge_metric_screening.csv", index=False)
    per_image.to_csv(outdir / "selected_slice_metrics.csv", index=False)
    gaps.to_csv(outdir / "selected_slice_metrics_with_HRR_gap.csv", index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(outdir / "analysis_config.json", "w", encoding="utf-8") as f:
        json.dump(config | {"registration_note": registration_note()}, f, indent=2, ensure_ascii=False)


def make_demo_dataset(outdir: Path) -> Path:
    rng = np.random.default_rng(7)
    data_dir = outdir / "demo_images"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = []
    modalities = [("HRR_52um", 1.0, 52.53), ("microPCCT_105um", 1.5, 105.06), ("microPCCT_210um", 2.5, 210.12), ("clinicalCT_500um", 4.0, 500.0)]
    for case in range(1, 4):
        for sl in range(1, 3):
            base = rng.normal(0, 0.05, (128, 128))
            yy, xx = np.mgrid[:128, :128]
            roi = ((yy - 64) ** 2 / 50**2 + (xx - 64) ** 2 / 44**2) <= 1
            trab = np.zeros_like(base)
            for _ in range(55):
                y, x = rng.integers(25, 103, size=2)
                angle = rng.uniform(0, np.pi)
                length = rng.integers(18, 55)
                ys = (y + np.sin(angle) * np.arange(-length, length)).astype(int)
                xs = (x + np.cos(angle) * np.arange(-length, length)).astype(int)
                good = (ys >= 0) & (ys < 128) & (xs >= 0) & (xs < 128)
                trab[ys[good], xs[good]] = 1
            trab = ndi.gaussian_filter(trab, 0.9)
            hrr = base + trab * roi
            for mod, sigma, res in modalities:
                img = ndi.gaussian_filter(hrr, sigma=sigma)
                img += rng.normal(0, 0.04 + sigma * 0.01, img.shape)
                img[~roi] = 0
                ip = data_dir / f"case{case:02d}_slice{sl:02d}_{mod}.npy"
                mp = data_dir / f"case{case:02d}_slice{sl:02d}_mask.npy"
                np.save(ip, img.astype(np.float32))
                np.save(mp, roi.astype(np.uint8))
                records.append({"case_id": f"{case:02d}", "slice_id": f"{sl:02d}", "modality": mod, "image_path": str(ip), "mask_path": str(mp), "resolution_um": res})
    manifest = outdir / "demo_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HRR-calibrated selected-slice trabecular metric analysis")
    p.add_argument("--manifest", type=Path, help="CSV manifest with one row per image")
    p.add_argument("--input-dir", type=Path, help="Optional DICOM root; filenames are parsed as in the LumbarSR selected-slice export")
    p.add_argument("--outdir", type=Path, default=Path("trabecular_outputs"))
    p.add_argument("--hrr-label", default="HRR_52um")
    p.add_argument("--default-resolution-um", type=float, default=105.06)
    p.add_argument("--threshold", choices=["otsu", "yen", "li", "fixed"], default="otsu")
    p.add_argument("--fixed-threshold", type=float, default=None)
    p.add_argument("--demo", action="store_true", help="Create and analyze a synthetic demo dataset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        args.outdir.mkdir(parents=True, exist_ok=True)
        args.manifest = make_demo_dataset(args.outdir)
    if args.manifest is None:
        raise SystemExit("Provide --manifest or run with --demo")
    per_image, gaps, summary, bridges = analyze_manifest(args)
    write_outputs(args.outdir, per_image, gaps, summary, bridges, args)
    print(f"Done. Outputs written to {args.outdir}")
    print(registration_note())


if __name__ == "__main__":
    main()
