#!/usr/bin/env python3
"""Reviewer 3.2 flux-tower GPP TAC validation.

This script identifies eddy-covariance sites in the northern permafrost
land-cover mask, aggregates sub-daily/hourly GPP to monthly means, removes
seasonality and long-term variation, calculates rolling lag-1 temporal
autocorrelation (TAC), and tests whether the pre/post-2008 reversal remains
detectable after controlling for site composition.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import math
import os
import platform
import re
import sys
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib_codex_cache"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow
import pyarrow.parquet as pq
import rasterio
import scipy
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rasterio.enums import Resampling
from scipy.stats import pearsonr
from shapely.geometry import Point
from statsmodels.tsa.seasonal import STL


PROJECT_DIR = Path("/Volumes/Zhengfei_01/Project 2 pf resilience")
CODE_DIR = PROJECT_DIR / "3_Code_new"


LANDCOVER_CLASS_NAMES = {
    0: "water_or_background",
    1: "evergreen_needleleaf_forest",
    2: "evergreen_broadleaf_forest",
    3: "deciduous_needleleaf_forest",
    4: "deciduous_broadleaf_forest",
    5: "mixed_forest",
    6: "closed_shrubland",
    7: "open_shrubland",
    8: "woody_savanna",
    9: "savanna",
    10: "grassland",
    11: "permanent_wetland",
    12: "cropland",
    13: "urban_built",
    14: "cropland_natural_mosaic",
    15: "snow_ice",
    16: "barren_sparse_vegetation",
}


@dataclass
class Config:
    flux_site_file: Path = Path(
        "/Volumes/Zhengfei_01/project 4 vegetation water carbon/"
        "1_Input/all_known_flux_sites_ge4_2026.csv"
    )
    permafrost_raster: Path = PROJECT_DIR / "1_Input/landcover_export_2010_5km.tif"
    gpp_file: Path = Path(
        "/Volumes/Zhengfei_01/project 4 vegetation water carbon/"
        "2_Output/fluxnet_HH_merged_24h.parquet"
    )
    output_dir: Path = PROJECT_DIR / "2_Output/flux_tower_validation_R3_2"
    breakpoint_year: int = 2008
    min_monthly_coverage: float = 0.70
    monthly_coverage_sensitivities: tuple[float, ...] = (0.60, 0.70, 0.80)
    min_total_years: int = 5
    min_valid_months_per_year: int = 8
    same_site_criteria: tuple[tuple[int, int], ...] = ((3, 3), (4, 4), (5, 5))
    primary_min_pre_years: int = 3
    primary_min_post_years: int = 3
    max_interpolated_gap_months: int = 2
    tac_window_years: int = 5
    tac_window_sensitivities: tuple[int, ...] = (3, 4, 5)
    min_valid_fraction_in_window: float = 0.80
    gpp_min: float = 0.0
    gpp_max: float = 100.0
    bootstrap_iterations: int = 1000
    random_seed: int = 42
    valid_landcover_classes: tuple[int, ...] = tuple(range(1, 13))
    growing_season_months: tuple[int, ...] = (6, 7, 8, 9)
    selected_gpp_variable: str | None = None
    quick: bool = False


@dataclass
class RunState:
    warnings: list[str] = field(default_factory=list)
    excluded_sites: dict[str, list[str]] = field(default_factory=dict)
    selected_columns: dict[str, str | None] = field(default_factory=dict)
    input_summaries: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    model_notes: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        logging.warning(message)
        self.warnings.append(message)

    def exclude(self, site_id: str, reason: str) -> None:
        self.excluded_sites.setdefault(site_id, []).append(reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--tac-window-years", type=int, default=None)
    parser.add_argument(
        "--valid-landcover-classes",
        type=str,
        default=None,
        help="Comma-separated raster classes treated as permafrost vegetation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer bootstrap iterations for smoke testing.",
    )
    return parser.parse_args()


def configure(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.coverage is not None:
        cfg.min_monthly_coverage = args.coverage
    if args.tac_window_years is not None:
        cfg.tac_window_years = args.tac_window_years
    if args.valid_landcover_classes:
        cfg.valid_landcover_classes = tuple(
            int(x.strip()) for x in args.valid_landcover_classes.split(",") if x.strip()
        )
    if args.quick:
        cfg.quick = True
        cfg.bootstrap_iterations = 100
    return cfg


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )


def ensure_dirs(cfg: Config) -> dict[str, Path]:
    dirs = {
        "tables": cfg.output_dir / "tables",
        "figures": cfg.output_dir / "figures",
        "intermediate": cfg.output_dir / "intermediate",
        "models": cfg.output_dir / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def package_versions() -> dict[str, str]:
    modules = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "geopandas": gpd.__version__,
        "rasterio": rasterio.__version__,
        "statsmodels": sm.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "pyarrow": pyarrow.__version__,
    }
    return modules


def normalize_site_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def first_matching_column(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    cols = list(columns)
    lower_map = {c.lower().strip(): c for c in cols}
    for pattern in patterns:
        pattern_l = pattern.lower()
        if pattern_l in lower_map:
            return lower_map[pattern_l]
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for col in cols:
            if regex.search(col):
                return col
    return None


def detect_site_columns(df: pd.DataFrame, state: RunState) -> dict[str, str | None]:
    cols = df.columns
    detected = {
        "site_id": first_matching_column(cols, ["site_id", "^site$", "site code", "siteid"]),
        "site_name": first_matching_column(cols, ["site name", "site_name", "name"]),
        "latitude": first_matching_column(cols, ["^lat$", "latitude"]),
        "longitude": first_matching_column(cols, ["^lon$", "^lng$", "longitude"]),
        "network": first_matching_column(cols, ["network", "^hub$", "data set", "dataset"]),
        "vegetation_type": first_matching_column(cols, ["igbp", "vegetation", "pft", "landcover"]),
        "country": first_matching_column(cols, ["country"]),
        "years": first_matching_column(cols, ["^years$", "period", "record"]),
        "start_year": first_matching_column(cols, ["start_year", "start year", "first year"]),
        "end_year": first_matching_column(cols, ["end_year", "end year", "last year"]),
    }
    required = ["site_id", "latitude", "longitude"]
    missing = [key for key in required if detected[key] is None]
    if missing:
        raise ValueError(f"Missing required site metadata columns: {missing}")
    state.selected_columns.update({f"site_{k}": v for k, v in detected.items()})
    return detected


def parse_year_range(value: Any) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan
    years = [int(x) for x in re.findall(r"(?:19|20)\d{2}", str(value))]
    if not years:
        return np.nan, np.nan
    return float(min(years)), float(max(years))


def load_site_metadata(cfg: Config, state: RunState) -> pd.DataFrame:
    logging.info("Reading flux-site metadata: %s", cfg.flux_site_file)
    df = pd.read_csv(cfg.flux_site_file)
    logging.info("Flux-site metadata columns: %s", list(df.columns))
    state.input_summaries["flux_site_metadata"] = {
        "path": str(cfg.flux_site_file),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    col = detect_site_columns(df, state)
    out = pd.DataFrame(
        {
            "site_id": df[col["site_id"]].map(normalize_site_id),
            "site_id_original": df[col["site_id"]].astype(str).str.strip(),
            "latitude": pd.to_numeric(df[col["latitude"]], errors="coerce"),
            "longitude": pd.to_numeric(df[col["longitude"]], errors="coerce"),
        }
    )
    optional_map = {
        "site_name": "site_name",
        "network": "network",
        "vegetation_type": "vegetation_type",
        "country": "country",
    }
    for target, key in optional_map.items():
        source = col[key]
        out[target] = df[source].astype(str).str.strip() if source else ""
    if col["start_year"] and col["end_year"]:
        out["metadata_start_year"] = pd.to_numeric(df[col["start_year"]], errors="coerce")
        out["metadata_end_year"] = pd.to_numeric(df[col["end_year"]], errors="coerce")
    elif col["years"]:
        parsed = df[col["years"]].map(parse_year_range)
        out["metadata_start_year"] = [x[0] for x in parsed]
        out["metadata_end_year"] = [x[1] for x in parsed]
    else:
        out["metadata_start_year"] = np.nan
        out["metadata_end_year"] = np.nan
    out = out.dropna(subset=["latitude", "longitude"])
    out = out.drop_duplicates(subset=["site_id"], keep="first")
    return out


def inspect_raster(cfg: Config, state: RunState) -> dict[str, Any]:
    logging.info("Inspecting permafrost-region raster: %s", cfg.permafrost_raster)
    with rasterio.open(cfg.permafrost_raster) as src:
        arr = src.read(1, masked=True)
        values = np.unique(arr.compressed()).astype(int).tolist()
        summary = {
            "path": str(cfg.permafrost_raster),
            "crs": str(src.crs),
            "bounds": tuple(float(x) for x in src.bounds),
            "nodata": None if src.nodata is None else float(src.nodata),
            "width": int(src.width),
            "height": int(src.height),
            "unique_values": values,
            "class_names": {str(v): LANDCOVER_CLASS_NAMES.get(v, "unknown") for v in values},
            "valid_landcover_classes": list(cfg.valid_landcover_classes),
            "valid_class_names": {
                str(v): LANDCOVER_CLASS_NAMES.get(v, "unknown")
                for v in cfg.valid_landcover_classes
            },
        }
    logging.info("Raster CRS: %s", summary["crs"])
    logging.info("Raster extent: %s", summary["bounds"])
    logging.info("Raster nodata: %s", summary["nodata"])
    logging.info("Raster unique values: %s", summary["unique_values"])
    logging.info("Configured valid permafrost vegetation classes: %s", summary["valid_class_names"])
    state.input_summaries["permafrost_raster"] = summary
    return summary


def sample_sites_in_permafrost(
    sites: pd.DataFrame, cfg: Config, state: RunState, dirs: dict[str, Path]
) -> pd.DataFrame:
    inspect_raster(cfg, state)
    geometry = [Point(xy) for xy in zip(sites["longitude"], sites["latitude"], strict=False)]
    gdf = gpd.GeoDataFrame(sites.copy(), geometry=geometry, crs="EPSG:4326")
    with rasterio.open(cfg.permafrost_raster) as src:
        if src.crs is not None and gdf.crs != src.crs:
            gdf_sample = gdf.to_crs(src.crs)
        else:
            gdf_sample = gdf
        coords = [(geom.x, geom.y) for geom in gdf_sample.geometry]
        sampled = [x[0] if len(x) else np.nan for x in src.sample(coords)]
    gdf["permafrost_class"] = pd.to_numeric(sampled, errors="coerce")
    gdf["permafrost_class_name"] = gdf["permafrost_class"].map(
        lambda x: LANDCOVER_CLASS_NAMES.get(int(x), "unknown") if pd.notna(x) else "missing"
    )
    gdf["in_valid_permafrost_pixel"] = gdf["permafrost_class"].isin(
        cfg.valid_landcover_classes
    )
    selected = pd.DataFrame(gdf[gdf["in_valid_permafrost_pixel"]].drop(columns="geometry"))
    selected = selected.sort_values(["latitude", "site_id"], ascending=[False, True])
    out_path = dirs["tables"] / "flux_sites_in_permafrost.csv"
    selected.to_csv(out_path, index=False)
    gdf.drop(columns="geometry").to_csv(dirs["tables"] / "all_flux_sites_with_raster_sample.csv", index=False)
    logging.info("Selected %s/%s flux sites in valid permafrost pixels.", len(selected), len(gdf))
    return selected


def inspect_gpp_file(cfg: Config, state: RunState) -> dict[str, Any]:
    logging.info("Inspecting GPP parquet: %s", cfg.gpp_file)
    parquet = pq.ParquetFile(cfg.gpp_file)
    columns = parquet.schema.names
    gpp_candidates = [
        c
        for c in columns
        if "GPP" in c.upper() and not c.upper().endswith("_QC") and not c.upper().endswith("QC")
    ]
    qc_candidates = [c for c in columns if "QC" in c.upper() or "FLAG" in c.upper()]
    timestamp_col = first_matching_column(
        columns, ["TIMESTAMP_START", "TIMESTAMP", "datetime", "date_time", "date"]
    )
    site_col = first_matching_column(columns, ["site_id", "^site$", "siteid"])
    if cfg.selected_gpp_variable:
        gpp_col = cfg.selected_gpp_variable
    elif "GPP_NT_VUT_REF" in columns:
        gpp_col = "GPP_NT_VUT_REF"
    elif "GPP_DT_VUT_REF" in columns:
        gpp_col = "GPP_DT_VUT_REF"
    elif gpp_candidates:
        gpp_col = gpp_candidates[0]
    else:
        raise ValueError("No GPP variable found in parquet file.")
    summary = {
        "path": str(cfg.gpp_file),
        "columns": columns,
        "row_groups": int(parquet.num_row_groups),
        "records": int(parquet.metadata.num_rows),
        "timestamp_column": timestamp_col,
        "site_id_column": site_col,
        "gpp_candidates": gpp_candidates,
        "selected_gpp_variable": gpp_col,
        "units": "not encoded in parquet schema; inferred as source GPP units",
        "quality_control_fields": qc_candidates,
    }
    logging.info("GPP parquet columns: %s", columns)
    logging.info("Selected GPP variable: %s", gpp_col)
    logging.info("GPP QC fields detected: %s", qc_candidates)
    if site_col is None:
        raise ValueError("No site-ID column found in GPP parquet.")
    state.selected_columns.update(
        {
            "gpp_site_id": site_col,
            "gpp_timestamp": timestamp_col,
            "gpp_variable": gpp_col,
            "gpp_qc_fields": ",".join(qc_candidates),
        }
    )
    state.input_summaries["gpp_file"] = summary
    return summary


def parquet_filter_for_sites(site_col: str, site_ids: list[str]) -> Any:
    field = ds.field(site_col)
    return field.isin(site_ids)


def load_gpp_for_sites(
    cfg: Config, state: RunState, permafrost_sites: pd.DataFrame
) -> pd.DataFrame:
    summary = inspect_gpp_file(cfg, state)
    site_col = summary["site_id_column"]
    gpp_col = summary["selected_gpp_variable"]
    timestamp_col = summary["timestamp_column"]
    columns = [site_col, gpp_col]
    parquet_cols = set(summary["columns"])
    for c in ["year", "month", "day", "hour", "minute", timestamp_col]:
        if c and c in parquet_cols and c not in columns:
            columns.append(c)
    site_ids = sorted(permafrost_sites["site_id"].dropna().unique().tolist())
    logging.info("Reading GPP records for %s permafrost metadata sites.", len(site_ids))
    dataset = ds.dataset(cfg.gpp_file, format="parquet")
    table = dataset.to_table(columns=columns, filter=parquet_filter_for_sites(site_col, site_ids))
    gpp = table.to_pandas()
    gpp["site_id"] = gpp[site_col].map(normalize_site_id)
    matched_sites = sorted(set(gpp["site_id"].unique()) & set(site_ids))
    unmatched = sorted(set(site_ids) - set(matched_sites))
    logging.info("Matched %s/%s permafrost sites to GPP data.", len(matched_sites), len(site_ids))
    if unmatched:
        state.warn(f"Permafrost metadata sites not found in GPP data: {unmatched}")
    state.input_summaries["gpp_file"]["permafrost_sites_in_metadata"] = len(site_ids)
    state.input_summaries["gpp_file"]["matched_permafrost_sites"] = len(matched_sites)
    state.input_summaries["gpp_file"]["unmatched_permafrost_site_ids"] = unmatched
    return standardize_gpp_records(gpp, summary, cfg, state)


def standardize_gpp_records(
    gpp: pd.DataFrame, summary: dict[str, Any], cfg: Config, state: RunState
) -> pd.DataFrame:
    gpp_col = summary["selected_gpp_variable"]
    timestamp_col = summary["timestamp_column"]
    gpp = gpp.copy()
    if timestamp_col and timestamp_col in gpp.columns:
        gpp["timestamp"] = pd.to_datetime(gpp[timestamp_col], errors="coerce")
    else:
        required = ["year", "month", "day"]
        missing = [c for c in required if c not in gpp.columns]
        if missing:
            raise ValueError(f"Cannot construct timestamps; missing columns {missing}")
        hour = pd.to_numeric(gpp["hour"], errors="coerce").fillna(0) if "hour" in gpp.columns else 0
        minute = (
            pd.to_numeric(gpp["minute"], errors="coerce").fillna(0)
            if "minute" in gpp.columns
            else 0
        )
        gpp["timestamp"] = pd.to_datetime(
            {
                "year": pd.to_numeric(gpp["year"], errors="coerce"),
                "month": pd.to_numeric(gpp["month"], errors="coerce"),
                "day": pd.to_numeric(gpp["day"], errors="coerce"),
                "hour": hour,
                "minute": minute,
            },
            errors="coerce",
        )
    before = len(gpp)
    gpp[gpp_col] = pd.to_numeric(gpp[gpp_col], errors="coerce")
    gpp = gpp.dropna(subset=["site_id", "timestamp", gpp_col])
    gpp = gpp[(gpp[gpp_col] >= cfg.gpp_min) & (gpp[gpp_col] <= cfg.gpp_max)]
    gpp = gpp.drop_duplicates(subset=["site_id", "timestamp"], keep="first")
    gpp["year"] = gpp["timestamp"].dt.year
    gpp["month"] = gpp["timestamp"].dt.month
    gpp["date"] = gpp["timestamp"].dt.to_period("M").dt.to_timestamp()
    logging.info("Valid GPP observations retained: %s/%s", len(gpp), before)
    state.input_summaries["gpp_file"]["valid_observations_retained"] = int(len(gpp))
    return gpp[["site_id", "timestamp", "date", "year", "month", gpp_col]].rename(
        columns={gpp_col: "gpp"}
    )


def infer_expected_observations_per_day(gpp: pd.DataFrame) -> int:
    sample = gpp[["site_id", "timestamp"]].copy()
    sample["day_date"] = sample["timestamp"].dt.floor("D")
    counts = sample.groupby(["site_id", "day_date"]).size()
    if counts.empty:
        return 1
    mode = int(counts.mode().iloc[0])
    if mode >= 40:
        return 48
    if mode >= 18:
        return 24
    return 1


def aggregate_monthly_gpp(
    gpp: pd.DataFrame, cfg: Config, dirs: dict[str, Path], coverage: float | None = None
) -> pd.DataFrame:
    threshold = cfg.min_monthly_coverage if coverage is None else coverage
    expected_per_day = infer_expected_observations_per_day(gpp)
    logging.info("Inferred expected observations per day: %s", expected_per_day)
    monthly = (
        gpp.groupby(["site_id", "year", "month", "date"], as_index=False)
        .agg(monthly_gpp=("gpp", "mean"), valid_observation_count=("gpp", "size"))
        .sort_values(["site_id", "date"])
    )
    monthly["days_in_month"] = monthly.apply(
        lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1], axis=1
    )
    monthly["expected_observation_count"] = monthly["days_in_month"] * expected_per_day
    monthly["monthly_coverage"] = (
        monthly["valid_observation_count"] / monthly["expected_observation_count"]
    )
    monthly = monthly[monthly["monthly_coverage"] >= threshold].copy()
    monthly = monthly.drop(columns=["days_in_month"])
    if coverage is None or math.isclose(threshold, cfg.min_monthly_coverage):
        path = dirs["intermediate"] / "monthly_gpp_by_site.parquet"
        monthly.to_parquet(path, index=False)
        monthly.to_csv(dirs["tables"] / "monthly_gpp_by_site.csv", index=False)
    return monthly


def valid_year_count(monthly: pd.DataFrame, cfg: Config) -> pd.Series:
    per_year = monthly.groupby(["site_id", "year"]).size().reset_index(name="valid_months")
    valid_years = per_year[per_year["valid_months"] >= cfg.min_valid_months_per_year]
    return valid_years.groupby("site_id").size()


def max_gap_months(dates: pd.Series) -> int:
    if len(dates) <= 1:
        return 0
    periods = pd.PeriodIndex(pd.to_datetime(dates), freq="M").sort_values()
    diffs = periods[1:].asi8 - periods[:-1].asi8
    return int(max(diffs.max() - 1, 0))


def build_site_record_table(monthly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    records = []
    valid_years = valid_year_count(monthly, cfg)
    per_year = monthly.groupby(["site_id", "year"]).size().reset_index(name="valid_months")
    pre_years = per_year[
        (per_year["year"] < cfg.breakpoint_year)
        & (per_year["valid_months"] >= cfg.min_valid_months_per_year)
    ].groupby("site_id").size()
    post_years = per_year[
        (per_year["year"] >= cfg.breakpoint_year)
        & (per_year["valid_months"] >= cfg.min_valid_months_per_year)
    ].groupby("site_id").size()
    for site_id, df in monthly.groupby("site_id"):
        dates = pd.to_datetime(df["date"]).sort_values()
        full_months = pd.period_range(dates.min(), dates.max(), freq="M")
        records.append(
            {
                "site_id": site_id,
                "first_valid_date": dates.min(),
                "last_valid_date": dates.max(),
                "total_valid_months": int(len(dates)),
                "total_valid_years": int(valid_years.get(site_id, 0)),
                "valid_pre_2008_years": int(pre_years.get(site_id, 0)),
                "valid_post_2008_years": int(post_years.get(site_id, 0)),
                "spans_2008": bool(
                    (dates.dt.year < cfg.breakpoint_year).any()
                    and (dates.dt.year >= cfg.breakpoint_year).any()
                ),
                "maximum_temporal_gap_months": max_gap_months(dates),
                "fraction_missing_months": float(1.0 - (len(dates) / len(full_months))),
            }
        )
    out = pd.DataFrame(records)
    out["included_all_sites"] = out["total_valid_years"] >= cfg.min_total_years
    for pre, post in cfg.same_site_criteria:
        name = f"included_same_site_{pre}_{post}"
        out[name] = (
            out["spans_2008"]
            & (out["valid_pre_2008_years"] >= pre)
            & (out["valid_post_2008_years"] >= post)
        )
    out["included_same_site_primary"] = out[
        f"included_same_site_{cfg.primary_min_pre_years}_{cfg.primary_min_post_years}"
    ]
    return out


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    s = series.copy()
    missing = s.isna()
    interpolation_flag = pd.Series(False, index=s.index)
    if not missing.any():
        return s, interpolation_flag
    groups = (missing != missing.shift()).cumsum()
    for _, idx in s[missing].groupby(groups[missing]).groups.items():
        idx = list(idx)
        if len(idx) <= max_gap:
            first_pos = s.index.get_loc(idx[0])
            last_pos = s.index.get_loc(idx[-1])
            has_left = first_pos > 0 and pd.notna(s.iloc[first_pos - 1])
            has_right = last_pos < len(s) - 1 and pd.notna(s.iloc[last_pos + 1])
            if has_left and has_right:
                interpolation_flag.loc[idx] = True
    filled = s.interpolate(method="linear", limit=max_gap, limit_area="inside")
    filled.loc[~interpolation_flag & missing] = np.nan
    return filled, interpolation_flag


def robust_stl_residual(series: pd.Series, cfg: Config) -> pd.DataFrame:
    filled, interpolation_flag = interpolate_short_gaps(series, cfg.max_interpolated_gap_months)
    out = pd.DataFrame(index=series.index)
    out["monthly_gpp"] = series
    out["trend"] = np.nan
    out["seasonal"] = np.nan
    out["residual"] = np.nan
    out["interpolation_flag"] = interpolation_flag.values
    valid_after_fill = filled.notna()
    seg_id = (valid_after_fill != valid_after_fill.shift()).cumsum()
    for _, idx in filled[valid_after_fill].groupby(seg_id[valid_after_fill]).groups.items():
        idx = list(idx)
        if len(idx) < 24:
            continue
        try:
            result = STL(filled.loc[idx], period=12, robust=True).fit()
        except Exception as exc:
            logging.warning("STL failed for segment ending %s: %s", idx[-1], exc)
            continue
        out.loc[idx, "trend"] = result.trend
        out.loc[idx, "seasonal"] = result.seasonal
        out.loc[idx, "residual"] = result.resid
    return out


def climatology_detrended_residual(series: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    out["monthly_gpp"] = series
    month_means = series.groupby(series.index.month).transform("mean")
    anomaly = series - month_means
    valid = anomaly.dropna()
    out["seasonal"] = month_means
    out["trend"] = np.nan
    out["residual"] = np.nan
    out["interpolation_flag"] = False
    if len(valid) >= 24:
        x = np.arange(len(series), dtype=float)
        mask = anomaly.notna().to_numpy()
        slope, intercept = np.polyfit(x[mask], anomaly.to_numpy()[mask], 1)
        trend = slope * x + intercept
        out["trend"] = trend
        out["residual"] = anomaly - trend
    return out


def calculate_residuals(
    monthly: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
    method: str = "stl",
    save_primary: bool = True,
) -> pd.DataFrame:
    empty_cols = [
        "date",
        "site_id",
        "monthly_gpp",
        "trend",
        "seasonal",
        "residual",
        "interpolation_flag",
        "residual_method",
    ]
    if monthly.empty or "site_id" not in monthly.columns:
        residuals = pd.DataFrame(columns=empty_cols)
        if save_primary:
            residuals.to_parquet(dirs["intermediate"] / "monthly_gpp_residuals_by_site.parquet", index=False)
            residuals.to_csv(dirs["tables"] / "monthly_gpp_residuals_by_site.csv", index=False)
        return residuals
    pieces = []
    for site_id, df in monthly.groupby("site_id"):
        df = df.sort_values("date")
        full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
        series = df.set_index("date")["monthly_gpp"].reindex(full_index)
        if method == "stl":
            resid = robust_stl_residual(series, cfg)
        elif method == "climatology_detrended":
            resid = climatology_detrended_residual(series)
        else:
            raise ValueError(f"Unknown residual method: {method}")
        resid = resid.reset_index().rename(columns={"index": "date"})
        resid["site_id"] = site_id
        resid["residual_method"] = method
        pieces.append(resid)
    residuals = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=empty_cols)
    if save_primary:
        residuals.to_parquet(dirs["intermediate"] / "monthly_gpp_residuals_by_site.parquet", index=False)
        residuals.to_csv(dirs["tables"] / "monthly_gpp_residuals_by_site.csv", index=False)
    return residuals


def lag1_autocorrelation(values: np.ndarray) -> float:
    if len(values) < 3:
        return np.nan
    x = values[:-1]
    y = values[1:]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    if np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(pearsonr(x[valid], y[valid])[0])


def paired_autocorrelation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson autocorrelation from explicit lagged pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    if np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(pearsonr(x[valid], y[valid])[0])


def rolling_tac_for_site(
    df: pd.DataFrame, cfg: Config, window_years: int
) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    window_months = window_years * 12
    rows = []
    if len(df) < window_months:
        return pd.DataFrame(rows)
    valid_resid = df["residual"].notna() & ~df["interpolation_flag"].astype(bool)
    for start in range(0, len(df) - window_months + 1):
        end = start + window_months
        win = df.iloc[start:end].copy()
        valid_fraction = float(valid_resid.iloc[start:end].mean())
        if valid_fraction < cfg.min_valid_fraction_in_window:
            continue
        pair_mask = (
            win["residual"].notna()
            & win["residual"].shift(1).notna()
            & ~win["interpolation_flag"].astype(bool)
            & ~win["interpolation_flag"].astype(bool).shift(1, fill_value=True)
        )
        # The complete monthly index prevents hidden jumps; this keeps the intent explicit.
        month_diff = pd.to_datetime(win["date"]).dt.to_period("M").astype(int).diff()
        pair_mask &= month_diff.eq(1)
        if pair_mask.sum() < max(3, int((window_months - 1) * cfg.min_valid_fraction_in_window)):
            continue
        tac = paired_autocorrelation(
            win["residual"].shift(1)[pair_mask].to_numpy(),
            win["residual"][pair_mask].to_numpy(),
        )
        if not np.isfinite(tac):
            continue
        centre = pd.to_datetime(win["date"].iloc[window_months // 2])
        rows.append(
            {
                "site_id": df["site_id"].iloc[0],
                "centre_date": centre,
                "centre_year": centre.year + (centre.month - 0.5) / 12.0,
                "year": centre.year,
                "month": centre.month,
                "TAC": tac,
                "number_of_valid_pairs": int(pair_mask.sum()),
                "valid_fraction_in_window": valid_fraction,
                "window_start": win["date"].iloc[0],
                "window_end": win["date"].iloc[-1],
                "tac_window_years": window_years,
                "residual_method": win["residual_method"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def calculate_rolling_tac(
    residuals: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
    window_years: int | None = None,
    save_primary: bool = True,
) -> pd.DataFrame:
    empty_cols = [
        "site_id",
        "centre_date",
        "centre_year",
        "year",
        "month",
        "TAC",
        "number_of_valid_pairs",
        "valid_fraction_in_window",
        "window_start",
        "window_end",
        "tac_window_years",
        "residual_method",
    ]
    years = cfg.tac_window_years if window_years is None else window_years
    if residuals.empty or "site_id" not in residuals.columns:
        tac = pd.DataFrame(columns=empty_cols)
        if save_primary:
            tac.to_csv(dirs["tables"] / "site_level_rolling_tac.csv", index=False)
        return tac
    pieces = [rolling_tac_for_site(df, cfg, years) for _, df in residuals.groupby("site_id")]
    tac = (
        pd.concat([p for p in pieces if not p.empty], ignore_index=True)
        if pieces and any(not p.empty for p in pieces)
        else pd.DataFrame(columns=empty_cols)
    )
    if save_primary:
        tac.to_csv(dirs["tables"] / "site_level_rolling_tac.csv", index=False)
    return tac


def slope_with_pvalue(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, float]:
    clean = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean[x_col].nunique() < 2:
        return np.nan, np.nan, np.nan
    res = st.linregress(clean[x_col], clean[y_col])
    return float(res.slope), float(res.pvalue), float(res.stderr)


def site_level_slopes(tac: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = [
        "site_id",
        "pre_2008_TAC_slope",
        "pre_2008_TAC_slope_p",
        "pre_2008_TAC_slope_se",
        "post_2008_TAC_slope",
        "post_2008_TAC_slope_p",
        "post_2008_TAC_slope_se",
        "TAC_slope_difference",
        "n_pre_tac",
        "n_post_tac",
    ]
    if tac.empty or "site_id" not in tac.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for site_id, df in tac.groupby("site_id"):
        pre = df[df["centre_year"] < cfg.breakpoint_year]
        post = df[df["centre_year"] >= cfg.breakpoint_year]
        pre_slope, pre_p, pre_se = slope_with_pvalue(pre, "centre_year", "TAC")
        post_slope, post_p, post_se = slope_with_pvalue(post, "centre_year", "TAC")
        rows.append(
            {
                "site_id": site_id,
                "pre_2008_TAC_slope": pre_slope,
                "pre_2008_TAC_slope_p": pre_p,
                "pre_2008_TAC_slope_se": pre_se,
                "post_2008_TAC_slope": post_slope,
                "post_2008_TAC_slope_p": post_p,
                "post_2008_TAC_slope_se": post_se,
                "TAC_slope_difference": post_slope - pre_slope
                if np.isfinite(pre_slope) and np.isfinite(post_slope)
                else np.nan,
                "n_pre_tac": int(len(pre)),
                "n_post_tac": int(len(post)),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def bootstrap_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan, np.nan
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def calculate_period_tac(
    residuals: pd.DataFrame,
    site_records: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    cols = [
        "site_id",
        "pre_2008_TAC",
        "post_2008_TAC",
        "equal_pair_count",
        "pre_2008_TAC_bootstrap_mean",
        "pre_2008_TAC_ci_low",
        "pre_2008_TAC_ci_high",
        "post_2008_TAC_bootstrap_mean",
        "post_2008_TAC_ci_low",
        "post_2008_TAC_ci_high",
        "post_minus_pre_TAC_mean",
        "post_minus_pre_TAC_ci_low",
        "post_minus_pre_TAC_ci_high",
    ]
    if residuals.empty or "site_id" not in residuals.columns:
        out = pd.DataFrame(columns=cols)
        out.to_csv(dirs["tables"] / "period_specific_TAC.csv", index=False)
        return out
    rng = np.random.default_rng(cfg.random_seed)
    eligible = set(site_records.loc[site_records["included_same_site_primary"], "site_id"])
    rows = []
    for site_id, df in residuals[residuals["site_id"].isin(eligible)].groupby("site_id"):
        clean = df.sort_values("date").copy()
        clean = clean[clean["residual"].notna() & ~clean["interpolation_flag"].astype(bool)]
        clean["period"] = np.where(
            pd.to_datetime(clean["date"]).dt.year < cfg.breakpoint_year, "pre_2008", "post_2008"
        )
        pair_rows = []
        for period, sub in clean.groupby("period"):
            sub = sub.sort_values("date")
            month_diff = pd.to_datetime(sub["date"]).dt.to_period("M").astype(int).diff()
            pair_mask = month_diff.eq(1)
            pair_rows.append(
                pd.DataFrame(
                    {
                        "period": period,
                        "x": sub["residual"].shift(1)[pair_mask].to_numpy(),
                        "y": sub["residual"][pair_mask].to_numpy(),
                    }
                )
            )
        pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
        if pairs.empty or pairs["period"].nunique() < 2:
            continue
        pre_pairs = pairs[pairs["period"] == "pre_2008"][["x", "y"]].dropna().to_numpy()
        post_pairs = pairs[pairs["period"] == "post_2008"][["x", "y"]].dropna().to_numpy()
        n = min(len(pre_pairs), len(post_pairs))
        if n < 6:
            continue
        boot = []
        for _ in range(cfg.bootstrap_iterations):
            pre_idx = rng.integers(0, len(pre_pairs), n)
            post_idx = rng.integers(0, len(post_pairs), n)
            pre_tac = pearsonr(pre_pairs[pre_idx, 0], pre_pairs[pre_idx, 1])[0]
            post_tac = pearsonr(post_pairs[post_idx, 0], post_pairs[post_idx, 1])[0]
            if np.isfinite(pre_tac) and np.isfinite(post_tac):
                boot.append((pre_tac, post_tac, post_tac - pre_tac))
        boot_arr = np.asarray(boot)
        rows.append(
            {
                "site_id": site_id,
                "pre_2008_TAC": lag1_autocorrelation(clean.loc[clean["period"] == "pre_2008", "residual"].to_numpy()),
                "post_2008_TAC": lag1_autocorrelation(clean.loc[clean["period"] == "post_2008", "residual"].to_numpy()),
                "equal_pair_count": int(n),
                "pre_2008_TAC_bootstrap_mean": float(np.nanmean(boot_arr[:, 0])) if len(boot_arr) else np.nan,
                "pre_2008_TAC_ci_low": float(np.nanpercentile(boot_arr[:, 0], 2.5)) if len(boot_arr) else np.nan,
                "pre_2008_TAC_ci_high": float(np.nanpercentile(boot_arr[:, 0], 97.5)) if len(boot_arr) else np.nan,
                "post_2008_TAC_bootstrap_mean": float(np.nanmean(boot_arr[:, 1])) if len(boot_arr) else np.nan,
                "post_2008_TAC_ci_low": float(np.nanpercentile(boot_arr[:, 1], 2.5)) if len(boot_arr) else np.nan,
                "post_2008_TAC_ci_high": float(np.nanpercentile(boot_arr[:, 1], 97.5)) if len(boot_arr) else np.nan,
                "post_minus_pre_TAC_mean": float(np.nanmean(boot_arr[:, 2])) if len(boot_arr) else np.nan,
                "post_minus_pre_TAC_ci_low": float(np.nanpercentile(boot_arr[:, 2], 2.5)) if len(boot_arr) else np.nan,
                "post_minus_pre_TAC_ci_high": float(np.nanpercentile(boot_arr[:, 2], 97.5)) if len(boot_arr) else np.nan,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(dirs["tables"] / "period_specific_TAC.csv", index=False)
    return out


def same_site_tests(
    slopes: pd.DataFrame, site_records: pd.DataFrame, cfg: Config, dirs: dict[str, Path]
) -> dict[str, Any]:
    if slopes.empty or "site_id" not in slopes.columns:
        summary = {
            "n_same_site_with_pre_post_slopes": 0,
            "mean_trend_difference": np.nan,
            "mean_trend_difference_ci_low": np.nan,
            "mean_trend_difference_ci_high": np.nan,
            "median_trend_difference": np.nan,
            "median_trend_difference_ci_low": np.nan,
            "median_trend_difference_ci_high": np.nan,
            "wilcoxon_greater_p": np.nan,
            "paired_ttest_greater_p": np.nan,
            "percent_negative_pre_2008_TAC_trend": np.nan,
            "percent_positive_post_2008_TAC_trend": np.nan,
            "percent_negative_to_positive_reversal": np.nan,
            "same_site_ids": [],
        }
        pd.DataFrame([summary]).to_csv(dirs["tables"] / "same_site_test_summary.csv", index=False)
        slopes.to_csv(dirs["tables"] / "same_site_slope_results.csv", index=False)
        return summary
    eligible = set(site_records.loc[site_records["included_same_site_primary"], "site_id"])
    same = slopes[slopes["site_id"].isin(eligible)].copy()
    same = same.dropna(subset=["pre_2008_TAC_slope", "post_2008_TAC_slope", "TAC_slope_difference"])
    rng = np.random.default_rng(cfg.random_seed)
    diffs = same["TAC_slope_difference"].to_numpy(dtype=float)
    mean_diff, mean_low, mean_high = bootstrap_ci(diffs, cfg.bootstrap_iterations, rng)
    median_boot = np.nan
    median_low = np.nan
    median_high = np.nan
    if len(diffs) > 1:
        med_samples = np.median(
            rng.choice(diffs, size=(cfg.bootstrap_iterations, len(diffs)), replace=True), axis=1
        )
        median_boot = float(np.median(diffs))
        median_low = float(np.percentile(med_samples, 2.5))
        median_high = float(np.percentile(med_samples, 97.5))
    try:
        wilcoxon_p = float(st.wilcoxon(diffs, alternative="greater").pvalue) if len(diffs) else np.nan
    except ValueError:
        wilcoxon_p = np.nan
    ttest_p = float(st.ttest_1samp(diffs, 0, alternative="greater").pvalue) if len(diffs) > 1 else np.nan
    reversal = (
        (same["pre_2008_TAC_slope"] < 0) & (same["post_2008_TAC_slope"] > 0)
    )
    summary = {
        "n_same_site_with_pre_post_slopes": int(len(same)),
        "mean_trend_difference": mean_diff,
        "mean_trend_difference_ci_low": mean_low,
        "mean_trend_difference_ci_high": mean_high,
        "median_trend_difference": median_boot,
        "median_trend_difference_ci_low": median_low,
        "median_trend_difference_ci_high": median_high,
        "wilcoxon_greater_p": wilcoxon_p,
        "paired_ttest_greater_p": ttest_p,
        "percent_negative_pre_2008_TAC_trend": float((same["pre_2008_TAC_slope"] < 0).mean() * 100)
        if len(same)
        else np.nan,
        "percent_positive_post_2008_TAC_trend": float((same["post_2008_TAC_slope"] > 0).mean() * 100)
        if len(same)
        else np.nan,
        "percent_negative_to_positive_reversal": float(reversal.mean() * 100) if len(same) else np.nan,
        "same_site_ids": sorted(same["site_id"].tolist()),
    }
    pd.DataFrame([summary]).to_csv(dirs["tables"] / "same_site_test_summary.csv", index=False)
    same.to_csv(dirs["tables"] / "same_site_slope_results.csv", index=False)
    return summary


def annual_site_tac(tac: pd.DataFrame) -> pd.DataFrame:
    if tac.empty or not {"site_id", "year", "TAC", "number_of_valid_pairs"}.issubset(tac.columns):
        return pd.DataFrame(columns=["site_id", "year", "TAC", "number_of_valid_pairs"])
    return (
        tac.groupby(["site_id", "year"], as_index=False)
        .agg(TAC=("TAC", "mean"), number_of_valid_pairs=("number_of_valid_pairs", "sum"))
        .dropna(subset=["TAC"])
    )


def bootstrap_annual_mean(
    values: np.ndarray, iterations: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan, np.nan
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def aggregate_trajectories(
    tac: pd.DataFrame,
    site_records: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)
    annual = annual_site_tac(tac)
    if annual.empty:
        trajectory = pd.DataFrame(
            columns=["year", "trajectory", "mean_TAC", "ci_low", "ci_high", "n_sites"]
        )
        trajectory.to_csv(dirs["tables"] / "annual_TAC_trajectories.csv", index=False)
        c2 = pd.DataFrame(
            columns=[
                "start_year",
                "end_year",
                "period_length_years",
                "n_common_sites",
                "site_ids",
                "slope",
                "slope_p",
                "slope_se",
                "score",
            ]
        )
        c2.to_csv(dirs["tables"] / "fixed_common_site_subset_trends.csv", index=False)
        return trajectory, c2
    rows = []
    for year, sub in annual.groupby("year"):
        mean, low, high = bootstrap_annual_mean(sub["TAC"].to_numpy(), cfg.bootstrap_iterations, rng)
        rows.append(
            {
                "year": int(year),
                "trajectory": "all_site_unbalanced",
                "mean_TAC": mean,
                "ci_low": low,
                "ci_high": high,
                "n_sites": int(sub["site_id"].nunique()),
            }
        )
    same_ids = set(site_records.loc[site_records["included_same_site_primary"], "site_id"])
    same_annual = annual[annual["site_id"].isin(same_ids)]
    for year, sub in same_annual.groupby("year"):
        mean, low, high = bootstrap_annual_mean(sub["TAC"].to_numpy(), cfg.bootstrap_iterations, rng)
        rows.append(
            {
                "year": int(year),
                "trajectory": "same_site_subset",
                "mean_TAC": mean,
                "ci_low": low,
                "ci_high": high,
                "n_sites": int(sub["site_id"].nunique()),
            }
        )
    counts = annual.groupby("year")["site_id"].nunique()
    min_sites = int(counts.min()) if len(counts) else 0
    balanced_rows = []
    if min_sites > 0:
        for year, sub in annual.groupby("year"):
            values = sub["TAC"].to_numpy()
            draws = []
            for _ in range(cfg.bootstrap_iterations):
                draw = rng.choice(values, size=min_sites, replace=False if len(values) >= min_sites else True)
                draws.append(float(np.nanmean(draw)))
            balanced_rows.append(
                {
                    "year": int(year),
                    "trajectory": "balanced_resampling",
                    "mean_TAC": float(np.mean(draws)),
                    "ci_low": float(np.percentile(draws, 2.5)),
                    "ci_high": float(np.percentile(draws, 97.5)),
                    "n_sites": min_sites,
                }
            )
    trajectory = pd.DataFrame(rows + balanced_rows)
    trajectory.to_csv(dirs["tables"] / "annual_TAC_trajectories.csv", index=False)
    c2 = fixed_common_site_subsets(annual, cfg)
    c2.to_csv(dirs["tables"] / "fixed_common_site_subset_trends.csv", index=False)
    return trajectory, c2


def fixed_common_site_subsets(annual: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = [
        "start_year",
        "end_year",
        "period_length_years",
        "n_common_sites",
        "site_ids",
        "slope",
        "slope_p",
        "slope_se",
        "score",
    ]
    if annual.empty:
        return pd.DataFrame(columns=cols)
    years = sorted(annual["year"].dropna().astype(int).unique().tolist())
    rows = []
    for i, start in enumerate(years):
        for end in years[i + cfg.min_total_years - 1 :]:
            period = list(range(start, end + 1))
            if not set(period).issubset(set(years)):
                continue
            pivot = annual[annual["year"].isin(period)].pivot_table(
                index="site_id", columns="year", values="TAC", aggfunc="mean"
            )
            common = pivot.dropna()
            if common.empty:
                continue
            yearly = common.mean(axis=0).reset_index()
            yearly.columns = ["year", "TAC"]
            slope, pvalue, se = slope_with_pvalue(yearly, "year", "TAC")
            rows.append(
                {
                    "start_year": int(start),
                    "end_year": int(end),
                    "period_length_years": int(end - start + 1),
                    "n_common_sites": int(len(common)),
                    "site_ids": ";".join(common.index.astype(str)),
                    "slope": slope,
                    "slope_p": pvalue,
                    "slope_se": se,
                    "score": int(end - start + 1) * int(len(common)),
                }
            )
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["score", "period_length_years", "n_common_sites"], ascending=False)
    return out


def fit_segmented_models(
    tac: pd.DataFrame, cfg: Config, dirs: dict[str, Path], state: RunState
) -> pd.DataFrame:
    cols = [
        "model",
        "term",
        "estimate",
        "ci_low",
        "ci_high",
        "pvalue",
        "n_observations",
        "n_sites",
        "converged",
    ]
    data = tac.dropna(subset=["TAC", "centre_year", "site_id"]).copy()
    data["time"] = data["centre_year"] - cfg.breakpoint_year
    data["post"] = np.maximum(data["time"], 0)
    data["weight"] = data["number_of_valid_pairs"].clip(lower=1)
    rows = []
    if len(data) < 10 or data["site_id"].nunique() < 2:
        state.warn("Too few rolling TAC observations for mixed-effects model.")
        out = pd.DataFrame(rows, columns=cols)
        out.to_csv(dirs["models"] / "mixed_effects_model_results.csv", index=False)
        return out
    model_kind = "random_slope"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mixed = smf.mixedlm("TAC ~ time + post", data, groups=data["site_id"], re_formula="~time")
            result = mixed.fit(method="lbfgs", maxiter=500, reml=False)
        if not result.converged:
            raise RuntimeError("random-slope mixed model did not converge")
    except Exception as exc:
        state.model_notes.append(f"Random-slope mixed model failed: {exc}; using random intercept.")
        model_kind = "random_intercept"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mixed = smf.mixedlm("TAC ~ time + post", data, groups=data["site_id"])
            result = mixed.fit(method="lbfgs", maxiter=500, reml=False)
    params = result.params
    conf = result.conf_int()
    for term in ["Intercept", "time", "post"]:
        rows.append(
            {
                "model": f"mixed_effects_{model_kind}",
                "term": term,
                "estimate": float(params.get(term, np.nan)),
                "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                "pvalue": float(result.pvalues.get(term, np.nan)),
                "n_observations": int(len(data)),
                "n_sites": int(data["site_id"].nunique()),
                "converged": bool(result.converged),
            }
        )
    pre_slope = float(params.get("time", np.nan))
    post_slope = float(params.get("time", np.nan) + params.get("post", np.nan))
    rows.append(
        {
            "model": f"mixed_effects_{model_kind}",
            "term": "derived_post_2008_slope",
            "estimate": post_slope,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "pvalue": np.nan,
            "n_observations": int(len(data)),
            "n_sites": int(data["site_id"].nunique()),
            "converged": bool(result.converged),
        }
    )
    rows.append(
        {
            "model": f"mixed_effects_{model_kind}",
            "term": "derived_pre_2008_slope",
            "estimate": pre_slope,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "pvalue": np.nan,
            "n_observations": int(len(data)),
            "n_sites": int(data["site_id"].nunique()),
            "converged": bool(result.converged),
        }
    )
    try:
        wls = smf.wls("TAC ~ time + post", data, weights=data["weight"]).fit(
            cov_type="cluster", cov_kwds={"groups": data["site_id"]}
        )
        conf_wls = wls.conf_int()
        for term in ["Intercept", "time", "post"]:
            rows.append(
                {
                    "model": "weighted_ols_cluster_site",
                    "term": term,
                    "estimate": float(wls.params.get(term, np.nan)),
                    "ci_low": float(conf_wls.loc[term, 0]) if term in conf_wls.index else np.nan,
                    "ci_high": float(conf_wls.loc[term, 1]) if term in conf_wls.index else np.nan,
                    "pvalue": float(wls.pvalues.get(term, np.nan)),
                    "n_observations": int(len(data)),
                    "n_sites": int(data["site_id"].nunique()),
                    "converged": True,
                }
            )
    except Exception as exc:
        state.model_notes.append(f"Weighted cluster-robust OLS failed: {exc}")
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(dirs["models"] / "mixed_effects_model_results.csv", index=False)
    return out


def leave_one_out_sensitivity(tac: pd.DataFrame, meta: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    annual = annual_site_tac(tac)
    if annual.empty:
        return pd.DataFrame(
            columns=["sensitivity", "excluded", "pre_slope", "post_slope", "slope_change"]
        )
    for site_id in sorted(annual["site_id"].unique()):
        sub = annual[annual["site_id"] != site_id].groupby("year", as_index=False)["TAC"].mean()
        pre = sub[sub["year"] < cfg.breakpoint_year]
        post = sub[sub["year"] >= cfg.breakpoint_year]
        pre_slope, _, _ = slope_with_pvalue(pre, "year", "TAC")
        post_slope, _, _ = slope_with_pvalue(post, "year", "TAC")
        rows.append(
            {
                "sensitivity": "leave_one_site_out",
                "excluded": site_id,
                "pre_slope": pre_slope,
                "post_slope": post_slope,
                "slope_change": post_slope - pre_slope
                if np.isfinite(pre_slope) and np.isfinite(post_slope)
                else np.nan,
            }
        )
    annual_meta = annual.merge(meta[["site_id", "network"]], on="site_id", how="left")
    for network in sorted(annual_meta["network"].dropna().unique()):
        sub = (
            annual_meta[annual_meta["network"] != network]
            .groupby("year", as_index=False)["TAC"]
            .mean()
        )
        pre = sub[sub["year"] < cfg.breakpoint_year]
        post = sub[sub["year"] >= cfg.breakpoint_year]
        pre_slope, _, _ = slope_with_pvalue(pre, "year", "TAC")
        post_slope, _, _ = slope_with_pvalue(post, "year", "TAC")
        rows.append(
            {
                "sensitivity": "leave_one_network_out",
                "excluded": network,
                "pre_slope": pre_slope,
                "post_slope": post_slope,
                "slope_change": post_slope - pre_slope
                if np.isfinite(pre_slope) and np.isfinite(post_slope)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def core_trend_summary(tac: pd.DataFrame, cfg: Config, label: str) -> dict[str, Any]:
    if tac.empty:
        return {
            "sensitivity": label,
            "n_tac_records": 0,
            "n_sites": 0,
            "pre_slope": np.nan,
            "pre_slope_p": np.nan,
            "post_slope": np.nan,
            "post_slope_p": np.nan,
            "slope_change": np.nan,
        }
    annual = annual_site_tac(tac).groupby("year", as_index=False)["TAC"].mean()
    pre = annual[annual["year"] < cfg.breakpoint_year]
    post = annual[annual["year"] >= cfg.breakpoint_year]
    pre_slope, pre_p, _ = slope_with_pvalue(pre, "year", "TAC")
    post_slope, post_p, _ = slope_with_pvalue(post, "year", "TAC")
    return {
        "sensitivity": label,
        "n_tac_records": int(len(tac)),
        "n_sites": int(tac["site_id"].nunique()) if len(tac) else 0,
        "pre_slope": pre_slope,
        "pre_slope_p": pre_p,
        "post_slope": post_slope,
        "post_slope_p": post_p,
        "slope_change": post_slope - pre_slope
        if np.isfinite(pre_slope) and np.isfinite(post_slope)
        else np.nan,
    }


def run_sensitivities(
    gpp: pd.DataFrame,
    monthly_primary: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    logging.info("Running coverage/window/detrending sensitivity summaries.")
    for coverage in cfg.monthly_coverage_sensitivities:
        monthly = monthly_primary if math.isclose(coverage, cfg.min_monthly_coverage) else aggregate_monthly_gpp(
            gpp, cfg, dirs, coverage=coverage
        )
        for method in ["stl", "climatology_detrended"]:
            residuals = calculate_residuals(monthly, cfg, dirs, method=method, save_primary=False)
            for window_years in cfg.tac_window_sensitivities:
                tac = calculate_rolling_tac(
                    residuals, cfg, dirs, window_years=window_years, save_primary=False
                )
                rows.append(
                    core_trend_summary(
                        tac,
                        cfg,
                        f"coverage_{coverage:.2f}_{method}_window_{window_years}yr",
                    )
                )
    long_gap_cut = 12
    low_gap_sites = set(
        meta.loc[meta["maximum_temporal_gap_months"] <= long_gap_cut, "site_id"].astype(str)
    )
    primary_resid = calculate_residuals(monthly_primary, cfg, dirs, method="stl", save_primary=False)
    primary_tac = calculate_rolling_tac(primary_resid, cfg, dirs, save_primary=False)
    rows.append(
        core_trend_summary(
            primary_tac[primary_tac["site_id"].isin(low_gap_sites)],
            cfg,
            f"exclude_sites_with_gap_gt_{long_gap_cut}_months",
        )
    )
    loo = leave_one_out_sensitivity(primary_tac, meta, cfg)
    pieces = [pd.DataFrame(rows)]
    if not loo.empty:
        pieces.append(loo)
    out = pd.concat(pieces, ignore_index=True, sort=False)
    out.to_csv(dirs["tables"] / "sensitivity_summary.csv", index=False)
    return out


def save_excel_if_possible(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except Exception as exc:
        logging.warning("Pandas Excel writer unavailable for %s: %s", path, exc)
        write_simple_xlsx(df, path)


def excel_col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell(value: Any, row_idx: int, col_idx: int) -> str:
    ref = f"{excel_col_name(col_idx)}{row_idx}"
    if pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (np.integer, int)):
        return f'<c r="{ref}"><v>{int(value)}</v></c>'
    if isinstance(value, (np.floating, float)) and np.isfinite(value):
        return f'<c r="{ref}"><v>{float(value):.15g}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_simple_xlsx(df: pd.DataFrame, path: Path) -> None:
    """Write a minimal XLSX workbook without optional Excel dependencies."""
    rows_xml = []
    header = "".join(xlsx_cell(col, 1, i) for i, col in enumerate(df.columns))
    rows_xml.append(f'<row r="1">{header}</row>')
    for r, (_, row) in enumerate(df.iterrows(), start=2):
        cells = "".join(xlsx_cell(row[col], r, c) for c, col in enumerate(df.columns))
        rows_xml.append(f'<row r="{r}">{cells}</row>')
    last_col = excel_col_name(max(len(df.columns) - 1, 0))
    dimension = f"A1:{last_col}{len(df) + 1}" if len(df.columns) else "A1"
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetData>'
        + "".join(rows_xml)
        + "</sheetData></worksheet>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Supplementary_Table" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)
    logging.info("Saved dependency-free Excel workbook: %s", path)


def create_supplementary_table(
    sites: pd.DataFrame,
    site_records: pd.DataFrame,
    period_tac: pd.DataFrame,
    slopes: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    table = sites.merge(site_records, on="site_id", how="left")
    table = table.merge(period_tac, on="site_id", how="left")
    table = table.merge(slopes, on="site_id", how="left")
    for pre, post in cfg.same_site_criteria:
        col = f"included_same_site_{pre}_{post}"
        if col not in table.columns:
            table[col] = False
    table["included_same_site_sensitivity"] = table[
        [f"included_same_site_{pre}_{post}" for pre, post in cfg.same_site_criteria[1:]]
    ].any(axis=1)
    notes = []
    for _, row in table.iterrows():
        site_notes = []
        if not bool(row.get("included_all_sites", False)):
            site_notes.append("not_group_A")
        if not bool(row.get("included_same_site_primary", False)):
            site_notes.append("not_primary_same_site")
        if pd.notna(row.get("maximum_temporal_gap_months")) and row["maximum_temporal_gap_months"] > 12:
            site_notes.append("long_gap_gt_12_months")
        notes.append(";".join(site_notes))
    table["quality_control_notes"] = notes
    preferred = [
        "site_id",
        "site_name",
        "network",
        "country",
        "latitude",
        "longitude",
        "vegetation_type",
        "permafrost_class",
        "permafrost_class_name",
        "metadata_start_year",
        "metadata_end_year",
        "first_valid_date",
        "last_valid_date",
        "total_valid_months",
        "total_valid_years",
        "valid_pre_2008_years",
        "valid_post_2008_years",
        "spans_2008",
        "maximum_temporal_gap_months",
        "fraction_missing_months",
        "pre_2008_TAC",
        "post_2008_TAC",
        "pre_2008_TAC_slope",
        "post_2008_TAC_slope",
        "TAC_slope_difference",
        "included_all_sites",
        "included_same_site_primary",
        "included_same_site_sensitivity",
        "quality_control_notes",
    ]
    cols = [c for c in preferred if c in table.columns] + [
        c for c in table.columns if c not in preferred
    ]
    table = table[cols]
    table.to_csv(dirs["tables"] / "Supplementary_Table_flux_sites.csv", index=False)
    save_excel_if_possible(table, dirs["tables"] / "Supplementary_Table_flux_sites.xlsx")
    return table


def save_fig(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_site_map(ax: plt.Axes, sites: pd.DataFrame, cfg: Config, title: str) -> None:
    with rasterio.open(cfg.permafrost_raster) as src:
        scale = max(1, int(max(src.width, src.height) / 1000))
        arr = src.read(1, out_shape=(1, src.height // scale, src.width // scale), resampling=Resampling.nearest)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    mask = np.isin(arr, list(cfg.valid_landcover_classes)).astype(float)
    mask[mask == 0] = np.nan
    ax.imshow(mask, extent=extent, origin="upper", cmap="Greys", alpha=0.35, aspect="auto")
    if not sites.empty:
        ax.scatter(
            sites["longitude"],
            sites["latitude"],
            s=28,
            c="#d95f02",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(25, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)


def make_figures(
    sites: pd.DataFrame,
    monthly: pd.DataFrame,
    site_records: pd.DataFrame,
    tac: pd.DataFrame,
    slopes: pd.DataFrame,
    trajectories: pd.DataFrame,
    model_results: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> None:
    logging.info("Creating figures.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1.1, 1.4]})
    plot_site_map(axes[0], sites, cfg, "Permafrost flux sites")
    timeline = site_records.merge(sites[["site_id", "latitude"]], on="site_id", how="left")
    timeline = timeline.sort_values(["latitude", "first_valid_date"], ascending=[False, True]).reset_index(drop=True)
    for i, row in timeline.iterrows():
        axes[1].plot([row["first_valid_date"], row["last_valid_date"]], [i, i], color="#3b6fb6", lw=2)
    axes[1].axvline(pd.Timestamp(f"{cfg.breakpoint_year}-01-01"), color="black", ls="--", lw=1)
    axes[1].set_yticks(range(len(timeline)))
    axes[1].set_yticklabels(timeline["site_id"], fontsize=7)
    axes[1].set_title("Monthly GPP record coverage")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Site")
    fig.tight_layout()
    save_fig(fig, dirs["figures"] / "Figure_1_site_availability_record_coverage")

    same_ids = set(site_records.loc[site_records["included_same_site_primary"], "site_id"])
    same_slopes = slopes[slopes["site_id"].isin(same_ids)].dropna(
        subset=["pre_2008_TAC_slope", "post_2008_TAC_slope"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for _, row in same_slopes.iterrows():
        axes[0, 0].plot(
            [0, 1],
            [row["pre_2008_TAC_slope"], row["post_2008_TAC_slope"]],
            color="#555555",
            alpha=0.55,
        )
    axes[0, 0].scatter(np.zeros(len(same_slopes)), same_slopes["pre_2008_TAC_slope"], color="#377eb8")
    axes[0, 0].scatter(np.ones(len(same_slopes)), same_slopes["post_2008_TAC_slope"], color="#e41a1c")
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_xticks([0, 1], ["Pre-2008", "Post-2008"])
    axes[0, 0].set_ylabel("Site TAC slope")
    axes[0, 0].set_title("Paired site slopes")
    diff = same_slopes["TAC_slope_difference"].dropna()
    axes[0, 1].hist(diff, bins=min(12, max(3, len(diff))), color="#7570b3", edgecolor="white")
    axes[0, 1].axvline(0, color="black", lw=0.8)
    axes[0, 1].set_title("Post-minus-pre slope")
    axes[0, 1].set_xlabel("Slope difference")
    plot_site_map(axes[1, 0], sites[sites["site_id"].isin(same_ids)], cfg, "Same-site subset")
    same_tac = tac[tac["site_id"].isin(same_ids)]
    if not same_tac.empty:
        annual = annual_site_tac(same_tac).groupby("year", as_index=False)["TAC"].mean()
        axes[1, 1].plot(annual["year"], annual["TAC"], color="black", marker="o")
    axes[1, 1].axvline(cfg.breakpoint_year, color="black", ls="--", lw=1)
    axes[1, 1].set_title("Same-site TAC trajectory")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("TAC")
    fig.tight_layout()
    save_fig(fig, dirs["figures"] / "Figure_2_same_site_validation")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {
        "all_site_unbalanced": "#1b9e77",
        "balanced_resampling": "#d95f02",
        "same_site_subset": "#7570b3",
    }
    labels = {
        "all_site_unbalanced": "All sites",
        "balanced_resampling": "Balanced resampling",
        "same_site_subset": "Same-site subset",
    }
    for name, sub in trajectories.groupby("trajectory"):
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["mean_TAC"], marker="o", color=colors.get(name), label=labels.get(name, name))
        ax.fill_between(sub["year"], sub["ci_low"], sub["ci_high"], color=colors.get(name), alpha=0.18)
    ax.axvline(cfg.breakpoint_year, color="black", ls="--", lw=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean TAC")
    ax.set_title("Aggregate TAC trajectories")
    if not trajectories.empty:
        ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, dirs["figures"] / "Figure_3_aggregate_TAC_trajectories")

    fig, ax = plt.subplots(figsize=(6, 4))
    model_plot = model_results[
        model_results["term"].isin(["time", "post", "derived_post_2008_slope", "derived_pre_2008_slope"])
        & model_results["model"].str.startswith("mixed_effects", na=False)
    ].copy()
    if not model_plot.empty:
        x = np.arange(len(model_plot))
        ax.bar(x, model_plot["estimate"], color="#4c78a8")
        has_ci = model_plot["ci_low"].notna() & model_plot["ci_high"].notna()
        ax.errorbar(
            x[has_ci],
            model_plot.loc[has_ci, "estimate"],
            yerr=[
                model_plot.loc[has_ci, "estimate"] - model_plot.loc[has_ci, "ci_low"],
                model_plot.loc[has_ci, "ci_high"] - model_plot.loc[has_ci, "estimate"],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )
        ax.set_xticks(x, model_plot["term"], rotation=30, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Slope estimate")
    ax.set_title("Segmented mixed-effects model")
    fig.tight_layout()
    save_fig(fig, dirs["figures"] / "Figure_4_mixed_effects_model")


def save_group_membership(site_records: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> None:
    rows = [
        {
            "group": "A_all_eligible_sites",
            "criterion": f">={cfg.min_total_years} valid years with >={cfg.min_valid_months_per_year} months/year",
            "n_sites": int(site_records["included_all_sites"].sum()),
            "site_ids": ";".join(sorted(site_records.loc[site_records["included_all_sites"], "site_id"])),
        }
    ]
    for pre, post in cfg.same_site_criteria:
        col = f"included_same_site_{pre}_{post}"
        rows.append(
            {
                "group": f"B_same_site_{pre}_{post}",
                "criterion": f">={pre} pre-2008 valid years and >={post} post-2008 valid years",
                "n_sites": int(site_records[col].sum()),
                "site_ids": ";".join(sorted(site_records.loc[site_records[col], "site_id"])),
            }
        )
    pd.DataFrame(rows).to_csv(dirs["tables"] / "site_group_membership.csv", index=False)


def save_run_log(cfg: Config, state: RunState, dirs: dict[str, Path], summaries: dict[str, Any]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(CODE_DIR / "R3.08_flux_gpp_TAC.py"),
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(cfg).items()
        },
        "selected_columns": state.selected_columns,
        "input_summaries": state.input_summaries,
        "warnings": state.warnings,
        "excluded_sites": state.excluded_sites,
        "model_notes": state.model_notes,
        "package_versions": state.package_versions,
        "result_summaries": summaries,
    }
    with open(dirs["tables"] / "run_log.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    cfg = configure(args)
    setup_logging(cfg.output_dir)
    dirs = ensure_dirs(cfg)
    state = RunState()
    state.package_versions = package_versions()
    logging.info("Starting Reviewer 3.2 flux-tower TAC validation.")
    logging.info("Output directory: %s", cfg.output_dir)

    sites_all = load_site_metadata(cfg, state)
    sites_pf = sample_sites_in_permafrost(sites_all, cfg, state, dirs)
    if sites_pf.empty:
        raise RuntimeError("No flux sites were found in configured valid permafrost raster classes.")
    gpp = load_gpp_for_sites(cfg, state, sites_pf)
    if gpp.empty:
        raise RuntimeError("No valid GPP observations matched the selected permafrost sites.")
    monthly = aggregate_monthly_gpp(gpp, cfg, dirs)
    site_records = build_site_record_table(monthly, cfg)
    site_records.to_csv(dirs["tables"] / "site_record_summary.csv", index=False)
    save_group_membership(site_records, cfg, dirs)

    group_a = set(site_records.loc[site_records["included_all_sites"], "site_id"])
    monthly_a = monthly[monthly["site_id"].isin(group_a)].copy()
    logging.info("Group A eligible sites: %s", len(group_a))
    if not group_a:
        state.warn(
            "No sites met Group A strict eligibility under the configured "
            f"{cfg.min_total_years} valid years and "
            f"{cfg.min_valid_months_per_year} valid months/year criteria. "
            "Primary rolling TAC outputs are therefore empty; inspect "
            "site_record_summary.csv and sensitivity_summary.csv."
        )
    residuals = calculate_residuals(monthly_a, cfg, dirs, method="stl", save_primary=True)
    tac = calculate_rolling_tac(residuals, cfg, dirs, save_primary=True)
    slopes = site_level_slopes(tac, cfg)
    slopes.to_csv(dirs["tables"] / "site_level_TAC_slopes.csv", index=False)
    period_tac = calculate_period_tac(residuals, site_records, cfg, dirs)
    same_site_summary = same_site_tests(slopes, site_records, cfg, dirs)
    trajectories, common_subsets = aggregate_trajectories(tac, site_records, cfg, dirs)
    model_results = fit_segmented_models(tac, cfg, dirs, state)
    sensitivity = run_sensitivities(gpp, monthly, sites_pf.merge(site_records, on="site_id", how="left"), cfg, dirs)

    supplementary = create_supplementary_table(
        sites_pf, site_records, period_tac, slopes, cfg, dirs
    )
    make_figures(
        sites_pf,
        monthly,
        site_records,
        tac,
        slopes,
        trajectories,
        model_results,
        cfg,
        dirs,
    )

    summaries = {
        "n_permafrost_sites": int(len(sites_pf)),
        "n_monthly_records": int(len(monthly)),
        "n_group_A_sites": int(len(group_a)),
        "n_rolling_TAC_records": int(len(tac)),
        "n_same_site_primary": int(site_records["included_same_site_primary"].sum()),
        "same_site_summary": same_site_summary,
        "n_sensitivity_rows": int(len(sensitivity)),
        "n_common_site_subset_rows": int(len(common_subsets)),
        "n_supplementary_rows": int(len(supplementary)),
    }
    save_run_log(cfg, state, dirs, summaries)
    logging.info("Finished. Key outputs are in %s", cfg.output_dir)


if __name__ == "__main__":
    main()
