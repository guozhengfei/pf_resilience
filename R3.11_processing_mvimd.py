"""
Process ERA5 annual vertically integrated moisture divergence for R3.4.

Goal:
  - Read annual global ERA5 MVIMD GeoTIFFs, 1990–2023
  - Extract pan-Arctic/permafrost-domain mean MVIMD and MFC
  - Convert MVIMD to moisture-flux convergence:
        MFC = -1 × MVIMD
  - Convert kg m-2 s-1 to mm day-1
  - Calculate annual anomalies
  - Calculate pre- and post-2008 trends
  - Plot one clean pan-Arctic/permafrost-domain MFC time series

Interpretation:
  ERA5 MVIMD > 0 = moisture-flux divergence / drying tendency
  ERA5 MVIMD < 0 = moisture-flux convergence / moistening tendency

  MFC = -MVIMD
  MFC > 0 = moisture convergence / moistening tendency
  MFC < 0 = moisture divergence / drying tendency
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("qtAgg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats
import pwlf


matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,
})


# ============================================================
# 1. User settings
# ============================================================

MVIMD_DIR = Path(
    "/Volumes/Zhengfei_01/Project 2 pf resilience/1_Input/"
    "era5_mvimd_annual_global_1990_2024"
)

# Permafrost mask raster.
# Update this path if your mask is different.
PERMAFROST_MASK_RASTER = Path(
    "/Volumes/Zhengfei_01/Project 2 pf resilience/1_Input/"
    "landcover_export_2010_5km.tif"
)

OUTPUT_DIR = Path(
    "/Volumes/Zhengfei_01/Project 2 pf resilience/2_Output/"
    "era5_mvimd_r34_pan_arctic"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1990
END_YEAR = 2023

# Periods for R3.4.
PRE_PERIOD = (1990, 2008)
POST_PERIOD = (2008, 2023)

# Piecewise linear fit settings.
PWLF_BREAK_YEAR = 2008
PWLF_FORCE_POINT = (2008.0, 0.07)

# Baseline for anomalies.
ANOMALY_BASELINE = (1990, 2023)

PERMAFROST_VALUES = [1,2,3,4,5,6,7,8,9,10,11]

USE_MASK = True


# ============================================================
# 2. Helper functions
# ============================================================

def parse_year_from_filename(path: Path) -> int:
    """Extract 4-digit year from filename."""
    m = re.search(r"(19\d{2}|20\d{2})", path.name)
    if m is None:
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    return int(m.group(1))


def find_annual_tifs(mvimd_dir: Path, start_year: int, end_year: int) -> dict:
    """Find annual GeoTIFFs and return {year: path}."""
    tif_paths = sorted(
        list(mvimd_dir.glob("*.tif")) +
        list(mvimd_dir.glob("*.tiff")) +
        list(mvimd_dir.glob("*.TIF")) +
        list(mvimd_dir.glob("*.TIFF"))
    )

    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No GeoTIFF files found in: {mvimd_dir}")

    year_to_path = {}
    for p in tif_paths:
        year = parse_year_from_filename(p)
        if start_year <= year <= end_year:
            year_to_path[year] = p

    missing = [y for y in range(start_year, end_year + 1) if y not in year_to_path]
    if missing:
        warnings.warn(f"Missing annual GeoTIFFs for years: {missing}")

    return dict(sorted(year_to_path.items()))


def get_lat_weights(src) -> np.ndarray:
    """
    Create 2D area weights proportional to cos(latitude).
    Appropriate for regular lat-lon grids.
    """
    height = src.height
    width = src.width
    transform = src.transform

    rows = np.arange(height)

    # Latitude of pixel centers.
    lats = transform.f + (rows + 0.5) * transform.e

    weights_1d = np.cos(np.deg2rad(lats))
    weights_1d = np.clip(weights_1d, 0, None)

    weights_2d = np.repeat(weights_1d[:, None], width, axis=1)
    return weights_2d.astype("float64")


def read_reproject_mask(mask_path: Path, template_src) -> np.ndarray:
    """
    Read permafrost mask raster and reproject/resample it to match ERA5 grid.
    Returns boolean mask.
    """
    with rasterio.open(mask_path) as mask_src:
        mask_data = mask_src.read(1)

        dst = np.full(
            (template_src.height, template_src.width),
            fill_value=0,
            dtype=mask_data.dtype
        )

        reproject(
            source=mask_data,
            destination=dst,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=template_src.transform,
            dst_crs=template_src.crs,
            resampling=Resampling.nearest
        )

    mask_bool = np.isin(dst, PERMAFROST_VALUES)

    if mask_bool.sum() == 0:
        raise ValueError(
            "The permafrost mask contains zero valid pixels after reprojection. "
            "Please check PERMAFROST_VALUES or the mask raster."
        )

    return mask_bool


def weighted_mean(arr: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted mean ignoring NaNs."""
    valid = np.isfinite(arr) & mask & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return np.nansum(arr[valid] * weights[valid]) / np.nansum(weights[valid])


def weighted_se(arr: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    """Weighted spatial standard error of the mean, ignoring NaNs."""
    valid = np.isfinite(arr) & mask & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        return np.nan

    values = arr[valid]
    w = weights[valid]
    mean = np.average(values, weights=w)
    variance = np.average((values - mean) ** 2, weights=w)
    effective_n = (w.sum() ** 2) / np.sum(w ** 2)
    return np.sqrt(variance / effective_n)*10


def lat_lon_grid(src):
    """Longitude/latitude grid from a raster source."""
    rows = np.arange(src.height)
    cols = np.arange(src.width)
    xs = src.transform.c + (cols + 0.5) * src.transform.a
    ys = src.transform.f + (rows + 0.5) * src.transform.e
    return np.meshgrid(xs, ys)


def circular_boundary():
    theta = np.linspace(0, 2 * np.pi, 100)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return mpath.Path(verts * 0.5 + [0.5, 0.5])


def linear_trend(years, values):
    """
    Linear trend.
    Returns slope per year, slope per decade, p-value, r-value, n.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())

    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n

    res = stats.linregress(x[valid], y[valid])
    return res.slope, res.slope * 10.0, res.pvalue, res.rvalue, n


def period_trend(df: pd.DataFrame, value_col: str, period: tuple):
    """Calculate trend in a specific period."""
    y0, y1 = period
    sub = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()
    return linear_trend(sub["year"], sub[value_col])


def pwlf_segment_fit(years, values, break_year=PWLF_BREAK_YEAR, force_point=PWLF_FORCE_POINT):
    """
    Two-segment piecewise linear fit forced through one point.

    The default forces the fitted curve through (2008, 0.1).
    Returns the fitted model, x/y values for plotting, and slopes.
    """
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 4:
        raise ValueError("Need at least 4 valid points for a forced two-segment pwlf fit.")

    x_fit = x[valid]
    y_fit = y[valid]
    breaks = np.array([x_fit.min(), float(break_year), x_fit.max()])
    x_c = np.array([force_point[0]], dtype=float)
    y_c = np.array([force_point[1]], dtype=float)

    model = pwlf.PiecewiseLinFit(x_fit, y_fit)
    model.fit_with_breaks_force_points(breaks, x_c, y_c)

    x_hat = np.linspace(x_fit.min(), x_fit.max(), 300)
    y_hat = model.predict(x_hat)
    slopes = np.asarray(model.slopes, dtype=float)

    return {
        "model": model,
        "x": x_hat,
        "y": y_hat,
        "breaks": breaks,
        "slopes_per_year": slopes,
        "slopes_per_decade": slopes * 10.0,
        "forced_x": force_point[0],
        "forced_y": force_point[1],
    }


# ============================================================
# 3. Read annual GeoTIFFs and calculate pan-Arctic/permafrost mean
# ============================================================

year_to_path = find_annual_tifs(MVIMD_DIR, START_YEAR, END_YEAR)

first_year = sorted(year_to_path.keys())[0]
with rasterio.open(year_to_path[first_year]) as src0:
    weights = get_lat_weights(src0)
    grid_longitudes, grid_latitudes = lat_lon_grid(src0)

    if USE_MASK:
        domain_mask = read_reproject_mask(PERMAFROST_MASK_RASTER, src0)
        domain_name = "Permafrost domain"
    else:
        domain_mask = np.ones((src0.height, src0.width), dtype=bool)
        domain_name = "Global valid pixels"

records = []
mfc_sum = None
mfc_count = None

for year, tif_path in year_to_path.items():
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        # Remove unrealistic values if any appear.
        # ERA5 MVIMD is typically small, kg m-2 s-1.
        arr[np.abs(arr) > 1] = np.nan

        # MVIMD:
        # positive = moisture-flux divergence
        # negative = moisture-flux convergence
        mvimd_kg_m2_s = arr

        # MFC:
        # positive = moisture convergence
        # negative = moisture divergence
        mfc_kg_m2_s = -1.0 * mvimd_kg_m2_s

        # Convert kg m-2 s-1 to mm day-1.
        # 1 kg m-2 water = 1 mm water.
        mvimd_mm_day = mvimd_kg_m2_s * 86400.0
        mfc_mm_day = mfc_kg_m2_s * 86400.0
        # plt.figure();plt.imshow(domain_mask)
        if mfc_sum is None:
            mfc_sum = np.zeros_like(mfc_mm_day, dtype="float64")
            mfc_count = np.zeros_like(mfc_mm_day, dtype="uint16")

        # Spatial map intentionally uses all valid ERA5 land/ocean pixels.
        # The permafrost mask is only used for the anomaly time series above.
        valid_map = np.isfinite(mfc_mm_day)
        mfc_sum[valid_map] += mfc_mm_day[valid_map]
        mfc_count[valid_map] += 1

        records.append({
            "year": year,
            "domain": domain_name,
            "MVIMD_kg_m2_s": weighted_mean(mvimd_kg_m2_s, domain_mask, weights),
            "MFC_kg_m2_s": weighted_mean(mfc_kg_m2_s, domain_mask, weights),
            "MVIMD_mm_day": weighted_mean(mvimd_mm_day, domain_mask, weights),
            "MFC_mm_day": weighted_mean(mfc_mm_day, domain_mask, weights),
            "MVIMD_mm_day_se": weighted_se(mvimd_mm_day, domain_mask, weights),
            "MFC_mm_day_se": weighted_se(mfc_mm_day, domain_mask, weights),
            "n_valid_pixels": int((np.isfinite(arr) & domain_mask).sum())
        })

df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
mean_mfc_map = np.full_like(mfc_sum, np.nan, dtype="float64")
mean_mfc_map[mfc_count > 0] = mfc_sum[mfc_count > 0] / mfc_count[mfc_count > 0]

# Calculate anomalies relative to baseline.
b0, b1 = ANOMALY_BASELINE
base = df[(df["year"] >= b0) & (df["year"] <= b1)]

for col in ["MVIMD_mm_day", "MFC_mm_day"]:
    df[col + "_anom"] = df[col] - base[col].mean()

df["MVIMD_mm_day_anom_se"] = df["MVIMD_mm_day_se"]
df["MFC_mm_day_anom_se"] = df["MFC_mm_day_se"]

out_csv = OUTPUT_DIR / "pan_arctic_permafrost_ERA5_MVIMD_MFC_annual_1990_2023.csv"
df.to_csv(out_csv, index=False)

print("Saved annual time series:")
print(out_csv)
print(df.head())


# ============================================================
# 4. Trend summary
# ============================================================

trend_rows = []

for value_col in [
    "MVIMD_mm_day",
    "MFC_mm_day",
    "MVIMD_mm_day_anom",
    "MFC_mm_day_anom"
]:
    pre_slope_y, pre_slope_dec, pre_p, pre_r, pre_n = period_trend(df, value_col, PRE_PERIOD)
    post_slope_y, post_slope_dec, post_p, post_r, post_n = period_trend(df, value_col, POST_PERIOD)

    trend_rows.append({
        "domain": domain_name,
        "variable": value_col,
        "pre_period": f"{PRE_PERIOD[0]}-{PRE_PERIOD[1]}",
        "pre_slope_per_year": pre_slope_y,
        "pre_slope_per_decade": pre_slope_dec,
        "pre_p": pre_p,
        "pre_r": pre_r,
        "pre_n": pre_n,
        "post_period": f"{POST_PERIOD[0]}-{POST_PERIOD[1]}",
        "post_slope_per_year": post_slope_y,
        "post_slope_per_decade": post_slope_dec,
        "post_p": post_p,
        "post_r": post_r,
        "post_n": post_n,
        "post_minus_pre_slope_per_decade": post_slope_dec - pre_slope_dec
    })

trend_df = pd.DataFrame(trend_rows)

out_trend_csv = OUTPUT_DIR / "pan_arctic_permafrost_MVIMD_MFC_pre_post_trends.csv"
trend_df.to_csv(out_trend_csv, index=False)

print("Saved trend summary:")
print(out_trend_csv)
print(trend_df)


# ============================================================
# 5. Plot pan-Arctic/permafrost MFC time series and mean map
# ============================================================

fig = plt.figure(figsize=(8, 2.9))
ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
ax = fig.add_subplot(1, 2, 2)

ax.plot(
    df["year"],
    df["MFC_mm_day_anom"],
    color="black",
    lw=2,
    label="MFC anomaly"
)
ax.fill_between(
    df["year"],
    df["MFC_mm_day_anom"] - df["MFC_mm_day_anom_se"],
    df["MFC_mm_day_anom"] + df["MFC_mm_day_anom_se"],
    color="black",
    alpha=0.18,
    linewidth=0,
    label="Spatial SE"
)

ax.axvline(2008, ls="--", color="0.45", lw=1.2)
ax.axhline(0, ls="-", color="0.75", lw=0.8)

# Add forced piecewise linear fit.
pwlf_fit = pwlf_segment_fit(df["year"], df["MFC_mm_day_anom"])
ax.plot(
    pwlf_fit["x"],
    pwlf_fit["y"],
    ls="--",
    lw=1.8,
    color="tab:red",
)

pwlf_summary = pd.DataFrame(
    [
        {
            "domain": domain_name,
            "variable": "MFC_mm_day_anom",
            "segment": f"{PRE_PERIOD[0]}-{PWLF_BREAK_YEAR}",
            "break_year": PWLF_BREAK_YEAR,
            "forced_x": pwlf_fit["forced_x"],
            "forced_y": pwlf_fit["forced_y"],
            "slope_per_year": pwlf_fit["slopes_per_year"][0],
            "slope_per_decade": pwlf_fit["slopes_per_decade"][0],
        },
        {
            "domain": domain_name,
            "variable": "MFC_mm_day_anom",
            "segment": f"{PWLF_BREAK_YEAR}-{POST_PERIOD[1]}",
            "break_year": PWLF_BREAK_YEAR,
            "forced_x": pwlf_fit["forced_x"],
            "forced_y": pwlf_fit["forced_y"],
            "slope_per_year": pwlf_fit["slopes_per_year"][1],
            "slope_per_decade": pwlf_fit["slopes_per_decade"][1],
        },
    ]
)
out_pwlf_csv = OUTPUT_DIR / "pan_arctic_permafrost_MFC_pwlf_forced_2008_0p1.csv"
pwlf_summary.to_csv(out_pwlf_csv, index=False)

ax.set_title("MFC anomaly over the permafrost domain")
ax.set_xlabel("Year")
ax.set_ylabel("MFC anomaly (mm day$^{-1}$)")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

map_values = mean_mfc_map
limit = np.nanpercentile(np.abs(map_values), 95)
mesh = ax_map.pcolormesh(
    grid_longitudes,
    grid_latitudes,
    map_values,
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    transform=ccrs.PlateCarree()
)
domain_mask = read_reproject_mask(PERMAFROST_MASK_RASTER, src0)
# plt.figure();plt.imshow(permafrost_extent)
permafrost_extent = np.where(domain_mask, 1.0, np.nan)

ax_map.coastlines(linewidth=1)
ax_map.set_extent([-180, 180, 15, 90], crs=ccrs.PlateCarree())
ax_map.set_boundary(circular_boundary(), transform=ax_map.transAxes)
ax_map.set_title(f"Mean MFC ({START_YEAR}-{END_YEAR})")
cbar = fig.colorbar(mesh, ax=ax_map, orientation="horizontal", fraction=0.05, pad=0.05)
cbar.set_label("MFC (mm day$^{-1}$)")
domain_mask = read_reproject_mask(PERMAFROST_MASK_RASTER, src0)


permafrost_extent = np.where(domain_mask, 1.0, np.nan)

ax_map.pcolormesh(grid_longitudes,grid_latitudes,permafrost_extent,cmap=matplotlib.colors.ListedColormap(["0.18"]),alpha=0.2,transform=ccrs.PlateCarree(),zorder=5)

out_png = OUTPUT_DIR / "pan_arctic_permafrost_MFC_anomaly_timeseries_mean_map.png"
fig.tight_layout()
fig.savefig(out_png, dpi=600)

# ============================================================
# 6. Optional wording helper
# ============================================================

mfc_trend = trend_df[trend_df["variable"] == "MFC_mm_day_anom"].iloc[0]

print("\nSuggested interpretation:")
print(
    f"Pre-2008 MFC trend: {mfc_trend['pre_slope_per_decade']:.4f} "
    f"mm day-1 decade-1, p = {mfc_trend['pre_p']:.3f}"
)
print(
    f"Post-2008 MFC trend: {mfc_trend['post_slope_per_decade']:.4f} "
    f"mm day-1 decade-1, p = {mfc_trend['post_p']:.3f}"
)

if mfc_trend["post_slope_per_decade"] < 0:
    print(
        "Post-2008 MFC decreases, indicating weakened moisture convergence "
        "or enhanced moisture divergence, consistent with atmospheric drying."
    )
else:
    print(
        "Post-2008 MFC does not decrease. Interpret the moisture-transport "
        "pathway cautiously and describe it as partial or regionally variable support."
    )
