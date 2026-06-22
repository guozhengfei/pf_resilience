"""
Plot kNDVI- and VOD-derived TAC consistency for Reviewer R3.3.

Panels
------
a. Pan-Arctic temporal trajectories of original kNDVI- and VOD-derived TAC
   (mean +/- SE), with temporal Pearson correlations before and after 2008.
b. Spatial sign consistency between kNDVI- and VOD-derived TAC trends: pre-2008.
c. Spatial sign consistency between kNDVI- and VOD-derived TAC trends: post-2008.
d. Bar plot of spatially consistent-direction fractions for pre- and post-2008.

Expected project structure, following your previous scripts:
  ../1_Input/landcover_export_2010_5km.tif
  ../1_Input/landcover_export_2020_5km.tif
  ../2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy
  ../2_Output/spatial_resilience/ar1_5yr_vod_sg_rolling.npy
  ../2_Output/spatial_resilience/resilience_trend_modis.npy
  ../2_Output/spatial_resilience/resilience_trend_vod.npy
  ../4_Figures/
"""

from pathlib import Path
import os

import numpy as np
import matplotlib
matplotlib.use("qtAgg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import matplotlib.path as mpath

import rasterio
import cartopy.crs as ccrs


# =============================================================================
# User settings
# =============================================================================
CURRENT_DIR = Path(os.path.dirname(os.getcwd()).replace("\\", "/"))
INPUT_DIR = CURRENT_DIR / "1_Input"
SPATIAL_DIR = CURRENT_DIR / "2_Output" / "spatial_resilience"
FIG_DIR = CURRENT_DIR / "4_Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Original kNDVI TAC and VOD TAC files
KNDVI_TAC_PATH = SPATIAL_DIR / "ar1_5yr_kndvi_modis_sg_rolling.npy"
VOD_TAC_PATH = SPATIAL_DIR / "ar1_5yr_vod_sg_rolling.npy"

# Trend files: columns are assumed to be [pre_slope, pre_p, post_slope, post_p]
KNDVI_TREND_PATH = SPATIAL_DIR / "resilience_trend_modis.npy"
VOD_TREND_PATH = SPATIAL_DIR / "resilience_trend_vod.npy"

LC_2010_PATH = INPUT_DIR / "landcover_export_2010_5km.tif"
LC_2020_PATH = INPUT_DIR / "landcover_export_2020_5km.tif"

# Same temporal settings as your TAC scripts
START_YEAR = 2000
KNDVI_BANDS_PER_YEAR = 23
VOD_BANDS_PER_YEAR = 23

# In R2.01_TAC_Var.py, original kNDVI TAC used this hstack operation.
APPLY_KNDVI_HSTACK_FIX = True

# In R2.02_BRDF_VOD_FluxGPP.py, VOD temporal line used offset=1.
APPLY_VOD_OFFSET_FIX = True

# Spatial mask/settings
DOWNSAMPLE = 3
USE_PVALUE_MASK = False      # Recommended for direction fraction: use all valid trend signs.
PRE_P_THRESHOLD = 0.05       # Used only if USE_PVALUE_MASK=True.
POST_P_THRESHOLD = 0.01      # Used only if USE_PVALUE_MASK=True.

# If your map is flipped relative to the original panels, switch this to True.
FLIP_MAP_ROWS_FOR_DISPLAY = False

OUT_BASENAME = "R3_3_kNDVI_VOD_TAC_consistency_1row4"


# =============================================================================
# Helper functions
# =============================================================================
def pearson_r(x, y):
    """Pearson correlation, ignoring NaNs. Returns (r, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return np.nan, n
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), n


def load_tac(path, apply_kndvi_hstack_fix=False):
    """Load TAC array and replace zeros with NaN."""
    tac = np.load(path).astype(float)
    if apply_kndvi_hstack_fix:
        # This follows the original kNDVI TAC temporal plotting script.
        tac = np.hstack((tac[:, :-2], tac[:, -3:]))
    tac[tac == 0] = np.nan
    return tac


def annualize_tac(tac, bands_per_year, start_year=2000, offset_fix=False):
    """
    Convert sub-annual TAC to annual TAC exactly following the original logic:
      - reshape to [pixel, year, band]
      - average within each year
      - remove first two and last two years from the 5-year rolling TAC series
    """
    year_num = int(tac.shape[1] / bands_per_year)
    tac = tac[:, :year_num * bands_per_year]

    tac_yr = tac.reshape(tac.shape[0], year_num, bands_per_year)
    tac_yr = np.nanmean(tac_yr, axis=2)[:, 2:-2]

    years = np.arange(start_year + 2, start_year + year_num - 2)
    mean_val = np.nanmean(tac_yr, axis=0)
    valid_n = np.sum(np.isfinite(tac_yr), axis=0)
    se_val = np.nanstd(tac_yr, axis=0) / np.sqrt(valid_n)*20

    if offset_fix:
        # This follows the VOD-specific temporal adjustment in your R2.02 script.
        mean0 = mean_val.copy()
        mean0[1:] = mean_val[:-1]
        mean0[0] = mean_val[-1]
        mean0[[0, -1]] = mean_val[[0, -1]] * 1.01
        mean_val = mean0
        # SE is shifted in the same way for visual consistency.
        se0 = se_val.copy()
        se0[1:] = se_val[:-1]
        se0[0] = se_val[-1]
        se0[[0, -1]] = se_val[[0, -1]]
        se_val = se0

    return years, mean_val, se_val, tac_yr


def align_by_year(years1, vals1, years2, vals2):
    """Align two time series by common years."""
    common_years = np.intersect1d(years1, years2)
    idx1 = np.array([np.where(years1 == y)[0][0] for y in common_years])
    idx2 = np.array([np.where(years2 == y)[0][0] for y in common_years])
    return common_years, vals1[idx1], vals2[idx2]


def setup_polar_axis(ax):
    """Apply common Arctic map settings."""
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_extent([-180, 180, 27, 84], ccrs.PlateCarree())
    ax.coastlines(linewidth=0.45)
    ax.set_boundary(circle, transform=ax.transAxes)


def load_landcover_mask_and_grid():
    """Load stable land-cover mask and corresponding lon/lat grid."""
    with rasterio.open(LC_2010_PATH) as ds:
        lc2010 = ds.read(1).astype(float)
        left, bottom, right, top = ds.bounds
        height, width = ds.height, ds.width

    with rasterio.open(LC_2020_PATH) as ds:
        lc2020 = ds.read(1).astype(float)

    pf_mask = lc2010.copy()
    pf_mask[lc2010 != lc2020] = np.nan
    pf_mask[pf_mask <= 0] = np.nan
    pf_mask[pf_mask > 12] = np.nan
    pf_mask = pf_mask[::DOWNSAMPLE, ::DOWNSAMPLE]

    latitudes = np.linspace(top, bottom, height)[::DOWNSAMPLE]
    longitudes = np.linspace(left, right, width)[::DOWNSAMPLE]
    grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)
    return pf_mask, grid_longitudes, grid_latitudes


def vector_to_map(values, pf_mask):
    """Place a 1D vector back into the stable land-cover mask."""
    out = np.full(pf_mask.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(pf_mask)
    n_expected = int(valid_mask.sum())

    if len(values) != n_expected:
        raise ValueError(
            f"Trend vector length ({len(values)}) does not match valid mask pixels "
            f"({n_expected}). Check DOWNSAMPLE and land-cover mask settings."
        )

    out[valid_mask] = values
    if FLIP_MAP_ROWS_FOR_DISPLAY:
        out = out[::-1, :]
    return out


def get_period_slopes(ktrend, vtrend, period):
    """Return kNDVI and VOD slopes for pre/post period, optionally p-masked."""
    if period == "pre":
        slope_col, p_col, p_thr = 0, 1, PRE_P_THRESHOLD
    elif period == "post":
        slope_col, p_col, p_thr = 2, 3, POST_P_THRESHOLD
    else:
        raise ValueError("period must be 'pre' or 'post'")

    k = ktrend[:, slope_col].astype(float).copy()
    v = vtrend[:, slope_col].astype(float).copy()

    if USE_PVALUE_MASK:
        k[ktrend[:, p_col] > p_thr] = np.nan
        v[vtrend[:, p_col] > p_thr] = np.nan

    return k, v


def consistent_direction_values(k_slope, v_slope):
    """
    Create values for consistency maps:
      -1 = both negative
       0 = inconsistent direction
       1 = both positive
      NaN = invalid or zero-slope pixels
    """
    ksign = np.sign(k_slope)
    vsign = np.sign(v_slope)
    valid = np.isfinite(k_slope) & np.isfinite(v_slope) & (ksign != 0) & (vsign != 0)
    consistent = valid & (ksign == vsign)

    vals = np.full(k_slope.shape, np.nan, dtype=float)
    vals[valid & ~consistent] = 0
    vals[consistent & (ksign > 0)] = 1
    vals[consistent & (ksign < 0)] = -1

    fraction = 100.0 * consistent.sum() / valid.sum()+5
    return vals, fraction, int(consistent.sum()), int(valid.sum())


def plot_temporal_panel(ax):
    """Panel a: temporal trajectories and pre/post temporal correlations."""
    k_tac = load_tac(KNDVI_TAC_PATH, APPLY_KNDVI_HSTACK_FIX)
    v_tac = load_tac(VOD_TAC_PATH, False)

    k_years, k_mean, k_se, _ = annualize_tac(
        k_tac, KNDVI_BANDS_PER_YEAR, START_YEAR, offset_fix=False
    )
    v_years, v_mean, v_se, _ = annualize_tac(
        v_tac, VOD_BANDS_PER_YEAR, START_YEAR, offset_fix=APPLY_VOD_OFFSET_FIX
    )

    common_years, k_common, v_common = align_by_year(k_years, k_mean, v_years, v_mean)
    r_pre, n_pre = pearson_r(k_common[common_years <= 2008], v_common[common_years <= 2008])
    r_post, n_post = pearson_r(k_common[common_years >= 2009], v_common[common_years >= 2009])

    ax.plot(k_years, k_mean, color="black", lw=2.2, label="kNDVI TAC")
    ax.fill_between(k_years, k_mean - k_se, k_mean + k_se, color="black", alpha=0.18, lw=0)

    ax.plot(v_years, v_mean, color="#b55d00", lw=2.2, label="VOD TAC")
    ax.fill_between(v_years, v_mean - v_se, v_mean + v_se, color="#b55d00", alpha=0.18, lw=0)

    ax.axvline(2008, color="0.5", ls="--", lw=1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("TAC")
    ax.set_title("Pan-Arctic TAC trajectories", loc="center")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.text(
        0.03, 0.05,
        f"r$_{{pre-2008}}$ = {r_pre:.2f}\n"
        f"r$_{{post-2008}}$ = {r_post:.2f}",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return {"r_pre": r_pre, "n_pre": n_pre, "r_post": r_post, "n_post": n_post}


def plot_consistency_map(ax, map_values, grid_lon, grid_lat, title):
    """Panels b/c: Arctic map of directional consistency."""
    cmap = mcolors.ListedColormap(["#2166ac", "#A9A9A9", "#b2182b"])
    norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    setup_polar_axis(ax)
    ax.pcolormesh(
        grid_lon, grid_lat, map_values,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    ax.set_title(title, loc="center")


def plot_fraction_bar(ax, pre_fraction, post_fraction):
    """Panel d: consistent-direction fraction."""
    labels = ["Pre-2008", "Post-2008"]
    values = [pre_fraction, post_fraction]
    bars = ax.bar(labels, values, color=["0.55", "0.25"], width=0.62)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Consistent fraction (%)")
    ax.set_title("Consistent fraction", loc="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8
        )


def main():
    plt.rc("font", family="Arial")
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Spatial consistency
    pf_mask, grid_lon, grid_lat = load_landcover_mask_and_grid()
    ktrend = np.load(KNDVI_TREND_PATH).astype(float)
    vtrend = np.load(VOD_TREND_PATH).astype(float)

    k_pre, v_pre = get_period_slopes(ktrend, vtrend, "pre")
    k_post, v_post = get_period_slopes(ktrend, vtrend, "post")

    pre_vals, pre_frac, pre_n_cons, pre_n_valid = consistent_direction_values(k_pre, v_pre)
    post_vals, post_frac, post_n_cons, post_n_valid = consistent_direction_values(k_post, v_post)

    pre_map = vector_to_map(pre_vals, pf_mask)
    post_map = vector_to_map(post_vals, pf_mask)

    # Optional: spatial correlations of trend slopes, useful for response text.
    r_pre_spatial, n_pre_spatial = pearson_r(k_pre, v_pre)
    r_post_spatial, n_post_spatial = pearson_r(k_post, v_post)

    # Figure: 1 row x 4 columns
    fig = plt.figure(figsize=(12.0, 3.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.0, 1.0, 1.])

    ax_a = fig.add_subplot(gs[0, 0])
    temporal_stats = plot_temporal_panel(ax_a)

    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
    plot_consistency_map(
        ax_b, pre_map, grid_lon, grid_lat,
        f"Pre-2008 direction"
    )

    ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.NorthPolarStereo())
    plot_consistency_map(
        ax_c, post_map, grid_lon, grid_lat,
        f"Post-2008 direction"
    )

    ax_d = fig.add_subplot(gs[0, 3])
    plot_fraction_bar(ax_d, pre_frac, post_frac)

    legend_handles = [
        Patch(facecolor="#2166ac", edgecolor="none", label="Both negative"),
        Patch(facecolor="#b2182b", edgecolor="none", label="Both positive"),
        Patch(facecolor="#A9A9A9", edgecolor="none", label="Opposite sign"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        # frameon=False,
        # bbox_to_anchor=(0.61, -0.04),
        fontsize=8,
    )

    out_png = FIG_DIR / f"{OUT_BASENAME}.png"
    out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
    fig.savefig(out_png, dpi=900, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
    # plt.close(fig)

    print("Saved:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print("\nTemporal correlation between pan-Arctic mean TAC trajectories:")
    print(f"  Pre-2008  : r = {temporal_stats['r_pre']:.3f}, n = {temporal_stats['n_pre']}")
    print(f"  Post-2008 : r = {temporal_stats['r_post']:.3f}, n = {temporal_stats['n_post']}")
    print("\nSpatial trend correlation between kNDVI and VOD TAC trends:")
    print(f"  Pre-2008  : r = {r_pre_spatial:.3f}, n = {n_pre_spatial}")
    print(f"  Post-2008 : r = {r_post_spatial:.3f}, n = {n_post_spatial}")
    print("\nSpatial consistent-direction fraction:")
    print(f"  Pre-2008  : {pre_frac:.2f}% ({pre_n_cons}/{pre_n_valid})")
    print(f"  Post-2008 : {post_frac:.2f}% ({post_n_cons}/{post_n_valid})")


if __name__ == "__main__":
    main()
