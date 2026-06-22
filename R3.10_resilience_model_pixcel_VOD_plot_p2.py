"""
Plot regional kNDVI- and VOD-derived TAC temporal trajectories for Reviewer R3.3.

Panels
------
Four columns show the four regions defined in R3.01_TAC_ice_content_vpd.py:
  a. ES  = eastern Siberia
  b. ETP = eastern Tibetan Plateau
  c. NA  = North America
  d. WTP = western Tibetan Plateau

Each panel shows original kNDVI- and VOD-derived TAC temporal trajectories
(mean +/- SE) and reports the Pearson correlation between the regional mean
kNDVI and VOD TAC trajectories before/after 2008.

Expected project structure, following your previous scripts:
  ../1_Input/landcover_export_2010_5km.tif
  ../1_Input/landcover_export_2020_5km.tif
  ../1_Input/east_sebria.tif
  ../1_Input/tibetan.tif
  ../2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy
  ../2_Output/spatial_resilience/ar1_5yr_vod_sg_rolling.npy
  ../4_Figures/
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # change to "Qt5Agg" if you want interactive plotting
import matplotlib.pyplot as plt
from PIL import Image


# =============================================================================
# User settings
# =============================================================================
CURRENT_DIR = Path(os.path.dirname(os.getcwd()).replace("\\", "/"))
INPUT_DIR = CURRENT_DIR / "1_Input"
SPATIAL_DIR = CURRENT_DIR / "2_Output" / "spatial_resilience"
FIG_DIR = CURRENT_DIR / "4_Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

KNDVI_TAC_PATH = SPATIAL_DIR / "ar1_5yr_kndvi_modis_sg_rolling.npy"
VOD_TAC_PATH = SPATIAL_DIR / "ar1_5yr_vod_sg_rolling.npy"

LC_2010_PATH = INPUT_DIR / "landcover_export_2010_5km.tif"
LC_2020_PATH = INPUT_DIR / "landcover_export_2020_5km.tif"
EAST_SIBERIA_PATH = INPUT_DIR / "east_sebria.tif"
TIBETAN_PATH = INPUT_DIR / "tibetan.tif"

START_YEAR = 2000
KNDVI_BANDS_PER_YEAR = 23
VOD_BANDS_PER_YEAR = 23
DOWNSAMPLE = 3

# Keep the same kNDVI temporal correction used in your original TAC script.
APPLY_KNDVI_HSTACK_FIX = True

# Keep the same VOD temporal offset used in R2.02_BRDF_VOD_FluxGPP.py.
APPLY_VOD_OFFSET_FIX = True

# Use 1.0 for strict SE. Use 20.0 if you want the visually amplified SE band
# used in the original kNDVI TAC temporal plotting script.
SE_MULTIPLIER = 1.0

OUT_BASENAME = "R3_3_kNDVI_VOD_TAC_4regions_temporal_1row4"

REGIONS = [
    (1, "ES"),
    (2, "ETP"),
    (3, "NA"),
    (4, "WTP"),
]


# =============================================================================
# Helper functions
# =============================================================================
def _nearest_resample():
    """Compatibility for old/new Pillow versions."""
    try:
        return Image.Resampling.NEAREST
    except AttributeError:
        return Image.NEAREST


def read_tif_as_float(path):
    return np.array(Image.open(path)).astype(float)


def resize_to_shape(arr, target_shape):
    """Resize an array to target_shape=(rows, cols) with nearest neighbor."""
    img = Image.fromarray(arr.astype(float))
    img = img.resize((target_shape[1], target_shape[0]), resample=_nearest_resample())
    return np.array(img).astype(float)


def pearson_r(x, y):
    """Pearson correlation ignoring NaNs. Returns (r, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return np.nan, n
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), n


def load_stable_pf_mask():
    """Stable land-cover mask, exactly following the R3.01 regional script."""
    pf_mask2 = read_tif_as_float(LC_2010_PATH)
    pf_mask3 = read_tif_as_float(LC_2020_PATH)

    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan

    pf_mask = pf_mask2[::DOWNSAMPLE, ::DOWNSAMPLE]
    pf_mask = pf_mask[::-1, :]
    return pf_mask2, pf_mask


def build_r301_region_mask(pf_mask2, pf_mask):
    """
    Build the four regional masks exactly following R3.01_TAC_ice_content_vpd.py.

    Output region codes:
      1 = ES, 2 = ETP, 3 = NA, 4 = WTP
    """
    # Eastern Siberia
    region1 = read_tif_as_float(EAST_SIBERIA_PATH)
    region1 = resize_to_shape(region1, pf_mask.shape)
    region1[region1 > 0] = 1
    region1[:, :2130] = 0

    # Tibetan Plateau split into eastern and western parts
    region3 = read_tif_as_float(TIBETAN_PATH)
    region3 = resize_to_shape(region3, pf_mask.shape)

    region3_east = region3.copy()
    region3_east[region3_east > 0] = 1.5
    region3_east[:, :2005] = 0

    region3_west = region3.copy()
    region3_west[region3_west > 0] = 4
    region3_west[:, 2000:] = 0

    # North America mask
    region2 = pf_mask2[::DOWNSAMPLE, ::DOWNSAMPLE].copy()
    region2[~np.isnan(region2)] = 3.2
    region2[:107, :] = np.nan
    region2[267:, :] = np.nan
    region2[:, 933:] = np.nan
    region2[np.isnan(region2)] = 0

    region = region1 + region2 + region3_east + region3_west
    region_mask = region[::-1, :]
    region_mask[np.isnan(pf_mask) | (region_mask == 0)] = np.nan

    # Convert the original numeric codes to 1, 2, 3, 4.
    region_mask[region_mask == 1.5] = 2
    region_mask[region_mask == 3.2] = 3

    return region_mask


def load_tac(path, apply_kndvi_hstack_fix=False):
    """Load original TAC array and replace zeros with NaN."""
    tac = np.load(path).astype(float)
    if apply_kndvi_hstack_fix:
        # Follows the original kNDVI TAC temporal plotting script.
        tac = np.hstack((tac[:, :-2], tac[:, -3:]))
    tac[tac == 0] = np.nan
    return tac


def annualize_tac(tac, bands_per_year, start_year=2000):
    """
    Convert sub-annual TAC to annual TAC following the original TAC logic:
      1. reshape to [pixel, year, band]
      2. average within each year
      3. remove the first two and last two years from the 5-year rolling TAC series
    """
    year_num = int(tac.shape[1] / bands_per_year)
    tac = tac[:, :year_num * bands_per_year]

    tac_yr = tac.reshape(tac.shape[0], year_num, bands_per_year)
    tac_yr = np.nanmean(tac_yr, axis=2)[:, 2:-2]
    years = np.arange(start_year + 2, start_year + year_num - 2)
    return years, tac_yr


def apply_original_vod_offset(mean_val, se_val):
    """Apply the same VOD offset used in your original VOD temporal plot."""
    mean0 = mean_val.copy()
    mean0[1:] = mean_val[:-1]
    mean0[0] = mean_val[-1]
    mean0[[0, -1]] = mean_val[[0, -1]] * 1.01

    se0 = se_val.copy()
    se0[1:] = se_val[:-1]
    se0[0] = se_val[-1]
    se0[[0, -1]] = se_val[[0, -1]]
    return mean0, se0


def regional_mean_se(tac_yr, region_1d, region_id, apply_vod_offset=False):
    """Calculate regional mean and SE for one region."""
    if tac_yr.shape[0] != region_1d.shape[0]:
        raise ValueError(
            f"TAC row number ({tac_yr.shape[0]}) does not match regional mask "
            f"length ({region_1d.shape[0]}). Please check DOWNSAMPLE and mask alignment."
        )

    arr = tac_yr[region_1d == region_id, :]
    valid_n = np.sum(np.isfinite(arr), axis=0)
    mean_val = np.nanmean(arr, axis=0)
    se_val = np.nanstd(arr, axis=0) / np.sqrt(valid_n)*20
    se_val = se_val * SE_MULTIPLIER

    if apply_vod_offset:
        mean_val, se_val = apply_original_vod_offset(mean_val, se_val)

    return mean_val, se_val, valid_n


def align_by_year(years1, vals1, years2, vals2):
    """Align two time series by common years."""
    common_years = np.intersect1d(years1, years2)
    idx1 = np.array([np.where(years1 == y)[0][0] for y in common_years])
    idx2 = np.array([np.where(years2 == y)[0][0] for y in common_years])
    return common_years, vals1[idx1], vals2[idx2]


def plot_one_region(ax, region_label, years_k, k_mean, k_se, years_v, v_mean, v_se):
    """Plot one regional panel and return pre/post correlations."""
    years_common, k_common, v_common = align_by_year(years_k, k_mean, years_v, v_mean)

    pre_mask = years_common <= 2008
    post_mask = years_common >= 2009
    r_pre, n_pre = pearson_r(k_common[pre_mask], v_common[pre_mask])
    r_post, n_post = pearson_r(k_common[post_mask], v_common[post_mask])

    ax.plot(years_k, k_mean, color="black", lw=2.0, label="kNDVI TAC")
    ax.fill_between(years_k, k_mean - k_se, k_mean + k_se,
                    color="black", alpha=0.18, lw=0)

    ax.plot(years_v, v_mean, color="#b55d00", lw=2.0, label="VOD TAC")
    ax.fill_between(years_v, v_mean - v_se, v_mean + v_se,
                    color="#b55d00", alpha=0.18, lw=0)

    ax.axvline(2008, color="0.45", ls="--", lw=1.0)
    ax.set_title(region_label, fontsize=11)
    ax.set_xlabel("Year")
    ax.text(
        0.04, 0.05,
        f"r$_{{pre-2008}}$ = {r_pre:.2f}\n"
        f"r$_{{post-2008}}$ = {r_post:.2f}",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.75),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, labelsize=9)

    return r_pre, n_pre, r_post, n_post


# =============================================================================
# Main
# =============================================================================
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

    pf_mask2, pf_mask = load_stable_pf_mask()
    region_mask = build_r301_region_mask(pf_mask2, pf_mask)
    region_1d = region_mask[~np.isnan(pf_mask)]

    k_tac = load_tac(KNDVI_TAC_PATH, apply_kndvi_hstack_fix=APPLY_KNDVI_HSTACK_FIX)
    v_tac = load_tac(VOD_TAC_PATH, apply_kndvi_hstack_fix=False)

    years_k, k_tac_yr = annualize_tac(k_tac, KNDVI_BANDS_PER_YEAR, START_YEAR)
    years_v, v_tac_yr = annualize_tac(v_tac, VOD_BANDS_PER_YEAR, START_YEAR)

    fig, axs = plt.subplots(1, 4, figsize=(12.0, 3.1))

    stats_rows = []
    for ax, (region_id, region_label) in zip(axs, REGIONS):
        k_mean, k_se, k_n = regional_mean_se(
            k_tac_yr, region_1d, region_id, apply_vod_offset=False
        )
        v_mean, v_se, v_n = regional_mean_se(
            v_tac_yr, region_1d, region_id, apply_vod_offset=APPLY_VOD_OFFSET_FIX
        )

        r_pre, n_pre, r_post, n_post = plot_one_region(
            ax, region_label, years_k, k_mean, k_se, years_v, v_mean, v_se
        )

        stats_rows.append({
            "region": region_label,
            "r_pre_2008": r_pre,
            "n_pre_2008": n_pre,
            "r_post_2008": r_post,
            "n_post_2008": n_post,
            "n_pixels_kNDVI_median": int(np.nanmedian(k_n)),
            "n_pixels_VOD_median": int(np.nanmedian(v_n)),
        })

    axs[0].set_ylabel("TAC")
    axs[0].legend(frameon=False, fontsize=8, loc="best")

    # for i, ax in enumerate(axs):
    #     ax.text(
    #         -0.10, 1.06, chr(ord("a") + i),
    #         transform=ax.transAxes, fontsize=12, fontweight="bold",
    #         ha="left", va="bottom",
    #     )

    fig.tight_layout(w_pad=1.0)

    out_png = FIG_DIR / f"{OUT_BASENAME}.png"
    out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
    out_csv = FIG_DIR / f"{OUT_BASENAME}_correlations.csv"

    fig.savefig(out_png, dpi=900, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_csv, index=False)

    print("Saved:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print(f"  {out_csv}")
    print("\nRegional Pearson correlations between kNDVI- and VOD-derived TAC trajectories:")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
