from pathlib import Path
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pwlf
from scipy import stats
from scipy.linalg import LinAlgWarning

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
import matplotlib

matplotlib.use("qtAgg")
import matplotlib.pyplot as plt


matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,
})


INPUT_CSV = Path("/Volumes/Zhengfei_01/Project 2 pf resilience/2_Output/SST_OCEAN_MFC_VAP.csv")
FIGURE_DIR = Path("/Volumes/Zhengfei_01/Project 2 pf resilience/4_Figures")
OUTPUT_PNG = FIGURE_DIR / "R312_OET_MFC_VVAP_timeseries.png"
OUTPUT_PDF = FIGURE_DIR / "R312_OET_MFC_VVAP_timeseries.pdf"
SCATTER_PNG = FIGURE_DIR / "R312_OET_MFC_VVAP_scatter_relationships.png"
SCATTER_PDF = FIGURE_DIR / "R312_OET_MFC_VVAP_scatter_relationships.pdf"
OUTPUT_CSV = FIGURE_DIR / "R312_OET_MFC_VVAP_timeseries_data.csv"

YEAR_SPLIT = 2008
TEAL = "#5b9699"
ORANGE = "#e7ad4c"


def add_piecewise_fit(ax, years, values, color, anchor=None):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(years) & np.isfinite(values)
    if valid.sum() < 4:
        return

    year_valid = years[valid]
    x = year_valid - YEAR_SPLIT
    y = values[valid]
    if anchor is None:
        anchor = float(np.interp(YEAR_SPLIT, year_valid, y))

    model = pwlf.PiecewiseLinFit(x, y, degree=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        model.fit(2, [0], [anchor])
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = model.predict(x_hat)
    ax.plot(x_hat + YEAR_SPLIT, y_hat, "--", color=color, lw=1.5)


def plot_series(ax, years, values, spread, color, label, anchor=None):
    ax.plot(years, values, color=color, lw=1.0, label=label)
    ax.fill_between(years, values - spread, values + spread, color=color, alpha=0.15, linewidth=0)
    add_piecewise_fit(ax, years, values, color, anchor=anchor)


def style_axis(ax):
    ax.axvline(YEAR_SPLIT, ls="--", color="0.55", lw=1.2)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)
    ax.set_xlabel("Year")
    ax.tick_params(labelsize=12, width=1.05)


def main():
    df = pd.read_csv(INPUT_CSV)
    years = df["Year"].to_numpy()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    fig, axs = plt.subplots(3, 1, figsize=(4.7, 6.5), sharex=True)

    plot_series(
        axs[0],
        years,
        df["VPD1_Anomaly"].to_numpy(),
        df["VPD1_StdDev"].to_numpy(),
        TEAL,
        "TerraClimate",
        anchor=-0.08,
    )
    plot_series(
        axs[0],
        years,
        df["VPD2_Anomaly"].to_numpy(),
        df["VPD2_StdDev"].to_numpy(),
        ORANGE,
        "CRU",
        anchor=-0.08,
    )
    axs[0].set_ylabel("VPD anomaly (hPa)")
    axs[0].set_ylim(-0.4, 0.4)
    axs[0].legend(frameon=False, fontsize=11, loc="upper left")
    axs[0].text(YEAR_SPLIT + 0.2, 0.34, "2008", fontsize=11)
    style_axis(axs[0])

    plot_series(
        axs[1],
        years,
        df["SVAP_Anomaly"].to_numpy(),
        df["SVAP_StdDev"].to_numpy(),
        TEAL,
        "SVAP",
        anchor=0.01,
    )
    plot_series(
        axs[1],
        years,
        df["VAP_Anomaly"].to_numpy(),
        df["VAP_StdDev"].to_numpy(),
        ORANGE,
        "AVAP",
        anchor=0.01,
    )
    axs[1].set_ylabel("VAP anomaly (hPa)")
    axs[1].set_ylim(-0.6, 0.6)
    axs[1].legend(frameon=False, fontsize=11, loc="upper left")
    style_axis(axs[1])

    ax_mfc = axs[2]
    ax_evap = ax_mfc.twinx()
    plot_series(
        ax_mfc,
        years,
        df["MFC_mm_day_anom"].to_numpy(),
        df["MFC_mm_day_anom_se"].to_numpy(),
        TEAL,
        "MFC",
        anchor=0.06,
    )
    plot_series(
        ax_evap,
        years,
        df["OceanE_Anomaly"].to_numpy(),
        df["OceanE_StdDev"].to_numpy(),
        ORANGE,
        "Ocean evaporation",
        anchor=1.0,
    )
    ax_mfc.set_ylabel("MFC anomaly (mm day$^{-1}$)")
    ax_evap.set_ylabel("Evaporation anomaly (mm)")
    ax_mfc.set_ylim(-0.18, 0.18)
    ax_evap.set_ylim(-6, 6)
    ax_mfc.legend(
        [ax_mfc.lines[0], ax_evap.lines[0]],
        ["MFC", "Evaporation"],
        frameon=True,
        facecolor="white",
        framealpha=0.6,
        edgecolor="none",
        fontsize=11,
        loc="upper left",
    )
    style_axis(ax_mfc)
    ax_evap.spines["top"].set_visible(False)
    ax_evap.spines["right"].set_linewidth(1.05)
    ax_evap.tick_params(labelsize=12, width=1.05)

    for ax in axs:
        ax.set_xlim(years.min() - 3, years.max() + 1)
        ax.set_xticks([1990, 2000, 2010, 2020])

    fig.tight_layout(h_pad=1.8)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
