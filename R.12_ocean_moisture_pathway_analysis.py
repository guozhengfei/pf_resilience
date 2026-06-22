"""
R3.4 ocean–atmosphere moisture-supply pathway analysis

Input CSV required columns:
    Year
    SVAP_Anomaly, VAP_Anomaly
    SST_Anomaly, OceanE_Anomaly
    MFC_mm_day_anom
Optional uncertainty columns used for error bars if present:
    SST_StdDev, OceanE_StdDev, VAP_StdDev, SVAP_StdDev, MFC_mm_day_anom_se

Purpose:
    1) Calculate VPD anomaly as SVAP anomaly - VAP/AVAP anomaly.
    2) Estimate pre- and post-2008 linear trends.
    3) Test pathway consistency using Pearson and partial correlations.
    4) Create publication-style figures for the R3.4 response.

Interpretation note:
    These correlations support pathway consistency, not definitive causal attribution.
    SST and ocean evaporation are ocean-domain variables; MFC, AVAP/VAP, VPD are
    land-domain variables. The script therefore uses fixed-domain annual mean anomalies.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("qtAgg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------
INPUT_CSV = Path('/Volumes/Zhengfei_01/Project 2 pf resilience/2_Output/SST_OCEAN_MFC_VAP.csv')
OUTPUT_DIR = Path('/Volumes/Zhengfei_01/Project 2 pf resilience/4_Figures')
BREAK_YEAR = 2008
FIG_DPI = 600

# Years <= BREAK_YEAR are treated as pre-2008; years > BREAK_YEAR as post-2008.
PRE_LABEL = f"pre-{BREAK_YEAR}"
POST_LABEL = f"post-{BREAK_YEAR}"

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def zscore(x: pd.Series) -> pd.Series:
    """Standardize a pandas Series using its sample standard deviation."""
    return (x - x.mean()) / x.std(ddof=1)


def linear_trend(year, y):
    """Ordinary least-squares linear trend y ~ year."""
    x = np.asarray(year, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) < 3:
        return dict(n=len(y), slope=np.nan, intercept=np.nan, r=np.nan, p=np.nan, se=np.nan)
    res = stats.linregress(x, y)
    return dict(n=len(y), slope=res.slope, intercept=res.intercept, r=res.rvalue, p=res.pvalue, se=res.stderr)


def residualize(y, covariates=None):
    """Return residuals after regressing y on covariates plus an intercept."""
    y = np.asarray(y, dtype=float)
    if covariates is None or len(covariates) == 0:
        return y - np.nanmean(y)

    Z = np.asarray(covariates, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    X = np.column_stack([np.ones(len(y)), Z])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def lag1_autocorr(x):
    """Lag-1 autocorrelation, robust to constant or short vectors."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.nanstd(x) == 0:
        return np.nan
    try:
        return stats.pearsonr(x[:-1], x[1:]).statistic
    except Exception:
        return np.nan


def effective_n_ar1(x_resid, y_resid, n, k=0):
    """
    Approximate effective sample size under AR(1) autocorrelation.
    Based on the common Pyper-Peterman style correction:
        neff = n * (1 - rho_x*rho_y) / (1 + rho_x*rho_y)
    """
    rho_x = lag1_autocorr(x_resid)
    rho_y = lag1_autocorr(y_resid)
    if not np.isfinite(rho_x) or not np.isfinite(rho_y) or abs(1 + rho_x * rho_y) < 1e-12:
        neff = n
    else:
        neff = n * (1 - rho_x * rho_y) / (1 + rho_x * rho_y)
    # Keep neff in a meaningful range for partial-correlation p-values.
    neff = float(np.clip(neff, k + 3.1, n))
    return neff, rho_x, rho_y


def p_value_from_r(r, n_eff, k=0):
    """Two-sided p-value for Pearson/partial correlation using effective df."""
    if not np.isfinite(r):
        return np.nan
    dfree = n_eff - k - 2
    if dfree <= 0:
        return np.nan
    if abs(r) >= 1:
        return 0.0
    tval = r * np.sqrt(dfree / max(1e-12, 1 - r * r))
    return 2 * stats.t.sf(abs(tval), dfree)


def fisher_ci(r, n_eff, k=0, alpha=0.05):
    """Approximate Fisher-z confidence interval for a partial correlation."""
    if not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    df_for_se = n_eff - k - 3
    if df_for_se <= 0:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(df_for_se)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh([z - zcrit * se, z + zcrit * se])
    return lo, hi


def partial_corr(df, x_col, y_col, control_cols=None):
    """Pearson or partial correlation, with raw and AR(1)-adjusted p-values."""
    control_cols = control_cols or []
    cols = [x_col, y_col] + control_cols
    sub = df[cols].dropna().copy()
    n = len(sub)
    k = len(control_cols)
    if n < k + 4:
        return dict(r=np.nan, p_raw=np.nan, p_ar1=np.nan, ci_low=np.nan, ci_high=np.nan,
                    n=n, n_eff=np.nan, k=k, rho_x=np.nan, rho_y=np.nan)

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    Z = sub[control_cols].to_numpy(dtype=float) if control_cols else None

    x_res = residualize(x, Z)
    y_res = residualize(y, Z)
    r, p_raw = stats.pearsonr(x_res, y_res)
    n_eff, rho_x, rho_y = effective_n_ar1(x_res, y_res, n=n, k=k)
    p_ar1 = p_value_from_r(r, n_eff=n_eff, k=k)
    ci_low, ci_high = fisher_ci(r, n_eff=n_eff, k=k)

    return dict(r=r, p_raw=p_raw, p_ar1=p_ar1, ci_low=ci_low, ci_high=ci_high,
                n=n, n_eff=n_eff, k=k, rho_x=rho_x, rho_y=rho_y)


def add_regression_line(ax, x, y):
    """Add simple regression line to a scatter plot."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    slope, intercept, *_ = stats.linregress(x[mask], y[mask])
    xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    ax.plot(xx, intercept + slope * xx, lw=1.5)


def add_panel_label(ax, label):
    ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, fontweight="bold")


def format_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT_CSV)

required = ["Year", "SVAP_Anomaly", "VAP_Anomaly", "SST_Anomaly", "OceanE_Anomaly", "MFC_mm_day_anom"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.sort_values("Year").reset_index(drop=True)

# VAP is actual vapour pressure; VPD anomaly is approximated as SVAP anomaly - VAP anomaly.
df["VPD_Anomaly"] = df["SVAP_Anomaly"] - df["VAP_Anomaly"]

# Add one-year lagged ocean variables for optional sensitivity tests.
df["SST_Anomaly_lag1"] = df["SST_Anomaly"].shift(1)
df["OceanE_Anomaly_lag1"] = df["OceanE_Anomaly"].shift(1)

# Standardized variables for plotting only.
for col in ["SST_Anomaly", "OceanE_Anomaly", "MFC_mm_day_anom", "VAP_Anomaly", "SVAP_Anomaly", "VPD_Anomaly"]:
    df[f"z_{col}"] = df[col]#zscore(df[col])

# ---------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------
trend_rows = []
trend_vars = {
    "SST": "SST_Anomaly",
    "Ocean evaporation": "OceanE_Anomaly",
    "Land MFC": "MFC_mm_day_anom",
    "Land AVAP/VAP": "VAP_Anomaly",
    "Land SVAP": "SVAP_Anomaly",
    "Land VPD": "VPD_Anomaly",
}
for name, col in trend_vars.items():
    for period, sub in [(PRE_LABEL, df[df["Year"] <= BREAK_YEAR]), (POST_LABEL, df[df["Year"] > BREAK_YEAR])]:
        out = linear_trend(sub["Year"], sub[col])
        trend_rows.append({
            "variable": name,
            "column": col,
            "period": period,
            "n": out["n"],
            "slope_per_year": out["slope"],
            "r": out["r"],
            "p": out["p"],
            "slope_se": out["se"],
        })
trends = pd.DataFrame(trend_rows)
trends.to_csv(OUTPUT_DIR / "R34_pre_post_trends.csv", index=False)

# ---------------------------------------------------------------------
# Pathway correlation analysis
# ---------------------------------------------------------------------
# Main tests are designed to link the upstream pathway:
#     SST/ocean evaporation -> land MFC -> land AVAP/VPD.
# Existing manuscript analyses already link VPD to TAC.
main_specs = [
    ("SST -> ocean evaporation", "SST_Anomaly", "OceanE_Anomaly", []),
    ("Ocean evaporation -> land MFC", "OceanE_Anomaly", "MFC_mm_day_anom", []),
    ("Ocean evaporation(t-1) -> land MFC", "OceanE_Anomaly_lag1", "MFC_mm_day_anom", []),
    ("Land MFC -> land AVAP", "MFC_mm_day_anom", "VAP_Anomaly", []),
    ("Land MFC -> land VPD | SVAP", "MFC_mm_day_anom", "VPD_Anomaly", ["SVAP_Anomaly"]),
    ("Ocean evaporation -> land AVAP", "OceanE_Anomaly", "VAP_Anomaly", []),
]

corr_rows = []
for detrend in [False, True]:
    for label, x_col, y_col, controls in main_specs:
        controls_use = list(controls)
        analysis_type = "raw annual anomalies"
        if detrend:
            controls_use = controls_use + ["Year"]
            analysis_type = "year-controlled anomalies"
        out = partial_corr(df, x_col, y_col, controls_use)
        corr_rows.append({
            "analysis_type": analysis_type,
            "pathway_link": label,
            "x": x_col,
            "y": y_col,
            "controls": ";".join(controls_use) if controls_use else "none",
            **out,
        })
correlations = pd.DataFrame(corr_rows)
correlations.to_csv(OUTPUT_DIR / "R34_pathway_correlations.csv", index=False)

# ---------------------------------------------------------------------
# Figure 1: pathway time series + correlation summary
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
ax = axes[0, 0]
ax.plot(df["Year"], df["z_SST_Anomaly"], marker="o", ms=3, lw=1.5, label="SST")
ax.plot(df["Year"], df["z_OceanE_Anomaly"], marker="o", ms=3, lw=1.5, label="Ocean evaporation")
ax.axvline(BREAK_YEAR, ls="--", lw=1)
ax.set_ylabel("Standardized anomaly")
ax.set_title("Ocean-domain anomalies")
ax.legend(frameon=False, fontsize=8)
add_panel_label(ax, "a")

ax = axes[0, 1]
ax.plot(df["Year"], df["z_MFC_mm_day_anom"], marker="o", ms=3, lw=1.5, label="Land MFC")
ax.plot(df["Year"], df["z_VAP_Anomaly"], marker="o", ms=3, lw=1.5, label="Land AVAP")
ax.plot(df["Year"], df["z_VPD_Anomaly"], marker="o", ms=3, lw=1.5, label="Land VPD")
ax.axvline(BREAK_YEAR, ls="--", lw=1)
ax.set_ylabel("Standardized anomaly")
ax.set_title("Land-domain moisture anomalies")
ax.legend(frameon=False, fontsize=8)
add_panel_label(ax, "b")

ax = axes[1, 0]
# MFC time series with optional SE bars and fitted pre/post trends.
y = df["MFC_mm_day_anom"].to_numpy(dtype=float)
if "MFC_mm_day_anom_se" in df.columns:
    ax.errorbar(df["Year"], y, yerr=df["MFC_mm_day_anom_se"], marker="o", ms=3, lw=1.2, capsize=2, label="Land MFC")
else:
    ax.plot(df["Year"], y, marker="o", ms=3, lw=1.2, label="Land MFC")
ax.axhline(0, lw=0.8)
ax.axvline(BREAK_YEAR, ls="--", lw=1)
for sub in [df[df["Year"] <= BREAK_YEAR], df[df["Year"] > BREAK_YEAR]]:
    out = linear_trend(sub["Year"], sub["MFC_mm_day_anom"])
    xx = np.array([sub["Year"].min(), sub["Year"].max()])
    ax.plot(xx, out["intercept"] + out["slope"] * xx, lw=2)
ax.set_xlabel("Year")
ax.set_ylabel("MFC anomaly (mm day$^{-1}$)")
ax.set_title("Land moisture-flux convergence")
add_panel_label(ax, "c")

ax = axes[1, 1]
main_corr = correlations[correlations["analysis_type"] == "raw annual anomalies"].copy()
plot_order = [
    "SST -> ocean evaporation",
    "Ocean evaporation -> land MFC",
    "Ocean evaporation(t-1) -> land MFC",
    "Land MFC -> land AVAP",
    "Land MFC -> land VPD | SVAP",
    "Ocean evaporation -> land AVAP",
]
main_corr["pathway_link"] = pd.Categorical(main_corr["pathway_link"], categories=plot_order, ordered=True)
main_corr = main_corr.sort_values("pathway_link")
ypos = np.arange(len(main_corr))
ax.axvline(0, lw=0.8)
# horizontal confidence intervals if available
for i, row in enumerate(main_corr.itertuples(index=False)):
    if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
        ax.plot([row.ci_low, row.ci_high], [i, i], lw=1.4)
ax.scatter(main_corr["r"], ypos, s=30)
ax.set_yticks(ypos)
ax.set_yticklabels(main_corr["pathway_link"], fontsize=7)
ax.set_xlabel("Correlation coefficient")
ax.set_title("Pathway consistency tests")
ax.set_xlim(-1, 1)
add_panel_label(ax, "d")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Fig_R34_pathway_timeseries_correlations.png", dpi=FIG_DPI, bbox_inches="tight")
# fig.savefig(OUTPUT_DIR / "Fig_R34_pathway_timeseries_correlations.pdf", bbox_inches="tight")
# plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: scatter/residual plots for pathway links
# ---------------------------------------------------------------------
scatter_specs = [
    ("SST -> ocean evaporation", "SST_Anomaly", "OceanE_Anomaly", [], "SST anomaly", "Ocean evaporation anomaly"),
    ("Ocean evaporation -> land MFC", "OceanE_Anomaly", "MFC_mm_day_anom", [], "Ocean evaporation anomaly", "Land MFC anomaly"),
    ("Ocean evaporation(t-1) -> land MFC", "OceanE_Anomaly_lag1", "MFC_mm_day_anom", [], "Ocean evaporation anomaly at t-1", "Land MFC anomaly at t"),
    ("Land MFC -> land AVAP", "MFC_mm_day_anom", "VAP_Anomaly", [], "Land MFC anomaly", "Land AVAP anomaly"),
    ("Land MFC -> land VPD | SVAP", "MFC_mm_day_anom", "VPD_Anomaly", ["SVAP_Anomaly"], "Land MFC residual", "Land VPD residual"),
    ("Ocean evaporation -> land AVAP", "OceanE_Anomaly", "VAP_Anomaly", [], "Ocean evaporation anomaly", "Land AVAP anomaly"),
]

fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
for ax, spec, lab in zip(axes.ravel(), scatter_specs, list("abcdef")):
    title, x_col, y_col, controls, xlabel, ylabel = spec
    cols = [x_col, y_col] + controls
    sub = df[cols].dropna().copy()
    if controls:
        Z = sub[controls].to_numpy(dtype=float)
        x_plot = residualize(sub[x_col].to_numpy(dtype=float), Z)
        y_plot = residualize(sub[y_col].to_numpy(dtype=float), Z)
    else:
        x_plot = sub[x_col].to_numpy(dtype=float)
        y_plot = sub[y_col].to_numpy(dtype=float)

    ax.scatter(x_plot, y_plot, s=28)
    add_regression_line(ax, x_plot, y_plot)
    out = partial_corr(df, x_col, y_col, controls)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.text(0.04, 0.94, f"r = {out['r']:.2f}\nP$_{{AR1}}$ = {format_p(out['p_ar1'])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8)
    add_panel_label(ax, lab)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Fig_R34_pathway_scatterplots.png", dpi=FIG_DPI, bbox_inches="tight")
# fig.savefig(OUTPUT_DIR / "Fig_R34_pathway_scatterplots.pdf", bbox_inches="tight")
# plt.close(fig)

# ---------------------------------------------------------------------
# Write a compact text summary for response-letter numbers
# ---------------------------------------------------------------------
summary_path = OUTPUT_DIR / "R34_response_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("R3.4 ocean–atmosphere moisture pathway analysis summary\n")
    f.write("=========================================================\n\n")
    f.write(f"Input file: {INPUT_CSV}\n")
    f.write(f"Years: {int(df['Year'].min())}-{int(df['Year'].max())}; n = {len(df)}\n")
    f.write(f"Breakpoint: {BREAK_YEAR}; pre = <= {BREAK_YEAR}, post = > {BREAK_YEAR}\n\n")

    f.write("Pre/post linear trends\n")
    f.write("----------------------\n")
    for row in trend_rows:
        f.write(
            f"{row['variable']} ({row['period']}): slope = {row['slope_per_year']:.4g} yr^-1, "
            f"r = {row['r']:.2f}, P = {format_p(row['p'])}, n = {row['n']}\n"
        )

    f.write("\nPathway correlations: raw annual anomalies\n")
    f.write("------------------------------------------\n")
    for row in correlations[correlations["analysis_type"] == "raw annual anomalies"].itertuples(index=False):
        f.write(
            f"{row.pathway_link}: r = {row.r:.2f}, P_raw = {format_p(row.p_raw)}, "
            f"P_AR1 = {format_p(row.p_ar1)}, n_eff = {row.n_eff:.1f}, controls = {row.controls}\n"
        )

    f.write("\nPathway correlations: year-controlled anomalies\n")
    f.write("-----------------------------------------------\n")
    for row in correlations[correlations["analysis_type"] == "year-controlled anomalies"].itertuples(index=False):
        f.write(
            f"{row.pathway_link}: partial r = {row.r:.2f}, P_raw = {format_p(row.p_raw)}, "
            f"P_AR1 = {format_p(row.p_ar1)}, n_eff = {row.n_eff:.1f}, controls = {row.controls}\n"
        )

print(f"Done. Outputs written to: {OUTPUT_DIR}")
print(f"  - {OUTPUT_DIR / 'R34_pre_post_trends.csv'}")
print(f"  - {OUTPUT_DIR / 'R34_pathway_correlations.csv'}")
print(f"  - {OUTPUT_DIR / 'Fig_R34_pathway_timeseries_correlations.png'}")
print(f"  - {OUTPUT_DIR / 'Fig_R34_pathway_scatterplots.png'}")
print(f"  - {summary_path}")

# Warn about deterministic relation when VPD is computed from SVAP and VAP.
warnings.warn(
    "VPD_Anomaly is calculated as SVAP_Anomaly - VAP_Anomaly. Do not use the "
    "partial correlation AVAP vs VPD controlling SVAP as evidence, because it is mathematically deterministic.",
    RuntimeWarning,
)
