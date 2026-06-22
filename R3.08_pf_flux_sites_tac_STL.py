from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL


INPUT_CSV = Path('../1_Input/pf_flux_60_sites_plus_additional_cleaned.csv')
OUTPUT_DIR = Path('../4_Figures/flux_site_tac_STL')

BANDS_YEAR = 12
TAC_YEARS = 3
TAC_WINDOW = BANDS_YEAR * TAC_YEARS
MIN_PERIODS = 12 * 2

GPP_COL = 'GPP_CUT'


def calc_ar1(x, min_pairs=24):
    """
    Calculate lag-1 autocorrelation with explicit validity checks.
    """
    x = pd.Series(x, dtype='float64').to_numpy()

    x0 = x[:-1]
    x1 = x[1:]

    valid = np.isfinite(x0) & np.isfinite(x1)

    if valid.sum() < min_pairs:
        return np.nan

    x0 = x0[valid]
    x1 = x1[valid]

    if np.nanstd(x0) == 0 or np.nanstd(x1) == 0:
        return np.nan

    return np.corrcoef(x0, x1)[0, 1]


def calc_gpp_residual_stl(
    df_site,
    gpp_col='GPP',
    period=12,
    seasonal=13,
    robust=True,
    min_valid=36,
):
    """
    Calculate GPP residuals for one site.

    Steps:
    1. Sort by year and month.
    2. Subtract annual mean GPP to remove interannual variation.
    3. Apply STL to the annual-mean-detrended monthly series.
    4. Return STL residuals.
    """

    df_site = df_site.sort_values(['year', 'month']).copy()
    gpp = pd.to_numeric(df_site[gpp_col], errors='coerce')

    residual = pd.Series(np.nan, index=df_site.index, name='gpp_resid')

    if gpp.notna().sum() < min_valid:
        return residual

    # Step 1: remove annual mean
    annual_mean = gpp.groupby(df_site['year']).transform('mean')
    detrended = gpp - annual_mean

    # STL cannot handle NaN
    detrended_interp = (
        detrended
        .interpolate(method='linear', limit_direction='both')
    )

    if detrended_interp.isna().any():
        return residual

    # Step 2: STL decomposition
    stl = STL(
        detrended_interp,
        period=period,
        seasonal=seasonal,
        robust=robust
    )

    result = stl.fit()

    residual.loc[df_site.index] = result.resid

    return residual


def save_site_figure(df_site):
    site = df_site['Site'].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(df_site['date'], df_site['GPP'], color='0.15', lw=1)
    axes[0].set_ylabel('GPP')
    axes[0].set_title(site)

    axes[1].axhline(0, color='0.7', lw=0.8)
    axes[1].plot(df_site['date'], df_site['gpp_resid'], color='C0', lw=1)
    axes[1].set_ylabel('GPP residual')

    axes[2].plot(df_site['date'], df_site['tac'], color='C3', lw=1)
    axes[2].set_ylabel('TAC')
    axes[2].set_xlabel('Year')

    for ax in axes:
        ax.grid(True, color='0.9', lw=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{site}_gpp_residual_tac.png', dpi=300)
    plt.close(fig)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: load data
df = pd.read_csv(INPUT_CSV)

required_cols = {'Site', 'year', 'month', GPP_COL}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df['GPP'] = pd.to_numeric(df[GPP_COL], errors='coerce')

df = df.dropna(subset=['Site', 'year', 'month']).copy()
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)

df['date'] = pd.to_datetime(
    df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
)

# Step 2: calculate residuals and TAC for each site
all_sites = []

for site, df_site in df.groupby('Site'):
    df_site = df_site.sort_values(['year', 'month']).copy()

    if df_site['GPP'].notna().sum() < MIN_PERIODS:
        print(f'Skipped {site}: too few valid GPP observations')
        continue

    df_site['gpp_resid'] = calc_gpp_residual_stl(
        df_site,
        gpp_col='GPP',
        period=12,
        seasonal=13,
        robust=True,
        min_valid=MIN_PERIODS,
    )

    if df_site['gpp_resid'].notna().sum() < MIN_PERIODS:
        print(f'Skipped {site}: too few valid residuals')
        continue

    df_site['tac'] = (
        df_site['gpp_resid']
        .rolling(TAC_WINDOW, min_periods=MIN_PERIODS, center=True)
        .apply(calc_ar1, raw=False)
    )

    save_site_figure(df_site)
    all_sites.append(df_site)

    print(f'Processed: {site}')

# Step 3: save monthly residuals and TAC
if not all_sites:
    raise RuntimeError('No sites were processed. Please check input data and filtering criteria.')

df_out = pd.concat(all_sites, ignore_index=True)
df_out.to_csv(OUTPUT_DIR / 'pf_flux_sites_monthly_residual_tac.csv', index=False)

print(f'Wrote figures and monthly TAC csv to: {OUTPUT_DIR}')
print(f'Processed sites: {df_out["Site"].nunique()}')
print('Remaining missing values:')
print(df_out[['GPP', 'gpp_resid', 'tac']].isna().sum())