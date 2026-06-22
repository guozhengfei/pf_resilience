from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


INPUT_CSV = Path('../1_Input/pf_flux_60_sites_plus_additional_final.csv')
OUTPUT_DIR = Path('../4_Figures/flux_site_sd')
BANDS_YEAR = 12
SD_YEARS = 3
SD_WINDOW = BANDS_YEAR * SD_YEARS
MIN_PERIODS = 12 * 2
SEASON_WINDOW_YEARS = 5


def calc_sd(x):
    """Calculate standard deviation."""
    return pd.Series(x).dropna().std()


def calc_gpp_residual(
    df_site,
    gpp_col='GPP',
    year_col='year',
    month_col='month',
    season_window_years=SEASON_WINDOW_YEARS,
    min_valid_years=3,
):

    df_site = df_site.sort_values([year_col, month_col]).copy()
    gpp = pd.to_numeric(df_site[gpp_col], errors='coerce')

    residual = pd.Series(np.nan, index=df_site.index, name=f'{gpp_col}_resid')

    # Need enough years
    years = np.sort(df_site[year_col].dropna().unique())
    if len(years) < min_valid_years:
        return residual

    # Step 1: remove annual mean
    annual_mean = gpp.groupby(df_site[year_col]).transform('mean')
    detrended = gpp - annual_mean

    # Step 2: moving monthly climatology
    seasonality = pd.Series(np.nan, index=df_site.index)

    half_window = season_window_years // 2

    for i, year in enumerate(years):
        # centered moving window
        start = max(0, i - half_window)
        end = min(len(years), i + half_window + 1)

        # expand window at edges to keep fixed window length if possible
        if end - start < season_window_years:
            if start == 0:
                end = min(len(years), season_window_years)
            elif end == len(years):
                start = max(0, len(years) - season_window_years)

        window_years = years[start:end]

        window_mask = df_site[year_col].isin(window_years)

        # monthly climatology within moving window
        clim = (
            detrended[window_mask]
            .groupby(df_site.loc[window_mask, month_col])
            .mean()
        )

        year_mask = df_site[year_col] == year
        year_idx = df_site.index[year_mask]

        seasonality.loc[year_idx] = (
            df_site.loc[year_idx, month_col]
            .map(clim)
            .values
        )

    # Step 3: residual = annual-mean-removed GPP - moving monthly climatology
    residual = detrended - seasonality
    residual.name = f'{gpp_col}_resid'

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

    # axes[2].axhline(0, color='0.7', lw=0.8)
    axes[2].plot(df_site['date'], df_site['sd'], color='C3', lw=1)
    axes[2].set_ylabel('SD')
    axes[2].set_xlabel('Year')

    for ax in axes:
        ax.grid(True, color='0.9', lw=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{site}_gpp_residual_sd.png', dpi=300)
    plt.close(fig)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: load data
df = pd.read_csv(INPUT_CSV)
df['GPP'] = pd.to_numeric(df['GPP_CUT'], errors='coerce')

# Step 2: calculate residuals and rolling SD for each site
all_sites = []
for site, df_site in df.groupby('Site'):
    if len(df_site) < MIN_PERIODS:
        continue

    df_site = df_site.sort_values('Time').copy()
    df_site['date'] = pd.to_datetime(df_site['Time'].astype(str), format='%Y%m')
    df_site['gpp_resid'] = calc_gpp_residual(df_site)
    df_site['sd'] = (
        df_site['gpp_resid']
        .rolling(SD_WINDOW, min_periods=MIN_PERIODS, center=True)
        .apply(calc_sd, raw=False)
    )
    df_site.loc[df_site.index[:12], 'sd'] = np.nan
    df_site.loc[df_site.index[-12:], 'sd'] = np.nan

    save_site_figure(df_site)
    all_sites.append(df_site)
    print(f'Processed: {site}')

# Step 3: save monthly residuals and SD
df_out = pd.concat(all_sites, ignore_index=True)
df_out.to_csv('/Volumes/Zhengfei_01/Project 2 pf resilience/1_Input/pf_flux_sites_monthly_residual_sd_final.csv', index=False)
print(f'Wrote figures and monthly SD csv to: {OUTPUT_DIR}')
