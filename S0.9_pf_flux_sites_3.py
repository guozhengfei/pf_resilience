from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path('/Volumes/Zhengfei_02/Fluxnet_2025/pf_fluxnet_monthly_gpp_5yr.csv')
OUTPUT_CSV = Path('/Volumes/Zhengfei_01/Project 2 pf resilience/1_Input/pf_flux_60_sites_plus_additional_cleaned.csv')

GPP_COLUMNS = ['GPP_VUT', 'GPP_CUT']
GROWING_MONTHS = [5, 6, 7, 8, 9]
YEAR_MIN = 2000
YEAR_MAX = 2023
CLIM_WINDOW = 3   # use valid same-month values within +/- 3 years


def fill_with_same_month_climatology(df, col, window=3):
    """
    Fill missing values using same-site, same-month observations
    within +/- `window` years. If multiple valid observations exist,
    use their mean.
    """
    filled = df[col].copy()

    for (_, month), idx in df.groupby(['Site', 'month']).groups.items():
        monthly = df.loc[idx].sort_values('year')

        for row_idx, row in monthly[monthly[col].isna()].iterrows():
            candidate = monthly[
                monthly[col].notna()
                & ((monthly['year'] - row['year']).abs() <= window)
            ]

            if not candidate.empty:
                filled.loc[row_idx] = candidate[col].mean()

    return filled


# Step 1: read data and clean missing values
df = pd.read_csv(INPUT_CSV)

required_cols = {'Site', 'year', 'month', *GPP_COLUMNS}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')

for col in GPP_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].mask(df[col] == -9999, np.nan)

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['month'] = pd.to_numeric(df['month'], errors='coerce')

df = df.dropna(subset=['Site', 'year', 'month']).copy()
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)

# Step 2: restrict target years first
df = df[(df['year'] >= YEAR_MIN) & (df['year'] <= YEAR_MAX)].copy()

# Step 3: remove site-years where the whole year GPP_CUT is blank
# This is appropriate if GPP_CUT is the main product.
blank_years = (
    df.groupby(['Site', 'year'])['GPP_CUT']
    .apply(lambda x: x.isna().all())
    .reset_index(name='remove')
)

remove_keys = blank_years.loc[blank_years['remove'], ['Site', 'year']]

cleaned = df.merge(
    remove_keys.assign(remove=True),
    on=['Site', 'year'],
    how='left'
)

cleaned = cleaned[cleaned['remove'].isna()].drop(columns='remove').copy()

# Step 4: set non-growing-season GPP to 0
growing = cleaned['month'].isin(GROWING_MONTHS)
cleaned.loc[~growing, GPP_COLUMNS] = 0

# Step 5: fill remaining growing-season NaN values
fill_summary = {}

for col in GPP_COLUMNS:
    n_missing_before = cleaned.loc[growing, col].isna().sum()

    cleaned[col] = fill_with_same_month_climatology(
        cleaned,
        col,
        window=CLIM_WINDOW
    )

    n_missing_after_clim = cleaned.loc[growing, col].isna().sum()

    # Optional fallback: site-level growing-season mean
    cleaned.loc[growing, col] = cleaned.loc[growing, col].fillna(
        cleaned.loc[growing].groupby('Site')[col].transform('mean')
    )

    n_missing_after_site = cleaned.loc[growing, col].isna().sum()

    # Last fallback: month-level mean across all sites
    cleaned.loc[growing, col] = cleaned.loc[growing, col].fillna(
        cleaned.loc[growing].groupby('month')[col].transform('mean')
    )

    n_missing_after_month = cleaned.loc[growing, col].isna().sum()

    fill_summary[col] = {
        'missing_before': int(n_missing_before),
        'missing_after_same_month_pm3yr': int(n_missing_after_clim),
        'missing_after_site_mean': int(n_missing_after_site),
        'missing_after_month_mean': int(n_missing_after_month),
    }

# Step 6: sort and export
cleaned = cleaned.sort_values(['Site', 'year', 'month']).reset_index(drop=True)
cleaned.to_csv(OUTPUT_CSV, index=False)

print(f'Input rows after year filtering: {len(df)}')
print(f'Removed site-years: {len(remove_keys)}')
print(f'Output rows: {len(cleaned)}')
print(f'Wrote: {OUTPUT_CSV}')

print('\nFilling summary:')
for col, summary in fill_summary.items():
    print(col, summary)

print('\nRemaining missing values:')
print(cleaned[GPP_COLUMNS].isna().sum())