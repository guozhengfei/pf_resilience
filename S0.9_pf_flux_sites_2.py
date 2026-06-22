from pathlib import Path

import pandas as pd


DATA_DIR = Path('/Volumes/Zhengfei_02/Fluxnet_2025')
SITE_CSV = DATA_DIR / 'pf_fluxnet_sites_5yr.csv'
OUTPUT_CSV = DATA_DIR / 'pf_fluxnet_monthly_gpp_5yr.csv'
GPP_COLUMNS = ['GPP_DT_VUT_REF', 'GPP_DT_CUT_REF']
YEAR_MAX = 2023


sites = pd.read_csv(SITE_CSV)
all_data = []

for _, site_info in sites.iterrows():
    site = site_info['site_id']
    folders = [p for p in DATA_DIR.glob(f'*_{site}_FLUXNET_*')
               if p.is_dir() and not p.name.startswith('._')]

    if not folders:
        print(f'Skip {site}: no site folder')
        continue

    flux_files = [p for p in folders[0].glob('*_FLUXMET_MM_*.csv')
                  if not p.name.startswith('._')]
    if not flux_files:
        print(f'Skip {site}: no monthly flux file')
        continue

    columns = pd.read_csv(flux_files[0], nrows=0).columns
    if not all(col in columns for col in GPP_COLUMNS):
        print(f'Skip {site}: missing daytime GPP columns')
        continue

    flux = pd.read_csv(flux_files[0], usecols=['TIMESTAMP', *GPP_COLUMNS])
    out = pd.DataFrame({
        'Time': flux['TIMESTAMP'].astype(int),
        'GPP_VUT': pd.to_numeric(flux['GPP_DT_VUT_REF'], errors='coerce').replace(-9999, pd.NA),
        'GPP_CUT': pd.to_numeric(flux['GPP_DT_CUT_REF'], errors='coerce').replace(-9999, pd.NA),
        'Site': site,
        'Lon': site_info['lon'],
        'Lat': site_info['lat'],
        'igbp': site_info['igbp']
    })
    out['year'] = out['Time'] // 100
    out['month'] = out['Time'] % 100
    out = out[out['year'] <= YEAR_MAX]

    all_data.append(out)
    print(f'Processed: {site}')

monthly_gpp = pd.concat(all_data, ignore_index=True)
monthly_gpp = monthly_gpp.sort_values(['Site', 'Time']).reset_index(drop=True)
monthly_gpp.to_csv(OUTPUT_CSV, index=False)

print(f'Sites processed: {monthly_gpp["Site"].nunique()}')
print(f'Rows saved: {len(monthly_gpp)}')
print(f'Wrote: {OUTPUT_CSV}')
