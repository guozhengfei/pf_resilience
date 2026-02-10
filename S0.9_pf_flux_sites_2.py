import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from pathlib import Path

def cal_monthly_mean(flux_df,output_columns):
    time_label = flux_df['TIMESTAMP_START']

    # parse TIMESTAMP_START values like "202412312330" -> datetime and split into components
    time_str = time_label.astype(str)  # ensure string
    ts = pd.to_datetime(time_str, format='%Y%m%d%H%M', errors='coerce')  # NaT for invalid
    flux_df['year'] = ts.dt.year
    flux_df['month'] = ts.dt.month
    flux_df['day'] = ts.dt.day
    flux_df['hour'] = ts.dt.hour
    flux_df['minute'] = ts.dt.minute

    for col in output_columns:
        try:
            flux_df.loc[flux_df[col] ==-9999, col] = np.nan
        except KeyError:
            flux_df[col] = np.nan
    # aggregate monthly mean for LE and H (group by year and month)
    monthly_mean_LE_H = flux_df.groupby(['year', 'month'])[output_columns].mean().reset_index()
    return monthly_mean_LE_H


def find_subfolders_with_patterns(root_dir, patterns):
    """
    Find all subfolders whose names include any of the specified patterns,
    while ignoring hidden folders (those starting with '.').

    Args:
        root_dir (str): The root directory to search in
        patterns (list): List of strings to match in folder names

    Returns:
        list: Path objects of matching subfolders
    """
    root_path = Path(root_dir)
    matching_folders = []

    if not root_path.exists():
        print(f"Error: Directory '{root_dir}' does not exist.")
        return matching_folders

    if not root_path.is_dir():
        print(f"Error: '{root_dir}' is not a directory.")
        return matching_folders

    for folder in root_path.rglob('*'):
        if folder.is_dir() and not folder.name.startswith('.'):
            for pattern in patterns:
                if pattern in folder.name:
                    matching_folders.append(folder)
                    break  # No need to check other patterns once one matches

    return matching_folders

# Example usage
if __name__ == "__main__":
    pf_ICOS = ['FI-Ouk', 'FI-Sod', 'GL-NuF', 'GL-ZaF', 'SE-Sto', 'FR-CLt',
       'FI-Var', 'CH-Dav', 'GL-ZaH', 'FI-Lom', 'FI-Ken', 'IT-Niv']
    pf_AMF = ['CA-ARB', 'CA-ARF', 'CA-CF1', 'CA-HPC', 'CA-Man', 'CA-Mtk',
       'CA-NS1', 'CA-NS2', 'CA-NS3', 'CA-NS4', 'CA-NS5', 'CA-NS6',
       'CA-NS7', 'CA-Qc2', 'CA-Qfo', 'CA-SCB', 'CA-SCC', 'CA-SF1',
       'CA-SMC', 'US-BZB', 'US-BZF', 'US-BZo', 'US-BZS', 'US-CAK',
       'US-Cms', 'US-EML', 'US-Fcr', 'US-GLE', 'US-HVs', 'US-ICh',
       'US-ICs', 'US-ICt', 'US-NGB', 'US-NGC', 'US-Prr', 'US-Rpf',
       'US-Sag', 'US-xBA', 'US-xBN', 'US-xDJ', 'US-xHE', 'US-xNW',
       'US-xTL', 'US-YK2']
    ICOS_site_info = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'ICOS', 'ICOS_stations.csv'))

    df_ICOS = []
    for site in pf_ICOS:
        subfolder = 'ICOSETC_'+site+'_ARCHIVE_L2'
        filename_flux = 'ICOSETC_'+site+'_FLUXNET_MM_L2.csv'
        try:
            flux_df = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'ICOS', subfolder,filename_flux))
        except FileNotFoundError:
            print('No monthly GPP')
            continue
        colum_list = flux_df.columns
        aim_str = [s for s in colum_list if 'GPP' in s and 'REF' in s]
        aim_columns = ['TIMESTAMP',aim_str[0]]
        merge_df = flux_df[aim_columns]

        merge_df['site']=site
        merge_df['lon'] = ICOS_site_info.loc[ICOS_site_info['site_ID'] == site, 'lon'].values[0]
        merge_df['lat'] = ICOS_site_info.loc[ICOS_site_info['site_ID'] == site, 'lat'].values[0]
        merge_df.columns = ['Time','GPP','Site', 'Lon', 'Lat']
        df_ICOS.append(merge_df)
        print(site)

    df_ICOS2 = pd.concat(df_ICOS,axis=0)
    time_str = df_ICOS2['Time'].astype(str)  # ensure string
    ts = pd.to_datetime(time_str, format='%Y%m', errors='coerce')  # NaT for invalid
    df_ICOS2['year'] = ts.dt.year
    df_ICOS2['month'] = ts.dt.month

    filenames = find_subfolders_with_patterns(os.path.join('..', '..', 'Fluxnet2025', 'AMF'), pf_AMF)
    AMF_site_info = pd.read_csv(os.path.join('..', '..', 'Fluxnet2025', 'AMF', 'AmeriFlux-site-info.csv'))
    df_AMF = []
    for filename in filenames:
        subfolder = str(filename)
        filename_flux0 = subfolder.split('/')[-1].split('SUBSET')
        filename_flux = filename_flux0[0]+'SUBSET'+'_MM'+filename_flux0[1]+'.csv'
        flux_df = pd.read_csv(os.path.join(subfolder, filename_flux))

        colum_list = flux_df.columns
        aim_str = [s for s in colum_list if 'GPP' in s and 'REF' in s]
        aim_columns = ['TIMESTAMP', aim_str[0]]
        merge_df = flux_df[aim_columns]

        site =filename_flux0[0].split('_')[1]
        merge_df['site'] = site
        merge_df['lon'] = AMF_site_info.loc[AMF_site_info['Site ID'] == site, 'lon'].values[0]
        merge_df['lat'] = AMF_site_info.loc[AMF_site_info['Site ID'] == site, 'lat'].values[0]
        merge_df.columns = ['Time', 'GPP', 'Site', 'Lon', 'Lat']
        df_AMF.append(merge_df)
        print(site)

    df_AMF2 = pd.concat(df_AMF, axis=0)
    time_str = df_AMF2['Time'].astype(str)  # ensure string
    ts = pd.to_datetime(time_str, format='%Y%m', errors='coerce')  # NaT for invalid
    df_AMF2['year'] = ts.dt.year
    df_AMF2['month'] = ts.dt.month

    df_pf = pd.concat([df_ICOS2,df_AMF2],axis=0)
    df_pf.to_csv(os.path.join('..', '1_Input', 'pf_flux_50_sites.csv'))


