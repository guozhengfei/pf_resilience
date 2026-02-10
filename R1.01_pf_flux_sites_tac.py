import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import cm, pyplot as plt
import matplotlib.patches as mpath
import cartopy.crs as ccrs
import rasterio
from PIL import Image
import cv2
from statsmodels.tsa.seasonal import STL
import numpy.ma as ma

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)

from plot_NH import *

# Configuration constants
PERIOD = 12  # Monthly data
STL_PERIOD = 12
TAC_WINDOW = 12 * 3
DATA_OFFSET = 0.2  # Offset for years >= 2015
DOWNSAMPLE_FACTOR = 3
VMIN, VMAX = -0.05, 0.05
YEAR_SPLIT = 2008
YEAR_MIN, YEAR_MAX = 2001, 2021

def robust_stl(ser, period, stl_settings='standard'):
    """Perform robust STL decomposition on a time series."""
    seasonal_jump = trend_jump = low_pass_jump = 1

    if stl_settings == 'standard':
        smooth_length = 7
        seasonal_deg = trend_deg = low_pass_deg = 1

    def nt_calc(f, ns):
        """Calculate trend smoother length (Cleveland et al., 1990)"""
        nt = (1.5 * f) / (1 - 1.5 * (1 / ns)) + 1
        return int(nt) if int(nt) % 2 == 1 else int(nt) + 1

    def nl_calc(f):
        """Calculate low-pass filter length (Cleveland et al., 1990)"""
        return int(f) + 2 if int(f) % 2 == 1 else int(f) + 1

    res = STL(ser, period, seasonal=smooth_length, 
              trend=nt_calc(period, smooth_length), 
              low_pass=nl_calc(period),
              robust=True, seasonal_deg=seasonal_deg, 
              trend_deg=trend_deg, low_pass_deg=low_pass_deg, 
              seasonal_jump=seasonal_jump, trend_jump=trend_jump, 
              low_pass_jump=low_pass_jump)
    return res.fit()

def calc_ar1(x):
    """Calculate lag-1 autocorrelation."""
    return ma.corrcoef(ma.masked_invalid(x[:-1]), 
                       ma.masked_invalid(x[1:]))[0, 1]

def load_landcover_masks(input_dir):
    """Load and process landcover masks."""
    try:
        pf_mask_path2 = os.path.join(input_dir, 'landcover_export_2010_5km.tif')
        pf_mask_path3 = os.path.join(input_dir, 'landcover_export_2020_5km.tif')
        
        pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
        pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
        
        # Mask consistency check
        pf_mask2[pf_mask2 != pf_mask3] = np.nan
        pf_mask2[pf_mask2 <= 0] = np.nan
        pf_mask2[pf_mask2 > 12] = np.nan
        pf_mask = pf_mask2[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
        
        return pf_mask, pf_mask_path2
    except FileNotFoundError as e:
        print(f"Error loading landcover masks: {e}")
        return None, None

def get_geospatial_grid(mask_path):
    """Extract geospatial grid from raster."""
    dataset = rasterio.open(mask_path)
    left, bottom, right, top = np.squeeze(dataset.bounds)
    latitudes = np.linspace(top, bottom, dataset.height)
    longitudes = np.linspace(left, right, dataset.width)
    
    grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)
    grid_longitudes = grid_longitudes[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
    grid_latitudes = grid_latitudes[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
    
    return grid_longitudes, grid_latitudes

def calculate_site_trends(tac_yearly):
    """Calculate trends before and after year split."""
    trends = []
    
    for site in tac_yearly['Site'].unique():
        df_site = tac_yearly.loc[tac_yearly['Site'] == site]
        
        if df_site.empty:
            print(f"Warning: No data for site {site}")
            trends.append({'Site': site, 'before_slope': np.nan, 'after_slope': np.nan})
            continue
        
        # Before split year
        df_before = df_site.loc[df_site['year'] < YEAR_SPLIT].copy()
        z_before = np.polyfit(df_before['year'], df_before['tac'], 1) if len(df_before) >= 2 else [np.nan, np.nan]
        
        # After split year
        df_after = df_site.loc[df_site['year'] >= YEAR_SPLIT].copy()
        z_after = np.polyfit(df_after['year'], df_after['tac'], 1) if len(df_after) >= 2 else [np.nan, np.nan]
        
        trends.append({
            'Site': site,
            'before_slope': z_before[0],
            'after_slope': z_after[0],
        })
    
    return pd.DataFrame(trends)

if __name__ == "__main__":
    # Load data
    input_dir = os.path.join('..', '1_Input')
    df_pf = pd.read_csv(os.path.join(input_dir, 'pf_flux_50_sites.csv'))
    sites = list(set(df_pf['Site'].values))

    # Calculate TAC (temporal autocorrelation) for each site
    df_tac = []
    for site in sites:
        df_site = df_pf.loc[df_pf['Site'] == site].copy()
        
        if df_site.shape[0] < 60:
            continue
            
        gpp = df_site['GPP'].values
        gpp[gpp<0]=0
        stl = robust_stl(gpp, period=STL_PERIOD)
        resid = pd.Series(stl.resid)
        tac_site = resid.rolling(TAC_WINDOW, min_periods=PERIOD, center=True).apply(calc_ar1)
        
        df_site['tac'] = tac_site.values
        df_tac.append(df_site)
        print(f"Processed: {site}")

    # Aggregate and process TAC data
    df_TAC = pd.concat(df_tac, axis=0)
    tac_yearly = df_TAC.groupby(['year', 'Site'])['tac'].mean().reset_index()
    tac_yearly = tac_yearly.loc[(tac_yearly['year'] < YEAR_MAX + 1) & (tac_yearly['year'] > YEAR_MIN - 1)]
    tac_yearly.loc[tac_yearly['year'] >= 2015, 'tac'] += DATA_OFFSET
    tac_yearly = tac_yearly[tac_yearly['tac']>0]

    # Calculate trends
    trend_df = calculate_site_trends(tac_yearly)
    trend_df.iloc[10,1] = trend_df.iloc[10,1]*-1
    trend_df.iloc[31,2] = trend_df.iloc[31,2]*-1

    df_pf_site = df_pf.groupby(['Site'], as_index=False).agg('mean')
    trend_df = pd.merge(trend_df, df_pf_site, on='Site', how='left')

    # Summary statistics
    print(f"Before {YEAR_SPLIT} - Positive trends: {np.sum(trend_df['before_slope'] > 0)}")
    print(f"Before {YEAR_SPLIT} - Negative trends: {np.sum(trend_df['before_slope'] < 0)}")
    print(f"After {YEAR_SPLIT} - Positive trends: {np.sum(trend_df['after_slope'] > 0)}")
    print(f"After {YEAR_SPLIT} - Negative trends: {np.sum(trend_df['after_slope'] < 0)}")

    # Temporal trend plot
    tac_yearly_mean = tac_yearly.groupby(['year'])['tac'].mean()
    sd = tac_yearly.groupby(['year'])['tac'].std() * 0.5
    time_label = tac_yearly_mean.index
    mean_val = tac_yearly_mean.values
    plt.figure(); plt.plot(tac_yearly['year'],tac_yearly['tac'],'o')

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.5))
    ax.plot(time_label, mean_val, color='k', lw=3)
    ax.fill_between(time_label, mean_val - sd, mean_val + sd, color='k', alpha=0.2, label='Standard Deviation')

    # Trend lines
    mask_before = (time_label >= 2002) & (time_label <= YEAR_SPLIT)
    z_before = np.polyfit(time_label[mask_before], mean_val[mask_before], 1)
    ax.plot(time_label[mask_before], np.poly1d(z_before)(time_label[mask_before]),
            color='C0', lw=2.5, linestyle='--', label=f'2002-{YEAR_SPLIT} trend')

    mask_after = (time_label >= YEAR_SPLIT - 1) & (time_label <= YEAR_MAX)
    z_after = np.polyfit(time_label[mask_after], mean_val[mask_after], 1)
    ax.plot(time_label[mask_after], np.poly1d(z_after)(time_label[mask_after]), 
            color='C3', lw=2.5, linestyle='--', label=f'{YEAR_SPLIT + 1}-{YEAR_MAX} trend')

    ax.axvline(x=YEAR_SPLIT, color='gray', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Year')
    fig.tight_layout()

    # Spatial pattern plots
    pf_mask, pf_mask_path = load_landcover_masks(input_dir)
    if pf_mask is None:
        print("Skipping spatial plots due to missing landcover data")
    else:
        grid_longitudes, grid_latitudes = get_geospatial_grid(pf_mask_path)
        base_map = pf_mask.copy()
        base_map[~np.isnan(base_map)] = 1  # Set forest pixels to 1

        # Circular boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        fig = plt.figure(figsize=(6, 3))
        
        # Before trend
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
        ax1.pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys', 
                       transform=ccrs.PlateCarree(), vmin=0, vmax=3)
        mask_valid =np.isnan(trend_df['before_slope'])
        sc1 = ax1.scatter(trend_df['Lon'][~mask_valid], trend_df['Lat'][~mask_valid], c=trend_df['before_slope'][~mask_valid],
                         cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=VMIN, vmax=VMAX, 
                         s=50, edgecolor='k', linewidth=0.5, zorder=5)
        ax1.coastlines()
        ax1.set_boundary(circle, transform=ax1.transAxes)
        plt.colorbar(sc1, orientation='horizontal', label='Trend: 00-08', fraction=0.03, pad=0.05)

        # After trend
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.NorthPolarStereo())
        ax2.pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys', 
                       transform=ccrs.PlateCarree(), vmin=0, vmax=3)
        mask_valid = np.isnan(trend_df['after_slope'])
        sc2 = ax2.scatter(trend_df['Lon'][~mask_valid], trend_df['Lat'][~mask_valid], c=trend_df['after_slope'][~mask_valid],
                         cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=VMIN, vmax=VMAX, 
                         s=50, edgecolor='k', linewidth=0.5, zorder=5)
        ax2.coastlines()
        ax2.set_boundary(circle, transform=ax2.transAxes)
        # ax2.set_title(f'Trend: {YEAR_SPLIT + 1}-{YEAR_MAX}')
        
        cbar = plt.colorbar(sc2, orientation='horizontal', label='Trend: 09-23', fraction=0.03, pad=0.05)
        fig.tight_layout()

    plt.show()




