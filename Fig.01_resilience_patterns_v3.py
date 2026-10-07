import numpy as np

from scipy.stats import sem, t
import pymannkendall as mk
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
from PIL import Image
import multiprocess as mp
import os


# Pointwise intervals assume independent sampling units (pixels or DGVMs).
def standard_error(values, axis=0):
    """Sample SD (ddof=1) / sqrt(non-missing n), along the sampling axis."""
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any() or np.any(np.isfinite(values).sum(axis=axis) < 2):
        raise ValueError("SE requires at least two finite observations per estimate.")
    return sem(values, axis=axis, ddof=1, nan_policy="omit")


def mean_ci95(values, axis=0):
    """Return the mean, Student-t 95% CI half-width and non-missing sample count."""
    values = np.asarray(values, dtype=float)
    n = np.isfinite(values).sum(axis=axis)
    half_width = t.ppf(0.975, n - 1) * standard_error(values, axis=axis)
    return np.nanmean(values, axis=axis), half_width, n

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr

if __name__ == '__main__':
    # load ar1 data: 5years window
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    ar1_5yr_modis = np.load(current_dir+'/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:,-3:]))
    tac[tac == 0] = np.nan; tac[:, 0]
    tac_yr = tac.reshape(tac.shape[0], 24, 23)
    tac_yr = np.nanmean(tac_yr, axis=2)[:,2:-2]
    tac_yr_rpt = np.tile(tac_yr[:,0],(tac_yr.shape[1],1)).T
    tac2 = tac_yr#-tac_yr_rpt

    # all pixel trend of ar1
    fig, axs = plt.subplots(1,3,figsize=(12*0.8,3*0.8))

    time_label_modis = np.linspace(2002,2021,20)


    # Each pixel contributes one annual TAC value; omit missing pixels.
    se = standard_error(tac2, axis=0)
    _, tac_ci95_half_width, tac_n = mean_ci95(tac2, axis=0)

    ## PFT information
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]
    
    PFT_1d = pf_mask[~np.isnan(pf_mask)]
    index = (PFT_1d==1) | (PFT_1d==3)
    NF = np.nanmean(tac2[index,:],axis=0)
    axs[0].plot(time_label_modis,NF,c='#d6604d')

    index = (PFT_1d==4) | (PFT_1d==5)
    MF = np.nanmean(tac2[index,:],axis=0)
    axs[0].plot(time_label_modis,MF,c='#f4a582')

    index = (PFT_1d==6) | (PFT_1d==7)
    SAV = np.nanmean(tac2[index,:],axis=0)
    axs[0].plot(time_label_modis,SAV,c='#92c5de')

    index = (PFT_1d==8) | (PFT_1d==9)
    SHR = np.nanmean(tac2[index,:],axis=0)
    axs[0].plot(time_label_modis, SHR,c='#4393c3')

    index = (PFT_1d==10)
    GRA = np.nanmean(tac2[index,:],axis=0)
    axs[0].plot(time_label_modis,GRA,c='#878787')

    mean_val_modis = np.nanmean(tac2, axis=0)
    axs[0].plot(time_label_modis, mean_val_modis, color='k', lw=3)
    axs[0].fill_between(time_label_modis, mean_val_modis - tac_ci95_half_width,
                        mean_val_modis + tac_ci95_half_width, color='k',
                        alpha=0.2, label='95% CI')

    # plot resilience trend for each PFT
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07 = trend[:, 0] * 23
    trend_07[trend[:, 1] > 0.05] = np.nan

    trend_22 = trend[:, 2] * 23
    trend_22[trend[:, 3] > 0.01] = np.nan


    # Needle forest
    index = (PFT_1d == 1) | (PFT_1d == 3)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[1].bar(0, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
            alpha=0.7)
    axs[1].bar(0.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k', alpha=0.7)

    # Mixed forest
    index = (PFT_1d == 4) | (PFT_1d == 5)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[1].bar(1, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
            alpha=0.7)
    axs[1].bar(1.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k', alpha=0.7)
    # np.sum(trend_10 > 0) / np.sum(~np.isnan(trend_10))

    # Shrubland
    index = (PFT_1d == 6) | (PFT_1d == 7)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[1].bar(2, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
            alpha=0.7)
    axs[1].bar(2.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k', alpha=0.7)

    # Savana
    index = (PFT_1d == 8) | (PFT_1d == 9)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[1].bar(3, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
            alpha=0.7)
    axs[1].bar(3.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k', alpha=0.7)

    # Grass
    index = (PFT_1d == 10)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[1].bar(4, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
            alpha=0.7)
    axs[1].bar(4.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k', alpha=0.7)
    axs[1].set_xticks(range(5),['NF','MF','SVA','SHR','GRA'])


    import tifffile as tf
    import cv2
    cdsi = tf.imread(current_dir+'/1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif')
    cdsi_new = np.empty_like(pf_mask2)+np.nan
    cdsi_new[:1251,:8015] = cdsi
    cdsi_new = cdsi_new[::3,::3]
    csdi_1d = cdsi_new[~np.isnan(pf_mask)]

    index = (csdi_1d == 0)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[2].bar(0, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
               alpha=0.7)
    axs[2].bar(0.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k',
               alpha=0.7)

    index = (csdi_1d == 1)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[2].bar(1, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
               alpha=0.7)
    axs[2].bar(1.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k',
               alpha=0.7)

    index = (csdi_1d == 2)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[2].bar(2, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
               alpha=0.7)
    axs[2].bar(2.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k',
               alpha=0.7)

    index = (csdi_1d == 3)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[2].bar(3, np.nanmean(trend_10), yerr=standard_error(trend_10), color='#4393c3', width=0.3, ec='k', hatch='//',
               alpha=0.7)
    axs[2].bar(3.3, np.nanmean(trend_23), yerr=standard_error(trend_23), color='#4393c3', width=0.3, ec='k',
               alpha=0.7)
    axs[2].set_xlim([-0.32-0.5,4.66-0.5])
    axs[2].set_xticks(range(4),['D','S','C','I'])

    fig.tight_layout()
    # figToPath = current_dir+'/4_Figures/Fig01_resilience_patterns_temporal'
    # plt.savefig(figToPath, dpi=900)
    #
    # Export temporal patterns data to CSV
    import pandas as pd

    # Create dataframe for temporal patterns (panel a)
    temporal_data = {
        'Year': time_label_modis,
        'Needle_Forest': NF,
        'Mixed_Forest': MF,
        'Savanna': SAV,
        'Shrubland': SHR,
        'Grassland': GRA,
        'All_Mean': mean_val_modis,
        'Standard_Error': se,
        'CI95_HalfWidth': tac_ci95_half_width,
        'N_Pixels': tac_n
    }
    temporal_df = pd.DataFrame(temporal_data)
    temporal_df.to_csv(current_dir + '/4_Figures/Fig.1/Fig01_temporal_patterns_2026.csv', index=False)
    #
    # # Create dataframe for PFT trends (panel b)
    # pft_names = ['Needle Forest', 'Mixed Forest', 'Savanna', 'Shrubland', 'Grassland']
    # pft_trends = {
    #     'PFT': pft_names,
    #     'Trend_2002_2007': [
    #         np.nanmean(trend_07[(PFT_1d == 1) | (PFT_1d == 3)]),
    #         np.nanmean(trend_07[(PFT_1d == 4) | (PFT_1d == 5)]),
    #         np.nanmean(trend_07[(PFT_1d == 6) | (PFT_1d == 7)]),
    #         np.nanmean(trend_07[(PFT_1d == 8) | (PFT_1d == 9)]),
    #         np.nanmean(trend_07[PFT_1d == 10])
    #     ],
    #     'Error_2002_2007': [
    #         standard_error(trend_07[(PFT_1d == 1) | (PFT_1d == 3)]),
    #         standard_error(trend_07[(PFT_1d == 4) | (PFT_1d == 5)]),
    #         standard_error(trend_07[(PFT_1d == 6) | (PFT_1d == 7)]),
    #         standard_error(trend_07[(PFT_1d == 8) | (PFT_1d == 9)]),
    #         standard_error(trend_07[PFT_1d == 10])
    #     ],
    #     'Trend_2008_2022': [
    #         np.nanmean(trend_22[(PFT_1d == 1) | (PFT_1d == 3)]),
    #         np.nanmean(trend_22[(PFT_1d == 4) | (PFT_1d == 5)]),
    #         np.nanmean(trend_22[(PFT_1d == 6) | (PFT_1d == 7)]),
    #         np.nanmean(trend_22[(PFT_1d == 8) | (PFT_1d == 9)]),
    #         np.nanmean(trend_22[PFT_1d == 10])
    #     ],
    #     'Error_2008_2022': [
    #         standard_error(trend_22[(PFT_1d == 1) | (PFT_1d == 3)]),
    #         standard_error(trend_22[(PFT_1d == 4) | (PFT_1d == 5)]),
    #         standard_error(trend_22[(PFT_1d == 6) | (PFT_1d == 7)]),
    #         standard_error(trend_22[(PFT_1d == 8) | (PFT_1d == 9)]),
    #         standard_error(trend_22[PFT_1d == 10])
    #     ]
    # }
    # pft_trends_df = pd.DataFrame(pft_trends)
    # pft_trends_df.to_csv(current_dir + '/4_Figures/Fig.1/Fig01_pft_trends.csv', index=False)
    #
    # # Create dataframe for CDSI trends (panel c)
    # cdsi_names = ['Discontinuous', 'Sporadic', 'Continuous', 'Isolated']
    # cdsi_trends = {
    #     'CDSI_Class': cdsi_names,
    #     'Trend_2002_2007': [
    #         np.nanmean(trend_07[csdi_1d == 0]),
    #         np.nanmean(trend_07[csdi_1d == 1]),
    #         np.nanmean(trend_07[csdi_1d == 2]),
    #         np.nanmean(trend_07[csdi_1d == 3])
    #     ],
    #     'Error_2002_2007': [
    #         standard_error(trend_07[csdi_1d == 0]),
    #         standard_error(trend_07[csdi_1d == 1]),
    #         standard_error(trend_07[csdi_1d == 2]),
    #         standard_error(trend_07[csdi_1d == 3])
    #     ],
    #     'Trend_2008_2022': [
    #         np.nanmean(trend_22[csdi_1d == 0]),
    #         np.nanmean(trend_22[csdi_1d == 1]),
    #         np.nanmean(trend_22[csdi_1d == 2]),
    #         np.nanmean(trend_22[csdi_1d == 3])
    #     ],
    #     'Error_2008_2022': [
    #         standard_error(trend_22[csdi_1d == 0]),
    #         standard_error(trend_22[csdi_1d == 1]),
    #         standard_error(trend_22[csdi_1d == 2]),
    #         standard_error(trend_22[csdi_1d == 3])
    #     ]
    # }
    # cdsi_trends_df = pd.DataFrame(cdsi_trends)
    # cdsi_trends_df.to_csv(current_dir + '/4_Figures/Fig.1/Fig01_cdsi_trends.csv', index=False)