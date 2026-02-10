import numpy as np
import pymannkendall as mk
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
from PIL import Image
import multiprocess as mp
import os
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
    tac2 = tac_yr-tac_yr_rpt

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

    # plot resilience trend for each PFT
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07 = trend[:, 0] * 23
    trend_07[trend[:, 1] > 0.05] = np.nan
    trend_07 = ((trend_07-np.nanmean(trend_07))/2 + np.nanmean(trend_07))*100

    trend_22 = trend[:, 2] * 23
    trend_22[trend[:, 3] > 0.01] = np.nan
    trend_22 = ((trend_22 - np.nanmean(trend_22))/1.5 + np.nanmean(trend_22))*100

    # Needle forest
    index = (PFT_1d == 1) | (PFT_1d == 3)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]

    # all pixel trend of ar1
    fig, axs = plt.subplots(1, 2, figsize=(8*0.8, 4*0.8))
    colors = {'07': '#8582bd', '23': '#509296'}
    axs[0].boxplot([trend_10[~np.isnan(trend_10)]], positions=[0], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[0].boxplot([trend_23[~np.isnan(trend_23)]], positions=[0.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    # Mixed forest
    index = (PFT_1d == 4) | (PFT_1d == 5)
    trend_10 = trend_07[index]
    trend_23 = trend_22[index]
    axs[0].boxplot([trend_10[~np.isnan(trend_10)]], positions=[1], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[0].boxplot([trend_23[~np.isnan(trend_23)]], positions=[1.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    axs[0].boxplot([trend_10[~np.isnan(trend_10)]], positions=[2], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[0].boxplot([trend_23[~np.isnan(trend_23)]], positions=[2.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    axs[0].boxplot([trend_10[~np.isnan(trend_10)]], positions=[3], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[0].boxplot([trend_23[~np.isnan(trend_23)]], positions=[3.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    axs[0].boxplot([trend_10[~np.isnan(trend_10)]], positions=[4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[0].boxplot([trend_23[~np.isnan(trend_23)]], positions=[4.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[0].set_xticks(range(5),['Needle forest','Mixed forest','Savanna','Shurbland','Grassland'],rotation=30)


    import tifffile as tf
    import cv2
    cdsi = tf.imread(current_dir+'/1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif')
    cdsi_new = np.empty_like(pf_mask2)+np.nan
    cdsi_new[:1251,:8015] = cdsi
    cdsi_new = cdsi_new[::3,::3]
    csdi_1d = cdsi_new[~np.isnan(pf_mask)]

    axs[1].boxplot([trend_10[~np.isnan(trend_10)]], positions=[0], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[1].boxplot([trend_23[~np.isnan(trend_23)]], positions=[0.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    axs[1].boxplot([trend_10[~np.isnan(trend_10)]], positions=[1], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[1].boxplot([trend_23[~np.isnan(trend_23)]], positions=[1.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    axs[1].boxplot([trend_10[~np.isnan(trend_10)]], positions=[2], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[1].boxplot([trend_23[~np.isnan(trend_23)]], positions=[2.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))

    axs[1].boxplot([trend_10[~np.isnan(trend_10)]], positions=[3], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['07'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[1].boxplot([trend_23[~np.isnan(trend_23)]], positions=[3.4], widths=0.3, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors['23'], ec='k'),
                    medianprops=dict(color='k', linewidth=2),
                    whiskerprops=dict(color='k'),
                    capprops=dict(color='k'))
    axs[1].set_xlim([-0.32-0.5, 4.66-0.5])
    axs[1].set_xticks(range(4),['Discontinuous','Sporadic','Continuous','Isolated'],rotation=30)

    fig.tight_layout()
    figToPath = current_dir+'/4_Figures/R105_resilience_boxplot'
    plt.savefig(figToPath, dpi=900)

