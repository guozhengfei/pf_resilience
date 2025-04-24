import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import matplotlib;
#matplotlib.use('Qt5Agg')
from PIL import Image
import multiprocess as mp
import os

if __name__ == '__main__':
    # load ar1 data: 5years window
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    fig, axs = plt.subplots(1, 3, figsize=(12 * 0.8, 3 * 0.8))

    for i in range(3):
        ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling_'+str(i+3)+'.npy')
        tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:, -3:]))
        tac[tac == 0] = np.nan;
        tac[:, 0]
        tac_yr = tac.reshape(tac.shape[0], 24, 23)
        tac_yr = np.nanmean(tac_yr, axis=2)[:, 2:-2]
        tac_yr_rpt = np.tile(tac_yr[:, 0], (tac_yr.shape[1], 1)).T
        tac2 = tac_yr - tac_yr_rpt

        # all pixel trend of ar1
        time_label_modis = np.linspace(2002, 2021, 20)

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
        index = (PFT_1d == 1) | (PFT_1d == 3)
        NF = np.nanmean(tac2[index, :], axis=0)
        axs[i].plot(time_label_modis, NF, c='#d6604d')

        index = (PFT_1d == 4) | (PFT_1d == 5)
        MF = np.nanmean(tac2[index, :], axis=0)
        axs[i].plot(time_label_modis, MF, c='#f4a582')

        index = (PFT_1d == 6) | (PFT_1d == 7)
        SAV = np.nanmean(tac2[index, :], axis=0)
        axs[i].plot(time_label_modis, SAV, c='#92c5de')

        index = (PFT_1d == 8) | (PFT_1d == 9)
        SHR = np.nanmean(tac2[index, :], axis=0)
        axs[i].plot(time_label_modis, SHR, c='#4393c3')

        index = (PFT_1d == 10)
        GRA = np.nanmean(tac2[index, :], axis=0)
        axs[i].plot(time_label_modis, GRA, c='#878787')

        mean_val_modis = np.nanmean(tac2, axis=0)
        axs[i].plot(time_label_modis, mean_val_modis, color='k', lw=3)

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS05_window_size_rolling_deseason'
    plt.savefig(figToPath, dpi=900)

    # plot resilience trend for each PFT
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07 = trend[:, 0] * 23
    trend_07[trend[:, 1] > 0.05] = np.nan