import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; #matplotlib.use('Qt5Agg')
import tifffile as tf
#from plot_NH import *
import os
from PIL import Image
import rasterio
import cv2
import scipy.signal as ss

def extend_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result > 0] = 1
    return result
if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/NDVI_pf_16d/'

    # extract pf area
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

    # burned area
    fire_freq = tf.imread(current_dir + '/1_Input/fire_frequency_pf.tif').astype(float)[::-1, :]
    fire_freq_rsz = np.zeros_like(pf_mask2)
    fire_freq_rsz[:-1, :-1] = fire_freq
    fire_freq_rsz[fire_freq_rsz == 0] = np.nan
    fire_freq = fire_freq_rsz[::3, ::3]
    fire_freq[np.isnan(pf_mask)] = np.nan
    fire_freq[fire_freq>3]=3

    # load ar1 data
    fire_map = fire_freq[::-1, :] * 1
    fire_map[fire_map > 3] = 3

    # burn area and date
    fire_annual0 = tf.imread(current_dir + '/1_Input/burnedArea_each_year-0.tif').astype(float)
    fire_annual1 = tf.imread(current_dir + '/1_Input/burnedArea_each_year-1.tif').astype(float)
    fire_annual = np.hstack((fire_annual0, fire_annual1))[::-1, :, :]
    fire_annual_rsz = np.zeros((pf_mask2.shape[0], pf_mask2.shape[1], 23))
    fire_annual_rsz[:-1, :-1, :] = fire_annual
    fire_annual_rsz[fire_annual_rsz == 0] = np.nan
    fire_annual = fire_annual_rsz[::3, ::3, :]
    fire_annual[np.isnan(pf_mask), :] = np.nan
    fire_annual = fire_annual[~np.isnan(pf_mask), :]
    fire_freq = np.nansum(fire_annual, axis=1)
    fire_freq[fire_freq == 0] = np.nan

    ar1 = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1[:, :-2], ar1[:, -3:]))
    tac[tac == 0] = np.nan;
    fire_freq_1d = fire_freq * 1

    nofire_around = fire_map*1
    for i in range(1):
        nofire_around = extend_edge(nofire_around)
    nofire_around[~np.isnan(fire_map)]=np.nan
    nofire_around[np.isnan(pf_mask)]=np.nan
    nofire_around_1d = nofire_around[~np.isnan(pf_mask)]


    ar1_fireArea = tac[~np.isnan(fire_freq_1d), :]
    ar1_nonfireArea = tac[~np.isnan(nofire_around_1d), :]
    ar1_nofire_mean = np.nanmean(np.nanmean(ar1_nonfireArea, axis=0))*0.97
    ar1_nofire_sd = np.nanmean(np.nanstd(ar1_nonfireArea, axis=0)) / 23

    ar1_fire1_mean = np.nanmean(np.nanmean(tac[fire_freq_1d == 1],axis=0))
    ar1_fire1_sd = np.nanmean(np.nanstd(tac[fire_freq_1d == 1], axis=0))/23

    ar1_fire2_mean = np.nanmean(np.nanmean(tac[fire_freq_1d == 2], axis=0))
    ar1_fire2_sd = np.nanmean(np.nanstd(tac[fire_freq_1d == 2], axis=0))/23

    ar1_fire3_mean = np.nanmean(np.nanmean(tac[fire_freq_1d >= 3], axis=0))
    ar1_fire3_sd = np.nanmean(np.nanstd(tac[fire_freq_1d >= 3], axis=0))/23

    ar1_fire1 = tac[fire_freq_1d == 1]
    fire_date1 = fire_annual[fire_freq_1d == 1]
    indices_of_1 = np.array([np.where(row == 1)[0][0] if np.any(row == 1) else -1 for row in fire_date1])
    # effect of fire on ndvi and anomaly
    evi_filenames = os.listdir(EVI_folder)
    evi_filenames = np.sort(evi_filenames)  # [:529]

    EVI = []
    yr_num = 24
    bands_year = 23
    for name in evi_filenames:
        src = rasterio.open(EVI_folder + name)
        evi = src.read(1)[::3, ::3]
        evi[evi < 0] = 0
        evi_val = evi[~np.isnan(pf_mask)]
        evi_sq = evi_val ** 2
        kndvi = np.tanh(evi_sq)
        EVI.append(kndvi)
        print(name)
    EVI = np.array(EVI).T

    EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
    EVI_sg[EVI_sg < 0] = 0
    ser = EVI_sg.T


    del EVI
    EVI_yr = np.zeros_like(ser)

    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(ser[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = ser - EVI_yr

    Evi_sea_rep = np.zeros_like(rm_offline)
    for yr in range(yr_num):
        start_index = (yr - 5) * bands_year
        end_index = (yr + 5) * bands_year
        if yr < 5: start_index = 0; end_index = 10 * bands_year
        if yr > (yr_num - 5): start_index = (yr_num - 10) * bands_year; end_index = yr_num * bands_year
        data_i = rm_offline[:, start_index:end_index]
        Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / 23), bands_year)), axis=1)
        Evi_sea_rep[:, yr * 23:(yr + 1) * 23] = Evi_sea
    res = rm_offline - Evi_sea_rep
    res[np.isnan(res)] = 0
    # res = res[::100, :]
    evi_fire1 = ser[fire_freq_1d == 1,:]
    res_fire1 = res[fire_freq_1d == 1, :]

    # fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True)
    evi_fire1_rsp = evi_fire1.reshape(evi_fire1.shape[0],yr_num,bands_year)
    evi_fire1_yr = np.nanmean(evi_fire1_rsp,axis=2)

    fig, axs = plt.subplots(5, 4, figsize=(16 * 0.6, 12 * 0.6))
    i = 0
    for row in range(5):
        for col in range(4):
            axs[row,col].plot(np.linspace(2000, 2023 , yr_num)[2:-1], np.nanmean(evi_fire1_yr[indices_of_1 == i+1, :], axis=0)[2:-1], 'o-',c='#d6604d')

            maxV = np.nanmean(evi_fire1_yr[indices_of_1 == i+1, :], axis=0)[2:-1].max()+0.02
            minV = np.nanmean(evi_fire1_yr[indices_of_1 == i + 1, :], axis=0)[2:-1].min() - 0.02
            axs[row,col].set_ylim([minV,maxV])
            axs[row,col].axvline(x=i+2003, color='red', linestyle='--')
            axs[row,col].text(2012, maxV-0.02, 'x='+str(i+2003), color='k', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
            i=i+1
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS02_fire_ndvi'
    plt.savefig(figToPath, dpi=900)

    fig, axs = plt.subplots(5, 4, figsize=(16 * 0.6, 12 * 0.6))
    i = 0
    for row in range(5):
        for col in range(4):
            axs[row, col].plot(np.linspace(2000, 2023 + 22 / 23, 552)[46:-23],
             np.nanmedian(ar1_fire1[indices_of_1 == i+1, :], axis=0)[46:-23], '#4393c3')

            maxV = np.nanmedian(ar1_fire1[indices_of_1 == i+1, :], axis=0)[46:-46].max() + 0.05
            minV = np.nanmedian(ar1_fire1[indices_of_1 == i + 1, :], axis=0)[46:-46].min() - 0.05
            axs[row, col].set_ylim([minV, maxV])
            axs[row, col].axvline(x=i + 2003, color='red', linestyle='--')
            axs[row, col].text(2012, maxV - 0.05, 'x=' + str(i + 2003), color='k', fontsize=12,
                               verticalalignment='bottom', horizontalalignment='right')
            i = i + 1
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS02_fire_tac'
    plt.savefig(figToPath, dpi=900)
