import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import rasterio
import tifffile as tf
import cv2

def ar1_series_4yr(array):
    yrs = 4  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    bands_year = 23

    t = bands_year*yrs
    ar1 = pd.Series(array).rolling(t, center=True).apply(calc_ar1).values
    return ar1


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
    fire_freq = tf.imread(current_dir + '/1_Input/fire_frequency_pf.tif').astype(float)[::-1,:]
    fire_freq_rsz = np.zeros_like(pf_mask2)
    fire_freq_rsz[:-1,:-1] = fire_freq
    fire_freq_rsz[fire_freq_rsz==0]=np.nan
    fire_freq = fire_freq_rsz[::3,::3]
    fire_freq[np.isnan(pf_mask)] = np.nan

    # burn area and date
    fire_annual0 = tf.imread(current_dir + '/1_Input/burnedArea_each_year-0.tif').astype(float)
    fire_annual1 = tf.imread(current_dir + '/1_Input/burnedArea_each_year-1.tif').astype(float)
    fire_annual = np.hstack((fire_annual0,fire_annual1))[::-1,:,: ]
    fire_annual_rsz = np.zeros((pf_mask2.shape[0],pf_mask2.shape[1],23))
    fire_annual_rsz[:-1,:-1,:] = fire_annual
    fire_annual_rsz[fire_annual_rsz == 0] = np.nan
    fire_annual = fire_annual_rsz[::3, ::3,:]
    fire_annual[np.isnan(pf_mask),:]=np.nan
    fire_annual = fire_annual[~np.isnan(pf_mask),:]
    fire_freq = np.nansum(fire_annual,axis=1)
    fire_freq[fire_freq==0] = np.nan


    plt.figure(); plt.hist(fire_freq.reshape(-1),50)

    ar1 = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1[:, :-2], ar1[:, -3:]))
    tac[tac == 0] = np.nan;

    fire_freq_1d = fire_freq*1

    ar1_fireArea = tac[~np.isnan(fire_freq_1d),:]
    plt.figure(); plt.plot(np.linspace(2000,2023+22/23,552),np.nanmean(ar1_fireArea,axis=0))

    ar1_nonfireArea = tac[np.isnan(fire_freq_1d), :]

    plt.plot(np.linspace(2000, 2023 + 22 / 23, 552), np.nanmean(ar1_nonfireArea, axis=0))


    ar1_fire1 = tac[fire_freq_1d==1]

    min_indices = np.nanargmin(ar1_fire1, axis=0)


    plt.figure();
    plt.plot(np.linspace(2000, 2023 + 22 / 23, 552), ar1_fire1[3062,:])

    fire_annual1 = np.nansum(fire_annual[fire_freq_1d == 2],axis=0)

    ar1_fire1 = tac[fire_freq_1d == 2]
    plt.figure();
    plt.plot(np.linspace(2000, 2023 + 22 / 23, 552), np.nanmean(ar1_fire1, axis=0))

    ar1_fire1 = tac[fire_freq_1d >= 4]
    plt.figure();
    plt.plot(np.linspace(2000, 2023 + 22 / 23, 552), np.nanmean(ar1_fire1, axis=0))

    plt.figure()
    plt.bar([0,1,2,3,4],[np.nanmean(tac) ,np.nanmean(tac[fire_freq_1d == 1]),
    np.nanmean(tac[fire_freq_1d == 2]),
    np.nanmean(tac[fire_freq_1d == 3]),
    np.nanmean(tac[fire_freq_1d > 3])])



    # output_file = current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy'
    # np.save(output_file, ar1_res_5sg)
