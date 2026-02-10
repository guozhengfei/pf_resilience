import cv2
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import multiprocess as mp
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
import os
import cv2
import xarray as xr

def fill_nan_with_climatology(EVI, bands_year=12):
    num_years = EVI.shape[1] // bands_year
    climatology = np.zeros((EVI.shape[0], EVI.shape[1]))

    for year in range(num_years):
        if year < 2:
            start = 0
            end = 5 * bands_year
        elif year > num_years - 3:
            start = (num_years - 5) * bands_year
            end = num_years * bands_year
        else:
            start = (year - 2) * bands_year
            end = (year + 3) * bands_year

        climatology[:, year * bands_year:(year + 1) * bands_year] = np.nanmean(
            EVI[:, start:end].reshape(EVI.shape[0], 5, bands_year), axis=1
        )

    EVI_filled = np.where(np.isnan(EVI), climatology, EVI)

    return EVI_filled

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/NDVI_pf_16d/'

    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]
    # plt.figure(); plt.imshow(pf_mask)
    # plt.figure(); plt.imshow(evi)
    # pf_mask_1d = pf_mask.flatten()
    evi_filenames = os.listdir(EVI_folder)
    evi_filenames = np.sort(evi_filenames)  # [:529]

    EVI = []
    yr_num = 24
    bands_year = 23
    for name in evi_filenames:
        src = rasterio.open(EVI_folder + name)
        evi = src.read(1)[::3, ::3]
        evi_val = evi[~np.isnan(pf_mask)]
        EVI.append(evi_val)
        print(name)
    EVI2 = np.array(EVI).T
    EVI2[EVI2<0.1]=np.nan
    # plt.figure();plt.hist(EVI2.flatten(),50)
    ratio = np.sum(np.isnan(EVI2),axis=1)/EVI2.shape[1]
    EVI2 = EVI2[ratio<0.65,:]
    EVI2 = fill_nan_with_climatology(EVI2,23)
    EVI2 = fill_nan_with_climatology(EVI2,23)
    EVI2 = fill_nan_with_climatology(EVI2,23)

    evi_yr = EVI2.reshape(EVI2.shape[0], 24, 23)
    evi_yr = np.nanmean(evi_yr, axis=2)

    plt.figure(); plt.plot(np.linspace(2000,2022,23),np.nanmean(evi_yr,axis=0)[:-1],'-o')
