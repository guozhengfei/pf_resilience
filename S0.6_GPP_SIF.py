import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
import os
import cv2
matplotlib.use('Qt5Agg')
from PIL import Image

def moving_sd(array):
    bands_yr = 12
    yr_num = int(array.shape[1]/bands_yr)
    std_yr = []
    for i in range(yr_num):
        std_i = np.nanstd(array[:,i*bands_yr:(i+1)*bands_yr],axis=1)#/np.nanmean(array[:,i*bands_yr:(i+3)*bands_yr],axis=1)
        std_yr.append(std_i)
    return np.array(std_yr)
def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr

# Open the NetCDF file
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
filenames = np.sort(os.listdir(current_dir+'/1_Input/GPP_SIF/'))

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
pf_mask_1d = pf_mask.flatten()

gpp = []
for name in filenames:
    filename = current_dir+'/1_Input/GPP_SIF/'+name
    data = tf.imread(filename)[::2,::2]
    rows = 1800
    # plt.figure(); plt.imshow(data_pf)
    data_pf = data[int(np.round((90 - 83.7) / 180 * rows)):int(np.round((90 - 27.0) / 180 * rows)), :]
    data_rsz = cv2.resize(data_pf[:, :], (pf_mask.shape[1], pf_mask.shape[0]))
    gpp.append(data_rsz)
    print(name)

gpp_arr = np.array(gpp).astype(float)
gpp_arr = gpp_arr[:,::-1,:]
gpp_arr[:,np.isnan(pf_mask)]=np.nan
plt.figure(); plt.imshow(gpp_arr[0,:,:])

gpp_rsz = gpp_arr.reshape((gpp_arr.shape[0],gpp_arr.shape[1]*gpp_arr.shape[2])).T
gpp_rsz = gpp_rsz[~np.isnan(pf_mask_1d), :]

yr_num = 23
bands_yr = 12

gpp_yr = np.zeros_like(gpp_rsz)
for year in range(yr_num):
    st = year * bands_yr
    ed = st + bands_yr
    evi_year = np.nanmean(gpp_rsz[:, st:ed], axis=1)
    evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_yr, axis=1)
    gpp_yr[:, st:ed] = evi_year_rep
rm_offline = gpp_rsz - gpp_yr
# plt.figure(); plt.plot(gpp[:,52,595]); plt.imshow(gpp[1,:,:])

gpp_sea = np.nanmean(np.reshape(rm_offline, (gpp_rsz.shape[0], int(rm_offline.shape[1] / bands_yr), bands_yr)), axis=1)
# plt.figure(); plt.plot(np.nanmean(gpp_sea,axis=0))

Evi_sea_rep = np.zeros_like(rm_offline)
for yr in range(yr_num):
    start_index = (yr - 3) * bands_yr
    end_index = (yr + 3) * bands_yr
    if yr < 3: start_index = 0; end_index = 6 * bands_yr
    if yr > (yr_num - 5): start_index = (yr_num - 6) * bands_yr; end_index = yr_num * bands_yr
    data_i = rm_offline[:, start_index:end_index]
    Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_yr), bands_yr)), axis=1)
    Evi_sea_rep[:, yr * bands_yr:(yr + 1) * bands_yr] = Evi_sea
res = rm_offline - Evi_sea_rep
np.save(current_dir+'/1_Input/data for drivers/gpp_gosif_monthly_rm_seasonality.npy', res)
# gpp_sea_rep = np.zeros_like(rm_offline)
# for yr in range(yr_num):
#     gpp_sea_rep[:, yr * bands_yr:(yr + 1) * bands_yr] = gpp_sea
# res = rm_offline - gpp_sea_rep

plt.figure(); plt.plot(np.linspace(2000,2022,23),smooth_array(np.nanmedian(moving_sd(res), axis=1),5))
