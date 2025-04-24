import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
import tifffile as tf
import os

# transfer the ar1 from 1d to 2d map

ar1_4yr = np.load(r'D:\Projects\Project_pf\Data\ar1_4yr_16d_kndvi_gimms.npy')
pf_mask0 = tf.imread(r'D:\Projects\Project_pf\Data\Land_Cover_PF_3g.tif').astype(float)
pf_mask0[pf_mask0 <= 0] = np.nan
pf_mask0[pf_mask0 > 12] = np.nan
EVI_folder = r'D:\Projects\Project_pf\Data\GIMMS_NDVI\\'
evi_filenames = os.listdir(EVI_folder)
evi1 = np.load(EVI_folder + evi_filenames[14])
evi2 = np.load(EVI_folder + evi_filenames[15])
merged_evi = np.where(np.isnan(evi1), evi2, evi1)  # Where condition True, yield x, otherwise yield y.
pf_mask0[merged_evi <= 0] = np.nan
pf_mask = pf_mask0[::3, ::3]
# all pixel mean
ar1_all = pf_mask * 1
ar1_all[~np.isnan(ar1_all)] = np.nanmean(ar1_4yr, axis=1)

# tmmxocessing data
Data_folder = r'D:\Projects\Project_pf\Data\data for drivers\\'
tmmx01 = tf.imread(Data_folder + 'tmmx_1982_2022-0000000000-0000000000.tif')
tmmx02 = tf.imread(Data_folder + 'tmmx_1982_2022-0000000000-0000002304.tif')
tmmx = np.concatenate((tmmx01, tmmx02), axis=1)[::3, ::3,]
tmmx = tmmx[:,:-1,:]
tmmx = tmmx[::-1,:,:].astype(float)
tmmx[tmmx==0]=np.nan

# annual mean
bands_year = 12
yr_num = 41
tmmx_annual_mean = []
tmmx_annual_cv = []
for i in range(yr_num):
    tmmx_mean_i = np.nanmean(tmmx[:,:,i*12:(i+1)*12],axis=2)
    tmmx_std_i = np.nanstd(tmmx[:,:,i*12:(i+1)*12],axis=2)
    tmmx_cv_i = tmmx_std_i/tmmx_mean_i
    tmmx_cv_i[(tmmx_cv_i > 1e308) | (tmmx_cv_i < -1e308)] = np.nan
    tmmx_annual_mean.append(tmmx_mean_i)
    tmmx_annual_cv.append(tmmx_cv_i)
tmmx_annual_mean = np.stack(tmmx_annual_mean,axis=2)
# tmmx_annual_mean[np.isnan(tmmx_annual_mean)] = 0
tmmx_annual_cv = np.stack(tmmx_annual_cv,axis=2)

tmmx_82_07_mean = np.nanmean(tmmx_annual_mean[:,:,:26],axis=2)
np.save(r'D:\Projects\Project_pf\Data\data for drivers\tmmx_82_07_mean.npy',tmmx_82_07_mean)
tmmx_08_22_mean = np.nanmean(tmmx_annual_mean[:,:,26:],axis=2)
np.save(r'D:\Projects\Project_pf\Data\data for drivers\tmmx_08_22_mean.npy',tmmx_08_22_mean)

tmmx_82_07_cv = np.nanmean(tmmx_annual_cv[:,:,:26],axis=2)
np.save(r'D:\Projects\Project_pf\Data\data for drivers\tmmx_82_07_cv.npy',tmmx_82_07_cv)
tmmx_08_22_cv = np.nanmean(tmmx_annual_cv[:,:,26:],axis=2)
np.save(r'D:\Projects\Project_pf\Data\data for drivers\tmmx_08_22_cv.npy',tmmx_08_22_cv)

# trend
tmmx_82_07 = tmmx_annual_mean[:,:,:26]
tmmx_82_07_rsp = np.reshape(tmmx_82_07,(tmmx_82_07.shape[0]*tmmx_82_07.shape[1],tmmx_82_07.shape[2]))
tmmx_82_07_trend = np.polyfit(np.arange(tmmx_82_07.shape[2]), tmmx_82_07_rsp.T, deg=1)[0,:]
tmmx_82_07_trend = np.reshape(tmmx_82_07_trend,(tmmx_82_07.shape[0],tmmx_82_07.shape[1]))
np.save(r'D:\Projects\Project_pf\Data\data for drivers\tmmx_82_07_trend.npy',tmmx_82_07_trend)

tmmx_08_22 = tmmx_annual_mean[:,:,26:]
tmmx_08_22_rsp = np.reshape(tmmx_08_22,(tmmx_08_22.shape[0]*tmmx_08_22.shape[1],tmmx_08_22.shape[2]))
tmmx_08_22_trend = np.polyfit(np.arange(tmmx_08_22.shape[2]), tmmx_08_22_rsp.T, deg=1)[0,:]
tmmx_08_22_trend = np.reshape(tmmx_08_22_trend,(tmmx_08_22.shape[0],tmmx_08_22.shape[1]))
np.save(r'D:\Projects\Project_pf\Data\data for drivers\tmmx_08_22_trend.npy',tmmx_08_22_trend)
# plt.figure()
# plt.imshow(tmmx_08_22_trend)
# plt.figure()
# plt.imshow(tmmx_82_07_trend)
fig,axs = plt.subplots(2,figsize=(6,6))
axs[0].plot(np.nanmean(np.nanmean(tmmx_annual_mean,axis=0),axis=0))
axs[1].plot(np.nanmean(np.nanmean(tmmx_annual_cv,axis=0),axis=0))
axs[0].set_title('tmmx')