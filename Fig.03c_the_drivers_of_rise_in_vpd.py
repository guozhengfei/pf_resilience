import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')
import tifffile as tf
from plot_NH import *
import os
from PIL import Image
import scipy.stats as st
import seaborn as sns
import pwlf


def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2
    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))
    return smoothed_arr

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:,-33:] # vpd from TerraClimate
oceanE_arr_yr = np.load(current_dir+'/1_Input/data for drivers/oceanE_yr_1990-2022.npy')
sst_yr_mean = np.load(current_dir+'/1_Input/data for drivers/SST_yr_1990-2022.npy')
svap_yr = smooth_2d_array(np.load(current_dir+'/1_Input/data for drivers/svap_yr_1982-2022.npy')[-33:,:],3)
vap_yr = smooth_2d_array(np.load(current_dir+'/1_Input/data for drivers/vap_yr_1982-2022.npy')[-33:,:],3)
Tmp_yr = np.load(current_dir+'/1_Input/data for drivers/Tmp_yr_1982-2022.npy')

# VPD data
vpd_annual_data = smooth_2d_array(vpd, 3)/10
vpd_cur = np.load(current_dir+'/1_Input/data for drivers/vpd_yr_1982-2022.npy')[-33:].T
vpd_cur = smooth_2d_array(vpd_cur,3)
vpd1_anom = np.nanmean(vpd_annual_data,axis=0)-np.nanmean(vpd_annual_data)
vpd1_sd = np.nanstd(vpd_annual_data,axis=0)/20
vpd2_anom = 2*(np.nanmean(vpd_cur,axis=0)-np.nanmean(vpd_cur))
vpd2_sd = np.nanmean(vpd_cur,axis=0)/30

# SVAP vs. Vap data
svap_anom = np.nanmean(svap_yr,axis=1)-np.nanmean(svap_yr)
svap_sd = np.nanstd(svap_yr,axis=1)/20
vap_anom = np.nanmean(vap_yr,axis=1)-np.nanmean(vap_yr)-0.05
vap_sd = np.nanmean(vap_yr,axis=1)/20

# SVAP vs. Vap data
svap_anom = np.nanmean(svap_yr,axis=1)-np.nanmean(svap_yr)
svap_sd = np.nanstd(svap_yr,axis=1)/40
vap_anom = np.nanmean(vap_yr,axis=1)-np.nanmean(vap_yr)-0.05
vap_sd = np.nanmean(vap_yr,axis=1)/40

# SST and OceanE
sst_anom = np.nanmean(sst_yr_mean,axis=1)-np.nanmean(sst_yr_mean)
sst_sd = np.nanstd(sst_yr_mean,axis=1)/30
OE_anom = np.nanmean(oceanE_arr_yr,axis=1)-np.nanmean(oceanE_arr_yr)
OE_sd = np.nanmean(oceanE_arr_yr,axis=1)/50

# plot figure
fig, axs = plt.subplots(3,1, figsize=(5, 7.5))
axs[0].plot(vpd1_anom,c='#d6604d')
axs[0].fill_between(range(33),vpd1_anom+vpd1_sd,vpd1_anom-vpd1_sd,alpha = 0.2,color='#d6604d')
axs[0].plot(vpd2_anom,c='#4393c3')
axs[0].fill_between(range(33),vpd2_anom+vpd2_sd,vpd2_anom-vpd2_sd,alpha = 0.2,color='#4393c3')
x = range(33)
y = vpd1_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[-0.08])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
axs[0].plot(xHat,yHat,'--',c='#d6604d')
y = vpd2_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[-0.08])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
axs[0].plot(xHat,yHat,'--',c='#4393c3')
axs[0].set_xticks(np.linspace(0,30,7),np.linspace(1990,1990+30,7).astype(int).astype(str))

axs[1].plot(svap_anom,c='#d6604d')
axs[1].fill_between(range(33),svap_anom+svap_sd,svap_anom-svap_sd,alpha = 0.2,color='#d6604d')
axs[1].plot(vap_anom,c='#4393c3')
axs[1].fill_between(range(33),vap_anom+vap_sd,vap_anom-vap_sd,alpha = 0.2,color='#4393c3')
y = svap_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[0.01])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
axs[1].plot(xHat,yHat,'--',c='#d6604d')
y = vap_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[0.01])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
axs[1].plot(xHat,yHat,'--',c='#4393c3')
axs[1].set_xticks(np.linspace(0,30,7),np.linspace(1990,1990+30,7).astype(int).astype(str))

axs[2].plot(sst_anom,c='#d6604d')
axs[2].fill_between(range(33),sst_anom+sst_sd,sst_anom-sst_sd,alpha = 0.2,color='#d6604d')
y = sst_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[0.1])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
axs[2].plot(xHat,yHat,'--',c='#d6604d')
axs[2].set_xticks(np.linspace(0,30,7),np.linspace(1990,1990+30,7).astype(int).astype(str))

ax2 = axs[2].twinx()
ax2.plot(OE_anom,c='#4393c3')
ax2.fill_between(range(33),OE_anom+OE_sd,OE_anom-OE_sd,alpha = 0.2,color='#4393c3')
y = OE_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[1])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
ax2.plot(xHat,yHat,'--',c='#4393c3')

fig.tight_layout()
figToPath = current_dir + '/4_Figures/Fig03c_resilience_drivers'
plt.savefig(figToPath, dpi=600)