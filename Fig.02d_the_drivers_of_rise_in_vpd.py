import numpy as np

from scipy.stats import sem, t
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
_, vpd1_ci95, vpd1_n = mean_ci95(vpd_annual_data, axis=0)
vpd2_anom = 2*(np.nanmean(vpd_cur,axis=0)-np.nanmean(vpd_cur))
_, vpd2_ci95, vpd2_n = mean_ci95(2 * vpd_cur, axis=0)

# CIs are conditional on the displayed fixed anomaly baseline/offset.
svap_anom = np.nanmean(svap_yr, axis=1) - np.nanmean(svap_yr)
_, svap_ci95, svap_n = mean_ci95(svap_yr, axis=1)
vap_anom = np.nanmean(vap_yr, axis=1) - np.nanmean(vap_yr) - 0.05
_, vap_ci95, vap_n = mean_ci95(vap_yr, axis=1)
sst_anom = np.nanmean(sst_yr_mean, axis=1) - np.nanmean(sst_yr_mean)
_, sst_ci95, sst_n = mean_ci95(sst_yr_mean, axis=1)
OE_anom = np.nanmean(oceanE_arr_yr, axis=1) - np.nanmean(oceanE_arr_yr)
_, OE_ci95, OE_n = mean_ci95(oceanE_arr_yr, axis=1)

# plot figure
fig, axs = plt.subplots(3,1, figsize=(5, 7.5))
axs[0].plot(vpd1_anom,c='#d6604d')
axs[0].fill_between(range(33),vpd1_anom+vpd1_ci95,vpd1_anom-vpd1_ci95,alpha = 0.2,color='#d6604d')
axs[0].plot(vpd2_anom,c='#4393c3')
axs[0].fill_between(range(33),vpd2_anom+vpd2_ci95,vpd2_anom-vpd2_ci95,alpha = 0.2,color='#4393c3')
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
axs[1].fill_between(range(33),svap_anom+svap_ci95,svap_anom-svap_ci95,alpha = 0.2,color='#d6604d')
axs[1].plot(vap_anom,c='#4393c3')
axs[1].fill_between(range(33),vap_anom+vap_ci95,vap_anom-vap_ci95,alpha = 0.2,color='#4393c3')
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
axs[2].fill_between(range(33),sst_anom+sst_ci95,sst_anom-sst_ci95,alpha = 0.2,color='#d6604d')
y = sst_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[0.1])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
axs[2].plot(xHat,yHat,'--',c='#d6604d')
axs[2].set_xticks(np.linspace(0,30,7),np.linspace(1990,1990+30,7).astype(int).astype(str))

ax2 = axs[2].twinx()
ax2.plot(OE_anom,c='#4393c3')
ax2.fill_between(range(33),OE_anom+OE_ci95,OE_anom-OE_ci95,alpha = 0.2,color='#4393c3')
y = OE_anom
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
res = my_pwlf_0.fit(2, [18],[1])
xHat = np.linspace(min(x), max(x), num=100)
yHat = my_pwlf_0.predict(xHat)
ax2.plot(xHat,yHat,'--',c='#4393c3')

fig.tight_layout()
figToPath = current_dir + '/4_Figures/Fig02c_resilience_drivers'
plt.savefig(figToPath, dpi=600)
    
# Create time series array
years = np.linspace(1990, 2022, 33)

# Export data for panel 1 (VPD)
vpd_data = pd.DataFrame({
    'Year': years,
    'VPD1_Anomaly': vpd1_anom,
    'VPD1_CI95_HalfWidth': vpd1_ci95,
    'VPD1_N': vpd1_n,
    'VPD2_Anomaly': vpd2_anom,
    'VPD2_CI95_HalfWidth': vpd2_ci95,
    'VPD2_N': vpd2_n
})
vpd_data.to_csv(current_dir + '/4_Figures/Fig02d_vpd_data.csv', index=False)

# Export data for panel 2 (SVAP and VAP)
vap_data = pd.DataFrame({
    'Year': years,
    'SVAP_Anomaly': svap_anom,
    'SVAP_CI95_HalfWidth': svap_ci95,
    'SVAP_N': svap_n,
    'VAP_Anomaly': vap_anom,
    'VAP_CI95_HalfWidth': vap_ci95,
    'VAP_N': vap_n
})
vap_data.to_csv(current_dir + '/4_Figures/Fig02c_vap_data.csv', index=False)

# Export data for panel 3 (SST and Ocean Evaporation)
sst_oe_data = pd.DataFrame({
    'Year': years,
    'SST_Anomaly': sst_anom,
    'SST_CI95_HalfWidth': sst_ci95,
    'SST_N': sst_n,
    'OceanE_Anomaly': OE_anom,
    'OceanE_CI95_HalfWidth': OE_ci95,
    'OceanE_N': OE_n
})
sst_oe_data.to_csv(current_dir + '/4_Figures/Fig02c_sst_oceane_data.csv', index=False)