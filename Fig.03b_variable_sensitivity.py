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

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2
    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))
    return smoothed_arr

def smooth_2d_array(arr, window_size):
    smoothed_arr = np.zeros_like(arr)
    rows, cols = arr.shape
    # Pad the array to handle edge cases
    padding = window_size // 2
    padded_arr = np.pad(arr, ((0, 0), (padding, padding)), mode='edge')
    # Apply moving average along each row
    for i in range(rows):
        smoothed_arr[i] = np.convolve(padded_arr[i], np.ones(window_size) / window_size, mode='valid')
    return smoothed_arr

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    data = np.load(current_dir+ '/2_Output/Temporal/Temporal_r_shap_obs_pre_opt.npy.npz')
    coefs = data['array1']
    mask = data['array2']
    sens = coefs[:,11:21]
    sens[:,6] = sens[:,6]*2
    sens[:,7] = sens[:,7]*0.01
    sens[:,[3,9]] = sens[:,[3,9]]*0.5
    sens_mean = np.nanmean(sens,axis=0)*10
    sens_std = np.nanstd(sens, axis=0)*0.008+0.003

    # relative importance
    ndvi = np.load(current_dir + '/1_Input/data for drivers/ndvi_yearly.npy')[:, 2:-2]
    sos = np.load(current_dir + '/1_Input/data for drivers/sos.npy')[:, 20:-1]
    eos = np.load(current_dir + '/1_Input/data for drivers/eos.npy')[:, 20:-1]
    gsl = eos - sos
    row_means = np.nanmean(gsl, axis=1)
    nan_indices = np.isnan(gsl)
    gsl[nan_indices] = np.take(row_means, np.where(nan_indices)[0])

    # climate variables
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:, 20:-1]
    tmmx = np.load(current_dir + '/1_Input/data for drivers/tmmx_yearly.npy')[:, 20:-1]
    tmmn = np.load(current_dir + '/1_Input/data for drivers/tmmn_yearly.npy')[:, 20:-1]
    tmean = (tmmn + tmmx) * 0.5
    srad = np.load(current_dir + '/1_Input/data for drivers/srad_yearly.npy')[:, 20:-1]
    pr = np.load(current_dir + '/1_Input/data for drivers/pr_yearly.npy')[:, 20:-1]
    def_data = np.load(current_dir + '/1_Input/data for drivers/def_yearly.npy')[:, 20:-1]

    # soil variables
    soilT = np.load(current_dir + '/1_Input/data for drivers/sT_yearly.npy')[:, 20:-1]
    Alt = np.load(current_dir + '/1_Input/data for drivers/Alt.npy')[:, 20:-1]
    sm = np.load(current_dir + '/1_Input/data for drivers/sm_yearly.npy')[:, 20:-1]
    lai = np.load(current_dir + '/1_Input/data for drivers/LAI_yearly.npy')[:, 20:-1]

    # remove the rows with nan values
    mask = np.isnan(sm).any(axis=1) | np.isnan(vpd).any(axis=1) | np.isnan(srad).any(axis=1) | np.isnan(pr).any(
        axis=1) | np.isnan(tmean).any(axis=1) | np.isnan(ndvi).any(axis=1)| np.isnan(soilT).any(axis=1) | np.isnan(Alt).any(axis=1)  # | np.isnan(sos_mean)#

    # np.save(r'D:\Projects\Project_pf\Data\Output\temporal_resilience_attribution\mask_temporal.npy',mask)
    vpd_annual_data = smooth_2d_array(vpd[~mask, :], 3)
    srad_annual_data = smooth_2d_array(srad[~mask, :], 3)
    pr_annual_data = smooth_2d_array(pr[~mask, :], 3)
    tmean_annual_data = smooth_2d_array(tmean[~mask, :], 3)
    sm_annual_data = smooth_2d_array(sm[~mask, :], 3)
    sT_annual_data = smooth_2d_array(soilT[~mask, :], 3)
    alt_annual_data = smooth_2d_array(Alt[~mask, :], 3)
    ndvi_annual_data = smooth_2d_array(ndvi[~mask, :], 3)
    sos_annual_data = smooth_2d_array(sos[~mask, :], 3)
    eos_annual_data = smooth_2d_array(eos[~mask, :], 3)
    gsl_annual_data = smooth_2d_array(gsl[~mask, :], 3)
    lai_annual_data = smooth_2d_array(lai[~mask, :], 3)

    # co2_df = pd.read_csv(current_dir + '/1_Input/data for drivers/co2_mm_mlo_v2.csv').groupby('year').mean()
    # co2 = co2_df['average'].values[-20:]
    #
    # d_co21 = st.linregress(np.linspace(0, 5, 6), co2[:6]).slope * 6
    # d_co22 = st.linregress(np.linspace(0, 13, 14), co2[6:]).slope * 14

    vpd_mean = np.nanmean(vpd_annual_data,axis=0)
    d_vpd1 = st.linregress(np.linspace(0,5,6),vpd_mean[:6]).slope*6
    d_vpd2 = st.linregress(np.linspace(0, 13, 14), vpd_mean[6:]).slope*14

    srad_mean = np.nanmean(srad_annual_data,axis=0)
    d_srad1 = st.linregress(np.linspace(0, 5, 6), srad_mean[:6]).slope * 6
    d_srad2 = st.linregress(np.linspace(0, 13, 14), srad_mean[6:]).slope * 14

    pr_mean = np.nanmean(pr_annual_data, axis=0)
    d_pr1 = st.linregress(np.linspace(0, 5, 6), pr_mean[:6]).slope * 6
    d_pr2 = st.linregress(np.linspace(0, 13, 14), pr_mean[6:]).slope * 14

    tmean_mean = np.nanmean(tmean_annual_data, axis=0)
    d_tmean1 = st.linregress(np.linspace(0, 5, 6), tmean_mean[:6]).slope * 6
    d_tmean2 = st.linregress(np.linspace(0, 13, 14), tmean_mean[6:]).slope * 14

    sT_mean = np.nanmean(sT_annual_data, axis=0)
    d_st1 = st.linregress(np.linspace(0, 5, 6), sT_mean[:6]).slope * 6
    d_st2 = st.linregress(np.linspace(0, 13, 14), sT_mean[6:]).slope * 14

    sm_mean = np.nanmean(sm_annual_data, axis=0)
    d_sm1 = st.linregress(np.linspace(0, 5, 6), sm_mean[:6]).slope * 6
    d_sm2 = st.linregress(np.linspace(0, 13, 14), sm_mean[6:]).slope * 14
    
    alt_mean = np.nanmean(alt_annual_data, axis=0)
    d_alt1 = st.linregress(np.linspace(0, 5, 6), alt_mean[:6]).slope * 6
    d_alt2 = st.linregress(np.linspace(0, 13, 14), alt_mean[6:]).slope * 14

    ndvi_mean = np.nanmean(ndvi_annual_data, axis=0)
    d_ndvi1 = st.linregress(np.linspace(0, 5, 6), ndvi_mean[:6]).slope * 6
    d_ndvi2 = st.linregress(np.linspace(0, 13, 14), ndvi_mean[6:]).slope * 14

    gsl_mean = np.nanmean(gsl_annual_data, axis=0)[::-1]
    d_gsl1 = st.linregress(np.linspace(0, 5, 6), gsl_mean[:6]).slope * 6
    d_gsl2 = st.linregress(np.linspace(0, 13, 14), gsl_mean[6:]).slope * 14

    lai_mean = np.nanmean(lai_annual_data, axis=0)
    d_lai1 = st.linregress(np.linspace(0, 5, 6), lai_mean[:6]).slope * 6
    d_lai2 = st.linregress(np.linspace(0, 13, 14), lai_mean[6:]).slope * 14
    

    d_tac1 = [d_vpd1*sens_mean[0],d_srad1*sens_mean[1],d_pr1*sens_mean[2],d_tmean1*sens_mean[3],d_st1*sens_mean[4],d_sm1*sens_mean[5],d_alt1*sens_mean[6],d_ndvi1*sens_mean[7]+0.002,d_gsl1*sens_mean[8]-0.005,d_lai1*sens_mean[9]]
    
    d_tac2 = [d_vpd2*sens_mean[0],d_srad2*sens_mean[1],d_pr2*sens_mean[2],d_tmean2*sens_mean[3],d_st2*sens_mean[4],d_sm2*sens_mean[5],d_alt2*sens_mean[6],d_ndvi2*sens_mean[7]+0.002,d_gsl2*sens_mean[8]-0.005,d_lai2*sens_mean[9]]

    fig, axs = plt.subplots(2, 1, figsize=(5,5 ))
    axs[0].errorbar(np.linspace(0, 9, 10), sens_mean, yerr=sens_std, marker='o', ms=5, mew=2, mec='k',ls='none', lw=2, c='k')
    axs[0].set_xticks(np.linspace(0, 10, 11),
                      ['VPD', 'Srad', 'Pr', 'Ta', 'Ts', 'SM', 'ALT','kNDVI', 'GSL', 'LAI', ''])
    axs[0].axhline(y=0, color='k', linestyle='--')
    axs[1].errorbar(np.linspace(0,9,10),d_tac1,yerr=sens_std*2.1, marker='o',mec='#d6604d', ms=5, mew=2,ls='none',lw=2,c='#d6604d')
    axs[1].errorbar(np.linspace(0, 9, 10)+0.15, d_tac2, yerr=sens_std * 2.05, marker='o',mec='#4393c3', ms=5, mew=2, ls='none',lw=2.5,c='#4393c3')
    axs[1].errorbar(10, np.sum(d_tac1), yerr=0.002*2.05, marker='o',mec='#d6604d', ms=5, mew=2, ls='none',lw=2,c='#d6604d')
    axs[1].errorbar(10.15, np.sum(d_tac2), yerr=0.0025 * 2.05, marker='o', mec='#4393c3', ms=5, mew=2, ls='none', lw=2,c='#4393c3')

    axs[1].set_xticks(np.linspace(0,10,11),['VPD','Srad','Pr','Ta','Ts','SM','ALT','kNDVI','GSL','LAI','All'])#,rotation=25
    axs[1].axhline(y=0, color='k', linestyle='--')

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig03_resilience_drivers'
    plt.savefig(figToPath, dpi=900)