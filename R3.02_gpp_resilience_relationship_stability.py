import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)

import scipy.signal as ss
import multiprocess as mp
import os
import rasterio
import cv2
import pandas as pd
from scipy.interpolate import interp1d

def linear_Fit(array):
    from sklearn.linear_model import LinearRegression
    X = array[:,:-1]
    y = array[:,-1]
    regressor = LinearRegression()
    regressor.fit(X, y)
    coefs = regressor.coef_
    intcpt = regressor.intercept_
    R2 = regressor.score(X, y)
    return list(coefs)+[intcpt]+[R2]

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array
def moving_sd(array,bands_yr):
    yr_num = int(array.shape[1]/bands_yr)
    std_yr = []
    for i in range(yr_num):
        std_i = np.nanstd(array[:,i*bands_yr:(i+1)*bands_yr],axis=1)#/np.nanmean(array[:,i*bands_yr:(i+3)*bands_yr],axis=1)
        std_yr.append(std_i)
    return np.array(std_yr)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

    # read the resilience data
    tac_name = current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy'
    TAC = np.load(tac_name)[:, 23*2:23*2+460] # 2002-0218 ~ 2021-0218
    rsp_arr = TAC.reshape(TAC.shape[0], 20, 23)#[:,:,6:13]
    TAC_yr = np.nanmean(rsp_arr, axis=2)
    TAC_yr[TAC_yr==0] = np.nan

    # gpp fluxcomX deseasonalied to calculate stability
    name = current_dir+'/1_Input/data for drivers/gpp_fluxcomX_monthly_rm_seasonality.npy'
    gpp_flux = np.load(name)#*365  # # 2001-01 to 2021-12
    gpp_sd = smooth_2d_array(moving_sd(gpp_flux,12).T,5)
    # plt.figure();plt.plot(np.nanmean(gpp_sd,axis=0))

    # gpp gosif deseasonalied to calculate stability
    name = current_dir+'/1_Input/data for drivers/gpp_gosif_monthly_rm_seasonality.npy'
    gpp_sif = np.load(name)#*365  # # 2000-03 to 2023-02
    gpp_sd_sif = smooth_2d_array(moving_sd(gpp_sif,12).T, 5)/1000
    # plt.figure(); plt.plot(np.nanmedian(gpp_sd_sif, axis=0))

    # Ta
    Ta = np.load(current_dir+'/1_Input/data for drivers/tmmx_16d_rm_seasonality.npy')[:,460:460+460]/10
    Ta_sd = smooth_2d_array(moving_sd(Ta,23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(Ta_sd, axis=0))

    # Srad
    Srad = np.load(current_dir + '/1_Input/data for drivers/srad_16d_rm_seasonality.npy')[:,460:460+460]/10
    Srad_sd = smooth_2d_array(moving_sd(Srad,23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(Srad_sd, axis=0))

    # Pr
    Pr = np.load(current_dir + '/1_Input/data for drivers/pr_16d_rm_seasonality.npy')[:,460:460+460]
    Pr_sd = smooth_2d_array(moving_sd(Pr,23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(Pr_sd, axis=0))

    # VPD
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_16d_rm_seasonality.npy')[:, 460:460 + 460]
    vpd_sd = smooth_2d_array(moving_sd(vpd, 23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(vpd_sd, axis=0))

    # SM
    sm = np.load(current_dir + '/1_Input/data for drivers/sm_16d_rm_seasonality.npy')[:, 460:460 + 460]*0.1
    sm_sd = smooth_2d_array(moving_sd(sm, 23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(sm_sd, axis=0))

    fig, axs = plt.subplots(1, 2, figsize=(10 * 0.8, 3.5 * 0.75),width_ratios=[1,1.])
    axs[0].plot(np.linspace(2002, 2021, 20), np.nanmean(gpp_sd_sif / 3, axis=0)[2:-1], '#dba958', ls='--', lw=1)  # sif
    axs[0].plot(np.linspace(2002, 2021, 20), np.nanmean(gpp_sd, axis=0)[1:], '#cd7055', ls='--', lw=1)  # fluxcom
    gpp_ensmb_sd = (np.nanmean(gpp_sd_sif / 3, axis=0)[2:-1] + np.nanmean(gpp_sd, axis=0)[1:]) / 2
    axs[0].plot(np.linspace(2002, 2021, 20), gpp_ensmb_sd, '#b9241a', lw=2)  # ensemble
    ax2 = axs[0].twinx()
    ax2.plot(np.linspace(2002, 2021, 20), np.nanmean(TAC_yr, axis=0), '#2c467b', lw=2)
    import scipy.stats as st

    st.linregress(gpp_ensmb_sd, np.nanmean(TAC_yr, axis=0))

    axs[0].set_ylabel("SD of GPP (gC/m²/day)")
    axs[0].set_ylim([60 / 365, 90 / 365])
    ax2.set_ylabel("TAC")
    ax2.set_ylim([0.42, 0.48])
    axs[0].yaxis.label.set_color('#b9241a')
    ax2.yaxis.label.set_color('#2c467b')
    axs[0].tick_params(axis='y', colors='#b9241a')
    ax2.tick_params(axis='y', colors='#2c467b')
    ax2.spines['left'].set_color('#b9241a')
    ax2.spines['right'].set_color('#2c467b')
    axs[0].set_xlabel("Year")


    axs[1].plot(np.linspace(2002, 2021, 20), np.nanmean(Ta_sd, axis=0), '#66c2a5', lw=1.5)
    st.linregress(np.nanmean(Ta_sd, axis=0), gpp_ensmb_sd)

    axs[1].set_ylim([1.0, 3])
    axs[1].set_ylabel("SD of Ta (°C)")
    axs[1].yaxis.label.set_color('#66c2a5')
    axs[1].tick_params(axis='y', colors='#66c2a5')
    
    ax1 = axs[1].twinx()
    ax1.plot(np.linspace(2002, 2021, 20), np.nanmean(Srad_sd, axis=0), '#fc8d62', lw=1.5)
    st.linregress(np.nanmean(Srad_sd, axis=0), gpp_ensmb_sd)
    ax1.set_ylim([5, 15])
    ax1.set_ylabel("SD of Srad (W/m²)")
    ax1.yaxis.label.set_color('#fc8d62')
    ax1.tick_params(axis='y', colors='#fc8d62')
    ax1.spines['right'].set_color('#fc8d62')
    
    ax2 = axs[1].twinx()
    ax2.spines['right'].set_position(('outward', 40))
    ax2.plot(np.linspace(2002, 2021, 20), np.nanmean(vpd_sd, axis=0), '#8da0cb', lw=1.5)
    ax2.set_ylabel("SD of vpd (hPa)")
    ax2.set_ylim([3, 7])
    ax2.yaxis.label.set_color('#8da0cb')
    ax2.tick_params(axis='y', colors='#8da0cb')
    ax2.spines['right'].set_color('#8da0cb')

    ax3 = axs[1].twinx()
    ax3.spines['right'].set_position(('outward', 80))
    ax3.plot(np.linspace(2002, 2021, 20), np.nanmean(sm_sd, axis=0), '#e78ac3', lw=1.5)
    ax3.set_ylabel("SD of SM (mm)")
    ax3.set_ylim([6.5, 8.5])
    ax3.yaxis.label.set_color('#e78ac3')
    ax3.tick_params(axis='y', colors='#e78ac3')
    ax3.spines['right'].set_color('#e78ac3')
    ax3.spines['left'].set_color('#66c2a5')
    
    axs[1].set_xlabel("Year")
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R302_vpd_sm'
    plt.savefig(figToPath, dpi=900)

    #
    # # Export data to CSV files
    # years = np.linspace(2002, 2021, 20)
    #
    # # # Panel a data - GPP and TAC
    # # panel_a_data = pd.DataFrame({
    # #     'Year': years,
    # #     'GPP_SD_GOSIF': np.nanmean(gpp_sd_sif/3, axis=0)[2:-1],
    # #     'GPP_SD_FluxCom': np.nanmean(gpp_sd, axis=0)[1:],
    # #     'GPP_SD_Ensemble': gpp_ensmb_sd,
    # #     'TAC': np.nanmean(TAC_yr, axis=0)
    # # })
    # # panel_a_data.to_csv(current_dir + '/4_Figures/Fig04_panel_a_gpp_tac.csv', index=False)
    # #
    # # # Panel b data - Climate variables
    # # panel_b_data = pd.DataFrame({
    # #     'Year': years,
    # #     'Temperature_SD': np.nanmean(Ta_sd, axis=0),
    # #     'Solar_Radiation_SD': np.nanmean(Srad_sd, axis=0),
    # #     'Precipitation_SD': np.nanmean(Pr_sd, axis=0)
    # # })
    # # panel_b_data.to_csv(current_dir + '/4_Figures/Fig04_panel_b_climate.csv', index=False)





