import numpy as np
import pandas as pd
import os
from causal_ccm.causal_ccm import ccm
import multiprocess as mp
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

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

def causal_analysis(array):
    from causal_ccm.causal_ccm import ccm
    X = array[0,:, 0]
    Y = array[0,:,-1]
    tau = 1  # time lag
    E = 2  # shadow manifold embedding dimensions
    L = len(X)  # length of time period to consider
    ccm1 = ccm(X, Y, tau, E, L)
    corr_, p = ccm1.causality()

    # checking convergence
    L_range = range(5, len(X))  # L values to test
    Xhat_My, Yhat_Mx = [], []  # correlation list
    for L in L_range:
        ccm_XY = ccm(X, Y, tau, E, L)  # define new ccm object # Testing for X -> Y
        Xhat_My.append(ccm_XY.causality()[0])
    return [corr_, p]+Xhat_My #corr_, p, Xhat_My

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    # read the resilience data
    tac_name = current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy'
    TAC = np.load(tac_name)[:, 23 * 2:23 * 2 + 460]  # 2002-0218 ~ 2021-0218
    rsp_arr = TAC.reshape(TAC.shape[0], 20, 23)  # [:,:,6:13]
    TAC_yr = np.nanmean(rsp_arr, axis=2)
    TAC_yr[TAC_yr == 0] = np.nan

    # gpp fluxcomX deseasonalied to calculate stability
    name = current_dir + '/1_Input/data for drivers/gpp_fluxcomX_monthly_rm_seasonality.npy'
    gpp_flux = np.load(name)  # # 2001-01 to 2021-12
    gpp_sd = smooth_2d_array(moving_sd(gpp_flux, 12).T, 5)
    # plt.figure();plt.plot(np.nanmean(gpp_sd,axis=0))

    # gpp gosif deseasonalied to calculate stability
    name = current_dir + '/1_Input/data for drivers/gpp_gosif_monthly_rm_seasonality.npy'
    gpp_sif = np.load(name)  # # 2000-03 to 2023-02
    gpp_sd_sif = smooth_2d_array(moving_sd(gpp_sif, 12).T, 5) / 1000

    gpp_ensmb_sd = (gpp_sd_sif[:,2:-1]/3 +gpp_sd[:,1:]) / 2
    # plt.figure(); plt.plot(np.nanmean(gpp_ensmb_sd, axis=0))
    mask = np.isnan(TAC_yr).any(axis=1) | np.isnan(gpp_ensmb_sd).any(axis=1)

    tac_gpp = np.stack((TAC_yr[~mask,:], gpp_ensmb_sd[~mask,:]), axis=2)
    divided_arrays = np.split(tac_gpp, tac_gpp.shape[0], axis=0)

    with mp.Pool(10) as pool:
        results = list(pool.map(causal_analysis, divided_arrays))

    results_arr = np.array(results)
    # plt.figure(); plt.hist(results_arr[:,1],50,range=[0,0.1])
    p_values = results_arr[:,1]
    np.sum(p_values<0.05)/p_values.shape[0]
    output_file = current_dir+'/2_Output/causal_tac_gpp.npz'
    np.savez(output_file, array1=results_arr, array2=mask)

    # np.save(output_file,results_arr)

