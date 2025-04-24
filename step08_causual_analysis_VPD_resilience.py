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
    data = np.load(current_dir+'/2_Output/data_for_causal_abalysis.npy')

    VPD_tac = data[:,:,[0,-1]]
    divided_arrays = np.split(VPD_tac, VPD_tac.shape[0], axis=0)

    with mp.Pool(6) as pool:
        results = list(pool.map(causal_analysis, divided_arrays))

    results_arr = np.array(results)
    # plt.figure(); plt.hist(results_arr[:,1],50,range=[0,0.1])
    p_values = results_arr[:,1]
    np.sum(p_values<0.05)/p_values.shape[0]
    output_file = current_dir+'/2_Output/causal_vpd_tac.npy'
    np.save(output_file,results_arr)
    #
    # Srad_tac = data[:,:,[1,-1]]
    # divided_arrays = np.split(Srad_tac, Srad_tac.shape[0], axis=0)
    #
    # with mp.Pool(6) as pool:
    #     results = list(pool.map(causal_analysis, divided_arrays))
    # results_arr2 = np.array(results)
    # plt.figure(); plt.hist(results_arr[:,1],50,range=[0,0.1])
    # p_values2 = results_arr2[:,1]
    # np.sum(p_values2<0.05)/p_values2.shape[0]
    # output_file = current_dir+'/2_Output/causal_srad_tac.npy'
    # np.save(output_file,results_arr2)
    # #
    # Pr_tac = data[:,:,[2,-1]]
    # divided_arrays = np.split(Pr_tac, Pr_tac.shape[0], axis=0)
    # with mp.Pool(6) as pool:
    #     results = list(pool.map(causal_analysis, divided_arrays))
    # results_arr3 = np.array(results)
    # plt.figure(); plt.hist(results_arr[:,1],50,range=[0,0.1])
    # p_values3 = results_arr3[:,1]
    # np.sum(p_values3<0.05)/p_values3.shape[0]
    # output_file = current_dir+'/2_Output/causal_pr_tac.npy'
    # np.save(output_file,results_arr3)
    # #
    # Ta_tac = data[:,:,[3,-1]]
    # divided_arrays = np.split(Ta_tac, Ta_tac.shape[0], axis=0)
    # with mp.Pool(6) as pool:
    #     results = list(pool.map(causal_analysis, divided_arrays))
    # results_arr4 = np.array(results)
    # plt.figure(); plt.hist(results_arr[:,1],50,range=[0,0.1])
    # p_values4 = results_arr4[:,1]
    # np.sum(p_values4<0.05)/p_values4.shape[0]
    # output_file = current_dir+'/2_Output/causal_ta_tac.npy'
    # np.save(output_file,results_arr4)
