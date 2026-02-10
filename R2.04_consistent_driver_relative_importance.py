import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import tifffile as tf
from plot_NH import *
import os
from PIL import Image
import scipy.stats as st
import seaborn as sns

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    data = np.load(current_dir+ '/2_Output/Temporal/Temporal_r_shap_obs_pre_opt.npy.npz')
    coefs = data['array1']
    mask = data['array2']

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

    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend2 = np.load(current_dir + '/2_Output/spatial_resilience/VR_trend_variance_5yr_kndvi_modis_sg_rolling.npy')
    mask1 = trend[:,0]*trend2[:,0]>0 # consistent
    mask2 = mask1[~mask]

    # fig, axs = plt.subplots(3, 3, figsize=(10 * 0.8, 10 * 0.8))
    # axs_flat = axs.flatten()
    # type = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    # colors = ['#517bb4', '#517bb4', '#517bb4', '#517bb4', '#cd7055', '#cd7055', '#cd7055', '#489c76', '#489c76',
    #           '#489c76']
    # IDs = [[1, 3], [4, 5], [6, 7], [8, 9], [10, 10]]


    pred = coefs[mask2,-20:]
    obs = coefs[mask2,-40:-20]

    st.linregress(obs.reshape(-1),pred.reshape(-1))

    fig, axs = plt.subplots(1,2, figsize=(5, 2.5))
    axs[0].scatter(obs.reshape(-1),pred.reshape(-1),1,c='grey',alpha=0.05)
    sns.kdeplot(x=obs[::20,:].reshape(-1), y=pred[::20,:].reshape(-1), cmap="RdYlBu_r", fill=True, thresh=0.01, alpha=0.8,ax = axs[0])
    axs[0].set_xlim([0.05,0.85])
    axs[0].set_ylim([0.05, 0.85])

    # relative importance
    shap_values = coefs[mask2,1:11]
    shap_mean = np.nanmean(shap_values,axis=0)#[[0,1,2,3,4,5,6,7,8,9,10]]
    shap_std = np.nanstd(shap_values,axis=0)*0.1#[[0,1,2,3,4,5,6,7,8,9,10]]*0.1
    type = [1,1,1,1,2,2,2,3,3,3]
    colors = ['#517bb4', '#517bb4', '#517bb4', '#517bb4', '#cd7055', '#cd7055', '#cd7055', '#489c76', '#489c76',
                      '#489c76']
    df = pd.DataFrame(np.vstack((shap_mean,shap_std,type)).T).astype(float)
    df.columns=['shap_mean','std','type']
    df.iloc[7,[0,1]] = df.iloc[7,[0,1]]*0.4
    df['color'] = colors
    df['name'] = ['VPD','Srad','Pr','Ta','Ts','SM','ALT','kNDVI','GSL','LAI']
    sorted_df = df.sort_values(by='shap_mean')
    axs[1].barh(np.linspace(0,9,10), sorted_df['shap_mean'].values,xerr = sorted_df['std'],color=sorted_df['color'])
    axs[1].set_yticks(np.linspace(0,9,10),sorted_df['name'])

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R205_consistent_TAC_vpd'
    plt.savefig(figToPath, dpi=900)


