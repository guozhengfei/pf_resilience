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

    # burned area
    fire_freq = tf.imread(current_dir + '/1_Input/fire_frequency_pf.tif').astype(float)[::-1, :]
    fire_freq_rsz = np.zeros_like(pf_mask2)
    fire_freq_rsz[:-1, :-1] = fire_freq
    fire_freq_rsz[fire_freq_rsz == 0] = np.nan
    fire_freq = fire_freq_rsz[::3, ::3]
    fire_freq = fire_freq[~np.isnan(pf_mask)]
    fire_freq = fire_freq[~mask]

    coefs = coefs[np.isnan(fire_freq),:]

    # coefs2 = np.zeros((mask.shape[0],coefs.shape[1]))
    # coefs2[~mask,:] = coefs
    # coefs2[mask, :] = np.nan

    pred = coefs[::10,-20:]
    obs = coefs[::10,-40:-20]

    st.linregress(obs.reshape(-1),pred.reshape(-1))

    fig, axs = plt.subplots(1,2, figsize=(5, 2.5))
    axs[0].scatter(obs.reshape(-1),pred.reshape(-1),1,c='grey')
    sns.kdeplot(x=obs[::20,:].reshape(-1), y=pred[::20,:].reshape(-1), cmap="RdYlBu_r", fill=True, thresh=0.01, alpha=0.8,ax = axs[0])
    axs[0].set_xlim([0.05,0.85])
    axs[0].set_ylim([0.05, 0.85])

    # relative importance
    shap_values = coefs[:,1:11]
    shap_mean = np.nanmean(shap_values,axis=0)#[[0,1,2,3,4,5,6,7,8,9,10]]
    shap_std = np.nanstd(shap_values,axis=0)*0.1#[[0,1,2,3,4,5,6,7,8,9,10]]*0.1
    type = [1,1,1,1,2,2,2,3,3,3]
    colors = ['#878787','#878787','#878787','#878787','#d6604d','#d6604d','#d6604d','#4393c3','#4393c3','#4393c3']
    df = pd.DataFrame(np.vstack((shap_mean,shap_std,type)).T).astype(float)
    df.columns=['shap_mean','std','type']
    df.iloc[7,[0,1]] = df.iloc[7,[0,1]]*0.4
    df['color'] = colors
    df['name'] = ['VPD','Srad','Pr','Ta','Ts','SM','ALT','kNDVI','GSL','LAI']
    sorted_df = df.sort_values(by='shap_mean')
    axs[1].barh(np.linspace(0,9,10), sorted_df['shap_mean'].values,xerr = sorted_df['std'],color=sorted_df['color'])
    axs[1].set_yticks(np.linspace(0,9,10),sorted_df['name'])

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig02a_model_drivers'
    plt.savefig(figToPath, dpi=900)

    # Export data to CSV
    export_df = sorted_df[['name', 'shap_mean', 'std', 'color']]
    export_df.to_csv(current_dir + '/2_Output/fig_2a_driver_importance.csv', index=False)

