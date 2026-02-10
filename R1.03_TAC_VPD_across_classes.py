import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import tifffile as tf
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

    fig, axs = plt.subplots(3,3,figsize=(10*0.8,10*0.8))
    axs_flat = axs.flatten()
    type = [1,1,1,1,2,2,2,3,3,3]
    colors = ['#517bb4','#517bb4','#517bb4','#517bb4','#cd7055','#cd7055','#cd7055','#489c76','#489c76','#489c76']
    IDs = [[1,3],[4,5],[6,7],[8,9],[10, 10]]

    for i in range(len(IDs)):
        ids = IDs[i]
        PFT_1d = pf_mask[~np.isnan(pf_mask)][~mask]
        index = (PFT_1d == ids[0]) | (PFT_1d == ids[1])

        shap_values = coefs[index,1:11]
        shap_mean = np.nanmean(shap_values,axis=0)#[[0,1,2,3,4,5,6,7,8,9,10]]
        shap_std = np.nanstd(shap_values,axis=0)*0.1#[[0,1,2,3,4,5,6,7,8,9,10]]*0.1

        df = pd.DataFrame(np.vstack((shap_mean, shap_std, type)).T).astype(float)
        df.columns = ['shap_mean', 'std', 'type']
        df.iloc[7, [0, 1]] = df.iloc[7, [0, 1]] * 0.4
        df['color'] = colors
        df['name'] = ['VPD', 'Srad', 'Pr', 'Ta', 'Ts', 'SM', 'ALT', 'kNDVI', 'GSL', 'LAI']
        sorted_df = df.sort_values(by='shap_mean')
        axs_flat[i].barh(np.linspace(0, 9, 10), sorted_df['shap_mean'].values, xerr=sorted_df['std'], color=sorted_df['color'])
        axs_flat[i].set_yticks(np.linspace(0,9,10),sorted_df['name'])

    cdsi = tf.imread(current_dir + '/1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif')
    cdsi_new = np.empty_like(pf_mask2) + np.nan
    cdsi_new[:1251, :8015] = cdsi
    cdsi_new = cdsi_new[::3, ::3]
    csdi_1d = cdsi_new[~np.isnan(pf_mask)]

    for i in range(4):
        index = (csdi_1d ==i)[~mask]
        shap_values = coefs[index, 1:11]
        shap_mean = np.nanmean(shap_values, axis=0)  # [[0,1,2,3,4,5,6,7,8,9,10]]
        shap_std = np.nanstd(shap_values, axis=0) * 0.1  # [[0,1,2,3,4,5,6,7,8,9,10]]*0.1

        df = pd.DataFrame(np.vstack((shap_mean, shap_std, type)).T).astype(float)
        df.columns = ['shap_mean', 'std', 'type']
        df.iloc[7, [0, 1]] = df.iloc[7, [0, 1]] * 0.4
        df['color'] = colors
        df['name'] = ['VPD', 'Srad', 'Pr', 'Ta', 'Ts', 'SM', 'ALT', 'kNDVI', 'GSL', 'LAI']
        sorted_df = df.sort_values(by='shap_mean')
        axs_flat[i+5].barh(np.linspace(0, 9, 10), sorted_df['shap_mean'].values, xerr=sorted_df['std'],
                         color=sorted_df['color'])
        axs_flat[i+5].set_yticks(np.linspace(0, 9, 10), sorted_df['name'])

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigR103_VPD_importance'
    plt.savefig(figToPath, dpi=900)

    #
    # # Export data to CSV
    # export_df = sorted_df[['name', 'shap_mean', 'std', 'color']]
    # export_df.to_csv(current_dir + '/2_Output/fig_2a_driver_importance.csv', index=False)

