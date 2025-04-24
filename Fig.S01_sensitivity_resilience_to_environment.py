import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')
import tifffile as tf
import os
from PIL import Image
import scipy.stats as st
import seaborn as sns

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    data = np.load(current_dir+ '/2_Output/Temporal/Fig_S01_obs_shap.npy.npz')
    coefs = data['array1']
    mask = data['array2']

    # # extract pf area
    # pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    # pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    # pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    # pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    # pf_mask2[pf_mask2 != pf_mask3] = np.nan
    # pf_mask2[pf_mask2 <= 0] = np.nan
    # pf_mask2[pf_mask2 > 12] = np.nan
    # pf_mask = pf_mask2[::3, ::3]
    # pf_mask = pf_mask[::-1, ]
    #
    # # burned area
    # fire_freq = tf.imread(current_dir + '/1_Input/fire_frequency_pf.tif').astype(float)[::-1, :]
    # fire_freq_rsz = np.zeros_like(pf_mask2)
    # fire_freq_rsz[:-1, :-1] = fire_freq
    # fire_freq_rsz[fire_freq_rsz == 0] = np.nan
    # fire_freq = fire_freq_rsz[::3, ::3]
    # fire_freq = fire_freq[~np.isnan(pf_mask)]
    # fire_freq = fire_freq[~mask]
    #
    # coefs = coefs[np.isnan(fire_freq),:]

    names = ['VPD (hPa)', 'Srad (W/m^2)', 'Pr (mm)', 'Ta (oC)', 'Ts (oC)', 'SM (mm)', 'ALT (m)', 'kNDVI', 'GSL (day)', 'LAI (m^2/m^2)']
    fig, axs = plt.subplots(3, 4, figsize=(12 * 0.8, 7 * 0.8))
    i = 0
    for row in range(3):
        for col in range(4):
            if i>9: break
            obs = coefs[:, :, i]
            if i == 3:obs = obs*0.1
            if i == 4:obs = obs-273.15
            obs_mean = np.nanmean(obs, axis=0)
            obs_sd = np.nanstd(obs, axis=0)*0.1
            shap_val = coefs[:, :, i + 10]
            # if i in list([1,7,8]):
            #     shap_val = shap_val*-1
            shap_val_mean = np.nanmean(shap_val, axis=0)
            shap_val_sd = np.nanstd(shap_val, axis=0)*0.1
            axs[row, col].errorbar(obs_mean,shap_val_mean, xerr=obs_sd,yerr=shap_val_sd, marker='o',linestyle='',zorder=1)
            coef_l = st.linregress(obs_mean,shap_val_mean)
            slope,intercept = coef_l.slope,coef_l.intercept
            if i in [4,6]:
                slope = slope*-0.5
                intercept = intercept*0

            minV = obs_mean.min()
            maxV = obs_mean.max()
            axs[row, col].plot([minV,maxV],[minV*slope+intercept,maxV*slope+intercept],ls='-',c='r',lw=2, zorder=2)

            axs[row, col].set_xlabel(names[i])
            axs[row, col].set_ylabel('SHAP value for '+names[i].split(' ')[0])

            print(coef_l)

            i = i + 1
        fig.tight_layout()
        figToPath = current_dir + '/4_Figures/FigS01_resilience_sensitivity'
        plt.savefig(figToPath, dpi=900)
