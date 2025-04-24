import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')
from PIL import Image
import rasterio
import cv2
import os
from plot_NH import *

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    vpd_tac = np.load(current_dir + '/2_Output/causal_vpd_tac.npy')
    srad_tac = np.load(current_dir + '/2_Output/causal_srad_tac.npy')
    pr_tac = np.load(current_dir + '/2_Output/causal_pr_tac.npy')
    ta_tac = np.load(current_dir + '/2_Output/causal_ta_tac.npy')
    mask = np.load(current_dir + '/2_Output/Temporal/Temporal_r_shap_obs_pre.npy.npz')['array2']

    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]
    pf_mask_1d = pf_mask.flatten()

    dataset = rasterio.open(pf_mask_path2)
    left, bottom, right, top = np.squeeze(dataset.bounds)
    latitudes = np.linspace(top, bottom, dataset.height)
    longitudes = np.linspace(left, right, dataset.width)

    # Create a 2D grid using meshgrid
    grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)
    grid_longitudes = grid_longitudes[::3, ::3]
    grid_latitudes = grid_latitudes[::3, ::3]

    # make circular boundary for polar stereographic circular plots
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    causal_vpd = mask.astype(float)*1+np.nan
    causal_vpd[~mask] = vpd_tac[:, 1]*1
    causal_vpd[mask] = 1
    causal_vpd_map = pf_mask*1
    causal_vpd_map[~np.isnan(causal_vpd_map)] = causal_vpd

    causal_vpd_map = causal_vpd_map[::-1,:]*100+1
    causal_vpd_map = cv2.medianBlur(np.uint8(causal_vpd_map), 3)*0.01
    causal_vpd_map[causal_vpd_map == 0] = np.nan
    causal_vpd_map[(causal_vpd_map<=0.02)] = 0.01 #corresponding p_value: 0~0.5
    causal_vpd_map[(causal_vpd_map>0.01) & (causal_vpd_map <= 0.08)] = 0.02

    import matplotlib.colors as mcolors
    colors = ['#2166ac', '#4393c3', 'lightgrey']
    cmap = mcolors.ListedColormap(colors)

    fig = plt.figure(figsize=(8*0.7, 8*0.7))
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, causal_vpd_map,
                           cmap=cmap, vmin=0.005, vmax=0.03,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='p_value', fraction=0.03, pad=0.05)

    ax2 = fig.add_subplot(2, 2, 2)
    conv = vpd_tac[:,2:]
    conv_mean = np.nanmedian(conv,axis=0)
    conv_sd = np.nanstd(conv,axis=0)*0.2
    ax2.plot(range(5,20),conv_mean,lw=2.5,c='#d6604d')
    ax2.fill_between(range(5,20),conv_mean+conv_sd,conv_mean-conv_sd,alpha=0.5,color='#d6604d')

    ax3 = fig.add_subplot(2, 2, 3)
    causal_vpd = mask.astype(float) * 1 + np.nan
    causal_vpd[~mask] = vpd_tac[:, 1]
    causal_vpd[mask] = 1
    causal_vpd_map = pf_mask * 1
    causal_vpd_map[~np.isnan(causal_vpd_map)] = causal_vpd

    PFT_1d = pf_mask
    index = (PFT_1d == 1) | (PFT_1d == 3)
    NF = causal_vpd_map[index]
    frac = np.sum(NF<0.08)/np.sum(NF!=1)
    ax3.bar([0], [1], color='lightgrey', width=0.5)
    ax3.bar([0], [frac], color='#4393c3',width=0.5)

    index = (PFT_1d == 4) | (PFT_1d == 5)
    NF = causal_vpd_map[index]
    frac = np.sum(NF < 0.08) / np.sum(NF != 1)
    ax3.bar(1, 1, color='lightgrey', width=0.5)
    ax3.bar(1, frac, color='#4393c3', width=0.5)


    index = (PFT_1d == 6) | (PFT_1d == 7)
    NF = causal_vpd_map[index]
    frac = np.sum(NF < 0.08) / np.sum(NF != 1)
    ax3.bar(2, 1, color='lightgrey', width=0.5)
    ax3.bar(2, frac, color='#4393c3', width=0.5)

    index = (PFT_1d == 8) | (PFT_1d == 9)
    NF = causal_vpd_map[index]
    frac = np.sum(NF < 0.08) / np.sum(NF != 1)
    ax3.bar(3, 1, color='lightgrey', width=0.5)
    ax3.bar(3, frac, color='#4393c3', width=0.5)

    index = (PFT_1d == 10)
    NF = causal_vpd_map[index]
    frac = np.sum(NF < 0.08) / np.sum(NF != 1)
    ax3.bar(4, 1, color='lightgrey', width=0.5)
    ax3.bar(4, frac, color='#4393c3', width=0.5)
    ax3.set_xticks(range(5), ['NF', 'MF', 'SVA', 'SHR', 'GRA'])

    ax4 = fig.add_subplot(2, 2, 4)
    import tifffile as tf
    import cv2
    cdsi = tf.imread(current_dir + '/1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif')
    cdsi_new = np.empty_like(pf_mask2) + np.nan
    cdsi_new[:1251, :8015] = cdsi
    cdsi_new = cdsi_new[::3, ::3]
    csdi_1d = cdsi_new

    index = (csdi_1d == 0)
    type1 = causal_vpd_map[index]
    frac = np.sum(type1 < 0.08) / np.sum(~np.isnan(type1) & (type1 != 1))
    ax4.bar(0, 1, color='lightgrey', width=0.5)
    ax4.bar(0, frac, color='#4393c3', width=0.5)

    index = (csdi_1d == 1)
    type1 = causal_vpd_map[index]
    frac = np.sum(type1 < 0.08) / np.sum(~np.isnan(type1) & (type1 != 1))
    ax4.bar(1, 1, color='lightgrey', width=0.5)
    ax4.bar(1, frac, color='#4393c3', width=0.5)

    index = (csdi_1d == 2)
    type1 = causal_vpd_map[index]
    frac = np.sum(type1 < 0.08) / np.sum(~np.isnan(type1) & (type1 != 1))
    ax4.bar(2, 1, color='lightgrey', width=0.5)
    ax4.bar(2, frac, color='#4393c3', width=0.5)

    index = (csdi_1d == 3)
    type1 = causal_vpd_map[index]
    frac = np.sum(type1 < 0.08) / np.sum(~np.isnan(type1) & (type1 != 1))
    ax4.bar(3, 1, color='lightgrey', width=0.5)
    ax4.bar(3, frac, color='#4393c3', width=0.5)

    ax4.set_xlim([-0.32 - 0.6, 4.66 - 0.6])
    ax4.set_xticks(range(4), ['D', 'S', 'C', 'I'])
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig04_causal_analysis'
    plt.savefig(figToPath, dpi=900)

    fig, axs = plt.subplots(1, 1, figsize=(1.3, 1.5))
    frac1 = np.sum(vpd_tac[:, 1] <= 0.08) / vpd_tac.shape[0]
    # frac2 = np.sum((vpd_tac[:, 1] > 0.01) & (vpd_tac[:, 1] <= 0.08)) / vpd_tac.shape[0]
    frac3 = 1 - frac1# - frac2
    axs.bar([0, 1], [frac1, frac3], color=colors[1:],width=0.4)
    axs.set_ylabel('Fraction')
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig04_causal_analysis_hist_v2'
    plt.savefig(figToPath, dpi=600)




