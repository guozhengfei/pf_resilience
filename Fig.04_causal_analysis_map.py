import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from PIL import Image
import rasterio
import cv2
import os
from plot_NH import *

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    vpd_tac = np.load(current_dir + '/2_Output/causal_VPD_tac.npy')
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
    causal_vpd[~mask] = vpd_tac[:, 1]
    causal_vpd[mask] = 1
    causal_vpd_map = pf_mask*1
    causal_vpd_map[~np.isnan(causal_vpd_map)] = causal_vpd
    causal_vpd_map = causal_vpd_map[::-1,:]*100+1
    causal_vpd_map = cv2.medianBlur(np.uint8(causal_vpd_map), 3)*0.01
    causal_vpd_map[(causal_vpd_map>=0.01)& (causal_vpd_map<=0.02)] = 0.01 #corresponding p_value: 0~0.5
    causal_vpd_map[(causal_vpd_map >= 0.02) & (causal_vpd_map <= 0.06)] = 0.02
    causal_vpd_map[causal_vpd_map >0.06] = 0.03
    causal_vpd_map[causal_vpd_map == 0] = np.nan

    # causal_srad = mask.astype(float) * 1 + np.nan
    # causal_srad[~mask] = srad_tac[:, 1]
    # causal_srad[mask] = 1
    # causal_srad_map = pf_mask * 1
    # causal_srad_map[~np.isnan(causal_srad_map)] = causal_srad
    # causal_srad_map = causal_srad_map[::-1, :] * 100 + 1
    # causal_srad_map = cv2.medianBlur(np.uint8(causal_srad_map), 3) * 0.01
    # causal_srad_map[(causal_srad_map >= 0.01) & (causal_srad_map <= 0.02)] = 0.01  # corresponding p_value: 0~0.5
    # causal_srad_map[(causal_srad_map >= 0.02) & (causal_srad_map <= 0.06)] = 0.02
    # causal_srad_map[causal_srad_map > 0.06] = 0.03
    # causal_srad_map[causal_srad_map == 0] = np.nan
    #
    # causal_pr = mask.astype(float) * 1 + np.nan
    # causal_pr[~mask] = pr_tac[:, 1]
    # causal_pr[mask] = 1
    # causal_pr_map = pf_mask * 1
    # causal_pr_map[~np.isnan(causal_pr_map)] = causal_pr
    # causal_pr_map = causal_pr_map[::-1, :] * 100 + 1
    # causal_pr_map = cv2.medianBlur(np.uint8(causal_pr_map), 3) * 0.01
    # causal_pr_map[(causal_pr_map >= 0.01) & (causal_pr_map <= 0.02)] = 0.01  # corresponding p_value: 0~0.5
    # causal_pr_map[(causal_pr_map >= 0.02) & (causal_pr_map <= 0.06)] = 0.02
    # causal_pr_map[causal_pr_map > 0.06] = 0.03
    # causal_pr_map[causal_pr_map == 0] = np.nan
    #
    # causal_ta = mask.astype(float) * 1 + np.nan
    # causal_ta[~mask] = ta_tac[:, 1]
    # causal_ta[mask] = 1
    # causal_ta_map = pf_mask * 1
    # causal_ta_map[~np.isnan(causal_ta_map)] = causal_ta
    # causal_ta_map = causal_ta_map[::-1, :] * 100 + 1
    # causal_ta_map = cv2.medianBlur(np.uint8(causal_ta_map), 3) * 0.01
    # causal_ta_map[(causal_ta_map >= 0.01) & (causal_ta_map <= 0.02)] = 0.01  # corresponding p_value: 0~0.5
    # causal_ta_map[(causal_ta_map >= 0.02) & (causal_ta_map <= 0.07)] = 0.02
    # causal_ta_map[causal_ta_map > 0.07] = 0.03
    # causal_ta_map[causal_ta_map == 0] = np.nan


    import matplotlib.colors as mcolors
    colors = ['blue', 'cornflowerblue', 'lightgrey']
    cmap = mcolors.ListedColormap(colors)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, causal_vpd_map,
                           cmap=cmap, vmin=0.01, vmax=0.03,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='p_value', fraction=0.03, pad=0.05)

    # ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.NorthPolarStereo())
    # this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, causal_srad_map,
    #                        cmap=cmap, vmin=0.01, vmax=0.03,
    #                        transform=ccrs.PlateCarree())
    # ax2.coastlines()
    # ax2.set_boundary(circle, transform=ax2.transAxes)
    # plt.colorbar(this2, orientation='horizontal', label='p_value', fraction=0.03, pad=0.05)
    #
    # ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.NorthPolarStereo())
    # this3 = ax3.pcolormesh(grid_longitudes, grid_latitudes, causal_pr_map,
    #                        cmap=cmap, vmin=0.01, vmax=0.03,
    #                        transform=ccrs.PlateCarree())
    # ax3.coastlines()
    # ax3.set_boundary(circle, transform=ax3.transAxes)
    # plt.colorbar(this3, orientation='horizontal', label='p_value', fraction=0.03, pad=0.05)
    #
    # ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.NorthPolarStereo())
    # this4 = ax4.pcolormesh(grid_longitudes, grid_latitudes, causal_ta_map,
    #                        cmap=cmap, vmin=0.01, vmax=0.03,
    #                        transform=ccrs.PlateCarree())
    # ax4.coastlines()
    # ax4.set_boundary(circle, transform=ax4.transAxes)
    # plt.colorbar(this4, orientation='horizontal', label='p_value', fraction=0.03, pad=0.05)
    plt.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig04_causal_analysis'
    plt.savefig(figToPath, dpi=900)

    fig, axs = plt.subplots(1, 4, figsize=(6, 1.5))
    frac1 = np.sum(vpd_tac[:,1]<=0.01)/vpd_tac.shape[0]
    frac2 = np.sum((vpd_tac[:, 1]> 0.01) &(vpd_tac[:, 1]<= 0.07)) / vpd_tac.shape[0]
    frac3 = 1-frac1-frac2
    axs[0].bar([0,1,2], [frac1,frac2,frac3], color=colors)
    
    frac1 = np.sum(srad_tac[:, 1] <= 0.01) / srad_tac.shape[0]
    frac2 = np.sum((srad_tac[:, 1] > 0.01) & (srad_tac[:, 1] <= 0.05)) / srad_tac.shape[0]
    frac3 = 1 - frac1 - frac2
    axs[1].bar([0, 1, 2], [frac1, frac2, frac3], color=colors)

    frac1 = np.sum(pr_tac[:, 1] <= 0.01) / pr_tac.shape[0]
    frac2 = np.sum((pr_tac[:, 1] > 0.01) & (pr_tac[:, 1] <= 0.05)) / pr_tac.shape[0]
    frac3 = 1 - frac1 - frac2
    axs[2].bar([0, 1, 2], [frac1, frac2, frac3], color=colors)

    frac1 = np.sum(ta_tac[:, 1] <= 0.01) / ta_tac.shape[0]
    frac2 = np.sum((ta_tac[:, 1] > 0.01) & (ta_tac[:, 1] <= 0.05)) / ta_tac.shape[0]
    frac3 = 1 - frac1 - frac2
    axs[3].bar([0, 1, 2], [frac1, frac2, frac3], color=colors)

    plt.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig04_causal_analysis_hist'
    plt.savefig(figToPath, dpi=600)
