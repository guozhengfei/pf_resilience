import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')
import tifffile as tf
from plot_NH import *
import os
from PIL import Image
import rasterio
import cv2

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
    trend = np.load(current_dir+'/2_Output/spatial_resilience/VR_trend_variance_5yr_kndvi_modis_sg_rolling.npy')
    trend_07 = trend[:,0]*23
    trend_07[trend[:,1]>0.05]=np.nan
    np.sum(trend_07[~np.isnan(trend_07)] > 0) / trend_07[~np.isnan(trend_07)].shape[0]
    # plt.figure(); plt.hist(trend_07[~np.isnan(trend_07)],20,ec='k',range=[-0.1,0.1])

    trend_22 = trend[:,2]*23
    trend_22[trend[:, 3] > 0.01] = np.nan
    np.sum(trend_22[~np.isnan(trend_22)] > 0) / trend_22[~np.isnan(trend_22)].shape[0]
    # plt.figure(); plt.hist(trend_22[~np.isnan(trend_22)], 20, ec='k', range=[-0.04, 0.04])

    trend_diff = trend_22 - trend_07
    np.sum(trend_diff[~np.isnan(trend_diff)] >= 0) / trend_diff[~np.isnan(trend_diff)].shape[0]

    fig, axs = plt.subplots(1,3,figsize=(5,1.6))
    n = axs[0].hist(trend_07[~np.isnan(trend_07)], 15, range=[-0.001*0.6,0.001*0.6],alpha=0, density=True)
    axs[0].bar(n[1][:8],n[0][:8],ec='k',width =n[1][1]-n[1][0],color='#92c5de')
    axs[0].bar(n[1][8:-1], n[0][8:], ec='k', width=n[1][1] - n[1][0], color='#f4a582')
    axs[0].set_ylabel('Frequency (%)')

    n = axs[1].hist(trend_22[~np.isnan(trend_22)]+0.00001, 15, ec='k', range=[-0.0004*0.6, 0.0004*0.6], color='#d6604d', alpha=0, density=True)
    axs[1].bar(n[1][:8], n[0][:8], ec='k', width=n[1][1] - n[1][0], color='#92c5de')
    axs[1].bar(n[1][8:-1], n[0][8:], ec='k', width=n[1][1] - n[1][0], color='#f4a582')
    axs[1].set_ylabel('Frequency (%)')

    n = axs[2].hist(trend_diff[~np.isnan(trend_diff)]+0.00005, 15, ec='k', range=[-0.0012*0.6, 0.0012*0.6], color='#d6604d', alpha=0, density=True)
    axs[2].bar(n[1][:8], n[0][:8], ec='k', width=n[1][1] - n[1][0], color='#92c5de')
    axs[2].bar(n[1][8:-1], n[0][8:], ec='k', width=n[1][1] - n[1][0], color='#f4a582')
    axs[2].set_ylabel('Frequency (%)')
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig01_resilience_hist_variance'
    plt.savefig(figToPath, dpi=600)

    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

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

    # load ar1 data
    trend_07_map = pf_mask * 1
    trend_07_map[~np.isnan(trend_07_map)] = trend[:, 0]*23
    trend_07_map = trend_07_map[::-1,:]
    trend_07_map[np.isnan(trend_07_map)]=0
    kernel = np.ones((3, 3), np.float32) / 9
    trend_07_map = cv2.filter2D(trend_07_map, -1, kernel)

    trend_23_map = pf_mask * 1
    trend_23_map[~np.isnan(trend_23_map)] = trend[:, 2]*23
    trend_23_map = trend_23_map[::-1, :]
    trend_23_map[np.isnan(trend_23_map)] = 0
    trend_23_map = cv2.filter2D(trend_23_map, -1, kernel)

    trend_diff_map = trend_23_map-trend_07_map
    # trend_all_map[~np.isnan(trend_all_map)] = trend[:, 4]*23
    # trend_all_map = trend_all_map[::-1, :]
    # trend_all_map[np.isnan(trend_all_map)] = 0
    # trend_all_map = cv2.filter2D(trend_all_map, -1, kernel)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map,
                           cmap='RdBu_r', vmin=-0.0004*0.6, vmax=0.0004*0.6,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='Var_trend_00-07', fraction=0.03, pad=0.05)

    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.NorthPolarStereo())
    this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, trend_23_map ,
                           cmap='RdBu_r', vmin=-0.00015*0.6, vmax=0.00015*0.6,
                           transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_boundary(circle, transform=ax2.transAxes)
    plt.colorbar(this2, orientation='horizontal', label='Var_trend_00-07', fraction=0.03, pad=0.05)

    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.NorthPolarStereo())
    this3 = ax3.pcolormesh(grid_longitudes, grid_latitudes, trend_diff_map,
                           cmap='RdBu_r', vmin=-0.0004*0.6, vmax=0.0004*0.6,
                           transform=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.set_boundary(circle, transform=ax3.transAxes)
    plt.colorbar(this3, orientation='horizontal', label='Var_trend_diff', fraction=0.03, pad=0.05)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig01_resilience_patterns_spatialal_variance'
    plt.savefig(figToPath, dpi=900)

    # #trend change grouped by PFT
    # PFT_1d = pf_mask[~np.isnan(pf_mask)]
    # PFT = PFT_1d
    #
    # set(PFT_1d.reshape(-1))
    # for i in [1,3,4,5,6,7,8,9,10,11]:
    #     frac = np.nansum(PFT==i)/np.nansum(~np.isnan(PFT_1d))
    #     print(i,frac)
    #
    # plt.figure(figsize=(4,3))
    # # Needle forest
    # index = (PFT==1) | (PFT==3)
    # trend_10 = trend_07[index]
    # trend_23 = trend_22[index]
    # plt.bar(0,np.nanmean(trend_10),yerr=np.nanstd(trend_10)*0.05,color='C0',width=0.3,ec='k', hatch='//',alpha=0.7)
    # plt.bar(0.3,np.nanmean(trend_23),yerr=np.nanstd(trend_23)*0.05,color='C0',width=0.3,ec='k',alpha=0.7)
    # np.sum(trend_10>0)/np.sum(~np.isnan(trend_10))
    #
    # # Mixed forest
    # index = (PFT==4) | (PFT==5)
    # trend_10 = trend_07[index]
    # trend_23 = trend_22[index]
    # plt.bar(1,np.nanmean(trend_10),yerr=np.nanstd(trend_10)*0.05,color='C1',width=0.3,ec='k', hatch='//',alpha=0.7)
    # plt.bar(1.3,np.nanmean(trend_23),yerr=np.nanstd(trend_23)*0.05,color='C1',width=0.3,ec='k',alpha=0.7)
    # np.sum(trend_10>0)/np.sum(~np.isnan(trend_10))
    #
    # #Shrubland
    # index = (PFT==6) | (PFT==7)
    # trend_10 = trend_07[index]
    # trend_23 = trend_22[index]
    # plt.bar(2,np.nanmean(trend_10),yerr=np.nanstd(trend_10)*0.05,color='C2',width=0.3,ec='k', hatch='//',alpha=0.7)
    # plt.bar(2.3,np.nanmean(trend_23),yerr=np.nanstd(trend_23)*0.05,color='C2',width=0.3,ec='k',alpha=0.7)
    # np.sum(trend_10>0)/np.sum(~np.isnan(trend_10))
    #
    # #Savana
    # index = (PFT==8) | (PFT==9)
    # trend_10 = trend_07[index]
    # trend_23 = trend_22[index]
    # plt.bar(3,np.nanmean(trend_10),yerr=np.nanstd(trend_10)*0.05,color='C3',width=0.3,ec='k', hatch='//',alpha=0.7)
    # plt.bar(3.3,np.nanmean(trend_23),yerr=np.nanstd(trend_23)*0.05,color='C3',width=0.3,ec='k',alpha=0.7)
    # np.sum(trend_10>0)/np.sum(~np.isnan(trend_10))
    #
    # #Grass
    # index = (PFT==10)
    # trend_10 = trend_07[index]
    # trend_23 = trend_22[index]
    # plt.bar(4,np.nanmean(trend_10),yerr=np.nanstd(trend_10)*0.05,color='C4',width=0.3,ec='k', hatch='//',alpha=0.7)
    # plt.bar(4.3,np.nanmean(trend_23),yerr=np.nanstd(trend_23)*0.05,color='C4',width=0.3,ec='k',alpha=0.7)
    #
    # np.sum(trend_10>0)/np.sum(~np.isnan(trend_10))
    # fig.tight_layout()
    # figToPath = current_dir + '/4_Figures/Fig01_resilience_PFT'
    # plt.savefig(figToPath, dpi=600)

