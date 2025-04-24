import numpy as np
import matplotlib.pyplot as plt
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
    trend = np.load(current_dir+'/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07 = trend[:,0]*23
    trend_07[trend[:,1]>0.05]=np.nan

    trend_22 = trend[:,2]*23
    trend_22[trend[:, 3] > 0.01] = np.nan

    trend_all = trend[:,4]*23
    trend_all[trend[:, 5] > 0.01] = np.nan

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

    trend_23_map = pf_mask * 1
    trend_23_map[~np.isnan(trend_23_map)] = trend[:, 2]*23
    trend_23_map = trend_23_map[::-1, :]

    trend_4_class = pf_mask[::-1, :] +np.nan
    trend_4_class[(trend_07_map > 0) & (trend_23_map>0)] = 1
    trend_4_class[(trend_07_map > 0) & (trend_23_map <= 0)] = 2
    trend_4_class[(trend_07_map <= 0) & (trend_23_map > 0)] = 3
    trend_4_class[(trend_07_map <= 0) & (trend_23_map <= 0)] = 4
    # trend_4_class[np.isnan(trend_4_class)] = 0
    # kernel = np.ones((3, 3), np.float32) / 9
    # trend_4_class = np.round(cv2.filter2D(trend_4_class, -1, kernel))
    trend_4_class = cv2.medianBlur(np.uint8(trend_4_class),3)*1.0
    trend_4_class[trend_4_class==0]=np.nan

    import matplotlib.colors as mcolors
    colors = ['royalblue', 'gold', 'lightcoral', 'turquoise']
    cmap = mcolors.ListedColormap(colors)

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_4_class,
                           cmap=cmap, vmin=1, vmax=4,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='ar1_trend_00-23', fraction=0.03, pad=0.05)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig02_resilience_change_types'
    plt.savefig(figToPath, dpi=900)


    trends = np.vstack((trend_07,trend_22,trend_all)).T
    mask = np.isnan(trends).any(axis=1)
    trends = trends[~mask,:]

    frac1 = np.sum((trends[:, 0] > 0) & (trends[:, 1] > 0))/trends.shape[0]
    frac2 = np.sum((trends[:, 0] > 0) & (trends[:, 1] < 0))/trends.shape[0]
    frac3 = np.sum((trends[:, 0] < 0) & (trends[:, 1] > 0))/trends.shape[0]
    frac4 = np.sum((trends[:, 0] < 0) & (trends[:, 1] < 0))/trends.shape[0]

    plt.figure(figsize=(1.5,1.5)); plt.bar([0,1,2,3],[frac1,frac2,frac3,frac4],width=0.5, color=colors)
    plt.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig01_resilience_change_type_hist'
    plt.savefig(figToPath, dpi=600)

    #
    #trend change grouped by PFT
    trend_pft = trends[:, 0]*1
    trend_pft[(trends[:, 0] > 0) & (trends[:, 1] > 0)] = 1
    trend_pft[(trends[:, 0] > 0) & (trends[:, 1] < 0)] = 2
    trend_pft[(trends[:, 0] < 0) & (trends[:, 1] > 0)] = 3
    trend_pft[(trends[:, 0] < 0) & (trends[:, 1] < 0)] = 4
    trend_PFT = pf_mask[~np.isnan(pf_mask)]+np.nan
    trend_PFT[~mask] = trend_pft

    PFT = pf_mask[~np.isnan(pf_mask)]
    frac_arr = np.zeros((5,4))
    # Needle forest
    index = (PFT==1) | (PFT==3)
    trend_i = trend_PFT[index]
    for i in range(4):
        frac_arr[0,i]=(trend_i == i+1).sum()/np.sum(~np.isnan(trend_i))

    # Mixed forest
    index = (PFT==4) | (PFT==5)
    trend_i = trend_PFT[index]
    for i in range(4):
        frac_arr[1, i] = (trend_i == i + 1).sum() / np.sum(~np.isnan(trend_i))

    #Shrubland
    index = (PFT==6) | (PFT==7)
    trend_i = trend_PFT[index]
    for i in range(4):
        frac_arr[2, i] = (trend_i == i + 1).sum() / np.sum(~np.isnan(trend_i))

    #Savana
    index = (PFT==8) | (PFT==9)
    trend_i = trend_PFT[index]
    for i in range(4):
        frac_arr[3, i] = (trend_i == i + 1).sum() / np.sum(~np.isnan(trend_i))

    #Grass
    index = (PFT==10)
    trend_10 = trend_07[index]
    trend_i = trend_PFT[index]
    for i in range(4):
        frac_arr[4, i] = (trend_i == i + 1).sum() / np.sum(~np.isnan(trend_i))

    plt.figure(figsize=(4, 4))
    y_offset = frac_arr[:,0]*0
    for col in range(frac_arr.shape[1]):
        plt.bar([0,1,2,3,4], frac_arr[:,col], 0.6, bottom=y_offset, color=colors[col])
        y_offset = y_offset + frac_arr[:,col]

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/Fig0b_resilience_type_PFT'
    plt.savefig(figToPath, dpi=600)

