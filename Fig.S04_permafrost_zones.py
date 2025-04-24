import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import matplotlib;
matplotlib.use('Qt5Agg')
from PIL import Image
import multiprocess as mp
import os
from plot_NH import *
import tifffile as tf

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))
    return smoothed_arr

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    ## PFT information
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = 0.7
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 >= 11] = 0.1
    pf_mask = pf_mask2[::3, ::3]

    cdsi = tf.imread(current_dir + '/1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif')
    cdsi_new = np.empty_like(pf_mask2) + np.nan
    cdsi_new[:1251, :8015] = cdsi
    cdsi_new = cdsi_new[::3, ::3]
    cdsi_new[cdsi_new>4]=np.nan
    plt.figure(); plt.imshow(cdsi_new)

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
    # Define custom colors for each land cover type
    custom_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
    # Create a custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(custom_colors)

    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, cdsi_new,cmap=cmap,transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='permafrost zones', fraction=0.03, pad=0.05)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS01_permafrost_zones'
    plt.savefig(figToPath, dpi=900)

    array = cdsi_new[~np.isnan(cdsi_new)]
    nums = np.sort(list(set(array)))
    fracs=[]
    for i in nums:
        fracs.append(np.sum(array==i)/array.shape[0])

    fig, axs = plt.subplots(1, figsize=(3 * 0.8, 3 * 0.8))
    axs.bar(range(4), fracs, ec='k', width=0.5, color=custom_colors)
    xtick = ['D','S','C','I']
    axs.set_xticks(range(4), xtick, rotation=90)
    axs.set_ylabel('Frequency')
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS04_permafrost_zones_hist'
    plt.savefig(figToPath, dpi=900)


