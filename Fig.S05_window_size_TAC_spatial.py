import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; #matplotlib.use('Qt5Agg')
import tifffile as tf
from plot_NH import *
import os
from PIL import Image
import rasterio
import cv2

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
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

    fig = plt.figure(figsize=(12*0.8, 4*0.8))
    for i in range(3):
        trend = np.load(current_dir+'/2_Output/spatial_resilience/resilience_trend_modis_'+str(i+3)+'yr.npy')

        trend_23_map = pf_mask * 1
        trend_23_map[~np.isnan(trend_23_map)] = trend[:, 4]*23
        trend_23_map = trend_23_map[::-1, :]
        trend_23_map[np.isnan(trend_23_map)] = 0
        kernel = np.ones((3, 3), np.float32) / 9
        trend_23_map = cv2.filter2D(trend_23_map, -1, kernel)
        # plt.figure(); plt.hist(trend[:, 4]*23,50)

        ax1 = fig.add_subplot(1, 3, i+1, projection=ccrs.NorthPolarStereo())
        this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_23_map,
                               cmap='RdBu_r', vmin=-0.015, vmax=0.015,
                               transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.set_boundary(circle, transform=ax1.transAxes)
        plt.colorbar(this1, orientation='horizontal', label='ar1_trend_00-23', fraction=0.03, pad=0.05)

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS05_window_size_spatial'
    plt.savefig(figToPath, dpi=900)



