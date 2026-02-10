import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
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

    fig = plt.figure(figsize=(8*0.8, 12*0.8))
    for i in range(3):
        trend = np.load(current_dir+'/2_Output/spatial_resilience/resilience_trend_modis_rolling_deseason_'+str(i+3)+'yr.npy')

        trend_07_map = pf_mask * 1
        trend_07_map[~np.isnan(trend_07_map)] = trend[:, 0] * 23
        trend_07_map = trend_07_map[::-1, :]
        trend_07_map[np.isnan(trend_07_map)] = 0
        kernel = np.ones((3, 3), np.float32) / 9
        trend_07_map = cv2.filter2D(trend_07_map, -1, kernel)
        trend_07_map[trend_07_map == 0] = np.nan

        trend_23_map = pf_mask * 1
        trend_23_map[~np.isnan(trend_23_map)] = trend[:, 2]*23
        trend_23_map = trend_23_map[::-1, :]
        trend_23_map[np.isnan(trend_23_map)] = 0
        kernel = np.ones((3, 3), np.float32) / 9
        trend_23_map = cv2.filter2D(trend_23_map, -1, kernel)
        trend_23_map[trend_23_map==0] = np.nan

        ax1 = fig.add_subplot(3, 2, 2*i+1, projection=ccrs.NorthPolarStereo())
        this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map,
                               cmap='RdBu_r', vmin=np.nanpercentile(abs(trend_07_map),80)*-1, vmax=np.nanpercentile(abs(trend_07_map),80),
                               transform=ccrs.PlateCarree())
        ax1.coastlines()
        ax1.set_boundary(circle, transform=ax1.transAxes)
        plt.colorbar(this1, orientation='horizontal', label='TAC trend:2000-2008', fraction=0.03, pad=0.05)

        ax2 = fig.add_subplot(3, 2, 2 * i + 2, projection=ccrs.NorthPolarStereo())
        this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, trend_23_map,
                               cmap='RdBu_r', vmin=np.nanpercentile(abs(trend_23_map), 80) * -1,
                               vmax=np.nanpercentile(abs(trend_23_map), 80),
                               transform=ccrs.PlateCarree())
        ax2.coastlines()
        ax2.set_boundary(circle, transform=ax2.transAxes)
        plt.colorbar(this2, orientation='horizontal', label='TAC trend:2009-2023', fraction=0.03, pad=0.05)

    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS06_window_size_rolling_deseason_spatial'
    plt.savefig(figToPath, dpi=900)



