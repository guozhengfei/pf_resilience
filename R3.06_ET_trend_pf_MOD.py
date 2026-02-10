import cv2
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import scipy.signal as ss
import multiprocess as mp
import rasterio
from tqdm import tqdm
import os
import cv2
import xarray as xr
from plot_NH import *
import scipy.stats as st
import tifffile as tf
def cal_slope_ktest(row):
    import numpy as np
    import pymannkendall as mk
    if np.sum(np.isnan(row))>0:
        slopes = [np.nan]*2
    else:
        coef_all = mk.original_test(row)
        slopes = [coef_all.slope, coef_all.p]
    return slopes

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'

    # open pf mask as dataset to keep georeference (transform/crs/size)
    pf_mask_ds = rasterio.open(pf_mask_path2)
    pf_mask2 = pf_mask_ds.read(1).astype(float)

    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan

    # create the pf_mask used later (downsampled and flipped) to match original pipeline
    pf_mask = pf_mask2[::3, ::3]
    #plt.figure();plt.imshow(pf_mask)

    ET1 = tf.imread(current_dir + '/1_Input/MOD_ET_2000_2024_Stacked.tif')*0.1/8*365
    plt.figure(); plt.imshow(ET1[:,:,0])

    ET1_rsp = np.reshape(ET1,(ET1.shape[0]*ET1.shape[1],ET1.shape[2]))

    mask = np.isnan(ET1_rsp[:,0])
    ET_valid = ET1_rsp[~mask,:-1]

    import pandas as pd
    evi_yr = pd.DataFrame(np.nanmean(ET_valid, axis=0))
    evi_yr = evi_yr.rolling(window=3, center=True, min_periods=1).mean()
    # plt.figure(); plt.plot(np.linspace(2002, 2023, 22), evi_yr[2:], 'o-')


    divided_arrays = [row for row in ET_valid]
    with mp.Pool(6) as pool:
        results = list(pool.map(cal_slope_ktest, divided_arrays))
    trends = np.array(results)
    np.sum(trends[:,0]>0)/trends.shape[0]
    dataset = rasterio.open(current_dir + '/1_Input/MOD_ET_2000_2024_Stacked.tif')
    left, bottom, right, top = np.squeeze(dataset.bounds)
    latitudes = np.linspace(top, bottom, dataset.height)
    longitudes = np.linspace(left, right, dataset.width)

    # Create a 2D grid using meshgrid
    grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)
    # grid_longitudes = grid_longitudes[::3, ::3]
    # grid_latitudes = grid_latitudes[::3, ::3]

    # make circular boundary for polar stereographic circular plots
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    trend_07_map = ET1[:,:,0].copy()

    trend_07_map[~np.isnan(trend_07_map)] = trends[:, 0]
    trend_07_map[np.isnan(trend_07_map)] = 0
    kernel = np.ones((3, 3), np.float32) / 9
    trend_07_map = cv2.filter2D(trend_07_map, -1, kernel)
    trend_07_map[trend_07_map==0] = np.nan
    plt.figure(); plt.hist(trend_07_map[~np.isnan(trend_07_map)],50)
    # plt.figure(); plt.imshow(pf_mask)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10 * 0.6, 4 * 0.6), gridspec_kw={'width_ratios': [1.5, 1]})

    # 在第一个子图（ax0）上绘图
    ax0.plot(np.linspace(2002, 2023, 22), evi_yr[2:], 'o-')
    coef = st.linregress(np.linspace(2002, 2023, 22), evi_yr.values[2:,0])
    ax0.plot([2002, 2023], [2002 * coef.slope + coef.intercept, 2023 * coef.slope + coef.intercept], 'r-')
    ax0.set_ylabel("ET (mm/year)")
    ax0.set_xlabel("Year")

    ax1.axis('off')
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map,
                           cmap='RdBu', vmin=-1.8, vmax=1.8,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='ET trend (mm/year)', fraction=0.03, pad=0.05)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R1_mod_ET'
    plt.savefig(figToPath, dpi=900)



