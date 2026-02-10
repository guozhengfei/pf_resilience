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
from rasterio.warp import reproject, Resampling

def fill_nan_with_climatology(EVI, bands_year=12):
    num_years = EVI.shape[1] // bands_year
    climatology = np.zeros((EVI.shape[0], EVI.shape[1]))

    for year in range(num_years):
        if year < 2:
            start = 0
            end = 5 * bands_year
        elif year > num_years - 3:
            start = (num_years - 5) * bands_year
            end = num_years * bands_year
        else:
            start = (year - 2) * bands_year
            end = (year + 3) * bands_year

        climatology[:, year * bands_year:(year + 1) * bands_year] = np.nanmean(
            EVI[:, start:end].reshape(EVI.shape[0], 5, bands_year), axis=1
        )

    EVI_filled = np.where(np.isnan(EVI), climatology, EVI)

    return EVI_filled

def cal_slope_ktest(row):
    import numpy as np
    import pymannkendall as mk
    if np.sum(np.isnan(row))>0:
        slopes = [np.nan]*2
    else:
        coef_all = mk.original_test(row)
        slopes = [coef_all.slope, coef_all.p]
    return slopes

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

    EVI_folder = current_dir + '/1_Input/vod_16day_month_aggregate/month_aggregate/'
    filenames = os.listdir(EVI_folder)
    evi_filenames = []
    for name in filenames:
        if name.startswith('.'): continue
        evi_filenames.append(name)
    evi_filenames = np.sort(evi_filenames)  # [:529]

    yr_num = 34
    bands_year = 12

    dst_height = pf_mask_ds.height
    dst_width = pf_mask_ds.width
    dst_transform = pf_mask_ds.transform
    dst_crs = pf_mask_ds.crs

    EVI = []
    for name in tqdm(evi_filenames, desc='Reproject/extract', unit='file'):
        src_path = os.path.join(EVI_folder, name)
        with rasterio.open(src_path) as src:
            # allocate destination array matching pf_mask2 georeference/size
            dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear)

        evi_ds = dst[::3, ::3]  # match downsample factor used for pf_mask
        evi_val = evi_ds[~np.isnan(pf_mask)]
        EVI.append(evi_val)

    # close pf_mask dataset
    pf_mask_ds.close()
    # plt.figure(); plt.imshow(evi_ds)

    EVI2 = np.array(EVI).T
    EVI2[EVI2<=0.2]=np.nan

    evi_mean = np.nanmean(EVI2,axis=1)
    EVI2 = fill_nan_with_climatology(EVI2,12)
    EVI2 = fill_nan_with_climatology(EVI2,12)
    EVI2 = fill_nan_with_climatology(EVI2,12)
    ratio = np.sum(np.isnan(EVI2),axis=1)/EVI2.shape[1]
    plt.figure(); plt.hist(ratio,50)

    EVI2 = EVI2[ratio<0.7,:]
    # plt.figure();plt.hist(evi_mean,50)

    evi_yr = EVI2.reshape(EVI2.shape[0], 34, 12)
    evi_yr[:,:,0:4]=np.nan
    evi_yr[:, :, -4:] = np.nan
    evi_yr = np.nanmean(evi_yr, axis=2)
    plt.figure(); plt.plot(np.linspace(1988, 2021, 34), np.nanmean(evi_yr, axis=0), 'o-')


    divided_arrays = [row for row in evi_yr]
    with mp.Pool(6) as pool:
        results = list(pool.map(cal_slope_ktest, divided_arrays))
    trends = np.array(results)

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

    trend_07_map = np.full_like(pf_mask, np.nan)

    # Create a boolean mask for the condition to avoid nested indexing issues
    valid_mask = ~np.isnan(pf_mask)
    # Assuming ratio has the same shape as the flattened valid_mask area
    mask2 = ratio*np.nan
    mask2[ratio<0.65]=trends[:, 0]
    trend_07_map[valid_mask] = mask2
    trend_07_map[np.isnan(trend_07_map)] = 0
    kernel = np.ones((3, 3), np.float32) / 9
    trend_07_map = cv2.filter2D(trend_07_map, -1, kernel)
    trend_07_map[trend_07_map==0] = np.nan
    plt.figure(); plt.hist(trend_07_map[~np.isnan(trend_07_map)],50)
    # plt.figure(); plt.imshow(pf_mask)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10 * 0.6, 4 * 0.6), gridspec_kw={'width_ratios': [1.5, 1]})

    # 在第一个子图（ax0）上绘图
    ax0.plot(np.linspace(1982, 2022, 41), np.nanmean(evi_yr, axis=0)[:-2], 'o-')
    coef = st.linregress(np.linspace(1982, 2022, 41), np.nanmean(evi_yr, axis=0)[:-2])
    ax0.plot([1982, 2022], [1982 * coef.slope + coef.intercept, 2022 * coef.slope + coef.intercept], 'r-')
    ax1.axis('off')
    ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map,
                           cmap='RdBu', vmin=-0.0015, vmax=0.0015,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='kNDVI_trend', fraction=0.03, pad=0.05)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R1_04_gimms_VI_trend'
    plt.savefig(figToPath, dpi=900)

    # fig = plt.figure(figsize=(2 * 0.8, 2 * 0.8))
    # plt.hist(trends[:, 0], 20, density=True, ec='k')
    # plt.xlim([-0.02, 0.04])
    # fig.tight_layout()
    # figToPath = current_dir + '/4_Figures/FigS03_GPP_trend_fluxcom_hist'
    # plt.savefig(figToPath, dpi=900)


