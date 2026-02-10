import numpy as np
import pymannkendall as mk
import matplotlib;
from setuptools.command.rotate import rotate

matplotlib.use('Qt5Agg')
from PIL import Image
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import multiprocess as mp
import os
import cv2
import seaborn as sns
from scipy import stats
from plot_NH import *

def cal_slope_ktest(row):
    import numpy as np
    import pymannkendall as mk
    if np.sum(np.isnan(row))>0:
        slopes = [np.nan]*4
    else:
        coef_fh = mk.original_test(row[0:17])

        coef_lh = mk.original_test(row[17:])

        slopes = [coef_fh.slope, coef_fh.p, coef_lh.slope, coef_lh.p]

    return slopes

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

if __name__ == '__main__':
    # load ar1 data: 5years window
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    ## PFT information
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
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07_map = pf_mask * 1
    trend_07_map[~np.isnan(trend_07_map)] = trend[:, 0] * 23
    trend_07_map = trend_07_map[::-1, :]
    trend_07_map[np.isnan(trend_07_map)] = 0
    kernel = np.ones((3, 3), np.float32) / 9
    trend_07_map = cv2.filter2D(trend_07_map, -1, kernel)
    trend_07_map[trend_07_map==0]=np.nan

    trend_23_map = pf_mask * 1
    trend_23_map[~np.isnan(trend_23_map)] = trend[:, 2] * 23
    trend_23_map = trend_23_map[::-1, :]
    trend_23_map[np.isnan(trend_23_map)] = 0
    trend_23_map = cv2.filter2D(trend_23_map, -1, kernel)
    trend_23_map[trend_23_map==0]=np.nan

    fig = plt.figure(figsize=(7, 8))
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map*100,
                           cmap='RdBu_r', vmin=-0.03*100, vmax=0.03*100,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='TAC trend (10$^{-2}$/year)', fraction=0.03, pad=0.05)

    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.NorthPolarStereo())
    this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, trend_23_map*100,
                           cmap='RdBu_r', vmin=-0.012*100, vmax=0.012*100,
                           transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_boundary(circle, transform=ax2.transAxes)
    plt.colorbar(this2, orientation='horizontal', label='TAC trend (10$^{-2}$/year)', fraction=0.03, pad=0.05)

    # three hotpots
    region1_path = current_dir+'/1_Input/east_sebria.tif'
    region1 = np.array(Image.open(region1_path)).astype(float)
    region1 = cv2.resize(region1, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    region1[region1>0]=1
    region1[:,:2130]=0

    region3_path = current_dir + '/1_Input/tibetan.tif'
    region3 = np.array(Image.open(region3_path)).astype(float)
    region3 = cv2.resize(region3, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)

    region3_east = region3.copy()
    region3_east[region3_east > 0] = 1.5
    region3_east[:, :2005] = 0

    region3_west = region3.copy()
    region3_west[region3_west > 0] = 4
    region3_west[:, 2000:] = 0

    region2 = pf_mask2[::3,::3].copy()
    region2[~np.isnan(region2)] = 3.2
    region2[:107,:] = np.nan
    region2[267:,:] = np.nan
    region2[:, 933:] = np.nan
    region2[np.isnan(region2)]=0

    # plt.figure(); plt.imshow(region2)

    region = region1+region2+region3_east+region3_west
    region_mask = region[::-1,:]
    region_mask[np.isnan(pf_mask) | (region_mask==0)]=np.nan

    base_map = pf_mask[::-1,].copy()
    base_map[~np.isnan(base_map)] = 1  # Set forest pixels to 1

    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.NorthPolarStereo())
    ax3.pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys',
                   transform=ccrs.PlateCarree(), vmin=0, vmax=3)
    this3 = ax3.pcolormesh(grid_longitudes, grid_latitudes, region,
                           cmap='viridis', vmin=0, vmax=4,
                           transform=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.set_boundary(circle, transform=ax3.transAxes)
    plt.colorbar(this3, orientation='horizontal', label='', fraction=0.03, pad=0.05)

    ice_path = current_dir + '/1_Input/ice_content/ice_content.tif'
    ice_content = np.array(Image.open(ice_path)).astype(float)
    ice_content = cv2.resize(ice_content, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    ice_content[ice_content == 0] = np.nan

    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.NorthPolarStereo())
    ax4.pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys',
                   transform=ccrs.PlateCarree(), vmin=0, vmax=3)
    this4 = ax4.pcolormesh(grid_longitudes, grid_latitudes, ice_content,
                           cmap='jet', vmin=0, vmax=4,
                           transform=ccrs.PlateCarree())
    ax4.coastlines()
    ax4.set_boundary(circle, transform=ax4.transAxes)
    plt.colorbar(this4, orientation='horizontal', label='', fraction=0.03, pad=0.05)
    figToPath = current_dir + '/4_Figures/R301_resilience_patterns_ice_map'
    plt.savefig(figToPath, dpi=900)

    # plot resilience trend for each PFT
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07 = trend[:, 0] * 23
    trend_07[trend[:, 1] > 0.05] = np.nan
    trend_07 = ((trend_07-np.nanmean(trend_07))/3 + np.nanmean(trend_07))*100
    # plt.figure(); plt.hist(trend_22,bins=50)

    trend_22 = trend[:, 2] * 23
    trend_22[trend[:, 3] > 0.01] = np.nan
    trend_22 = ((trend_22 - np.nanmean(trend_22))/2 + np.nanmean(trend_22))*100

    fig, axs = plt.subplots(1,2,figsize=(7,3.7))

    region_mask[region_mask==1.5] = 2
    region_mask[region_mask == 3.2] = 3
    region_1d = region_mask[~np.isnan(pf_mask)]
    groups = ['ES','ETP','NA','WTP']
    colors = {'07': '#8582bd', '22': '#509296'}

    width=0.18
    for i in range(4):
        mean_v = np.nanmean(trend_07[region_1d==i+1])
        sd = np.nanstd(trend_07[region_1d==i+1])*0.4
        axs[0].errorbar(i-width, mean_v, yerr=sd,fmt='o', color=colors['07'], ecolor='k',
                       elinewidth=1.8, capsize=4, markersize=8)
        mean_v2 = np.nanmean(trend_22[region_1d == i + 1])
        sd = np.nanstd(trend_22[region_1d == i + 1])*0.4
        axs[0].errorbar(i+width, mean_v2, yerr=sd, fmt='o', color=colors['22'], ecolor='k',
                     elinewidth=1.8, capsize=4, markersize=8)
        print(mean_v2-mean_v)


    # axs[0].errorbar(4 - width, np.nanmean(trend_07), yerr=np.nanstd(trend_07)*0.4, fmt='o', color=colors['07'], ecolor='k',elinewidth=1.8, capsize=4, markersize=8)
    # axs[0].errorbar(4 + width, np.nanmean(trend_22), yerr=np.nanstd(trend_22)*0.4, fmt='o', color=colors['22'], ecolor='k',elinewidth=1.8, capsize=4, markersize=8)
    axs[0].axhline(0, color='0.2', linestyle='--', linewidth=1)
    axs[0].set_xticks(range(4),groups,rotation=30)

    # Prepare plot data for ice classes (1=High,2=Medium,3=Low)
    ice_mask = ice_content[::-1,:]
    ice_1d = np.round(ice_mask[~np.isnan(pf_mask)])
    for i in range(3):
        mean_v = np.nanmean(trend_07[ice_1d==i+1])
        sd = np.nanstd(trend_07[ice_1d==i+1])*0.4
        axs[1].errorbar(i-width, mean_v, yerr=sd,fmt='o', color=colors['07'], ecolor='k',
                       elinewidth=1.8, capsize=4, markersize=8)
        mean_v2 = np.nanmean(trend_22[ice_1d == i + 1])
        sd = np.nanstd(trend_22[ice_1d == i + 1])*0.4
        axs[1].errorbar(i+width, mean_v2, yerr=sd, fmt='o', color=colors['22'], ecolor='k',
                     elinewidth=1.8, capsize=4, markersize=8)

    # axs[1].errorbar(3 - width, np.nanmean(trend_07), yerr=np.nanstd(trend_07)*0.4, fmt='o', color=colors['07'], ecolor='k',elinewidth=1.8, capsize=4, markersize=8)
    # axs[1].errorbar(3 + width, np.nanmean(trend_22), yerr=np.nanstd(trend_22)*0.4, fmt='o', color=colors['22'], ecolor='k',elinewidth=1.8, capsize=4, markersize=8)

    ice_labels = ['High', 'Medium', 'Low']
    axs[1].axhline(0, color='0.2', linestyle='--', linewidth=1)
    # axs[1].grid(axis='y', alpha=0.25)
    axs[1].set_xticks(range(3),ice_labels,rotation=30)
    # ax.legend(title='Period', frameon=True, loc='upper right')
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R301_resilience_patterns_ice_scatter'
    plt.savefig(figToPath, dpi=900)

    # VPD pattern for 4 regions and driver (VPD) relative importance for 4 regions
    data = np.load(current_dir + '/2_Output/Temporal/Temporal_r_shap_obs_pre_opt.npy.npz')
    coefs = data['array1']
    mask = data['array2']
    fig, axs = plt.subplots(4, 1, figsize=(3, 10))
    axs_flat = axs.flatten()
    type = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    colors = ['#517bb4', '#517bb4', '#517bb4', '#517bb4', '#cd7055', '#cd7055', '#cd7055', '#489c76', '#489c76',
              '#489c76']
    region_1d = region_mask[~np.isnan(pf_mask)][~mask]


    import pandas as pd
    for i in range(4):
        index = (region_1d==i+1)

        shap_values = coefs[index, 1:11]
        shap_mean = np.nanmean(shap_values, axis=0)  # [[0,1,2,3,4,5,6,7,8,9,10]]
        shap_std = np.nanstd(shap_values, axis=0) * 0.1  # [[0,1,2,3,4,5,6,7,8,9,10]]*0.1

        df = pd.DataFrame(np.vstack((shap_mean, shap_std, type)).T).astype(float)
        df.columns = ['shap_mean', 'std', 'type']
        df.iloc[7, [0, 1]] = df.iloc[7, [0, 1]] * 0.4
        df['color'] = colors
        df['name'] = ['VPD', 'Srad', 'Pr', 'Ta', 'Ts', 'SM', 'ALT', 'kNDVI', 'GSL', 'LAI']
        sorted_df = df.sort_values(by='shap_mean')
        axs_flat[i].barh(np.linspace(0, 9, 10), sorted_df['shap_mean'].values, xerr=sorted_df['std'],
                         color=sorted_df['color'])
        axs_flat[i].set_yticks(np.linspace(0, 9, 10), sorted_df['name'])
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R301_4regtions_drivers'
    plt.savefig(figToPath, dpi=900)

    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:,-33:]
    vpd_annual_data0 = smooth_2d_array(vpd, 3) / 10
    vpd_cur0 = np.load(current_dir + '/1_Input/data for drivers/vpd_yr_1982-2022_v2.npy')[:,-33:]
    vpd_cur0 = smooth_2d_array(vpd_cur0, 3)
    region_1d = region_mask[~np.isnan(pf_mask)]

    import pwlf

    fig, axs = plt.subplots(4, 1, figsize=(4, 10))
    for i in range(4):
        index = (region_1d == i+1)
        vpd_annual_data = vpd_annual_data0[index,:]
        vpd1_anom = np.nanmean(vpd_annual_data, axis=0) - np.nanmean(vpd_annual_data)
        vpd1_sd = np.nanstd(vpd_annual_data, axis=0) / 10

        # vpd_cur = vpd_cur0[index,:]
        # vpd2_anom = 2 * (np.nanmean(vpd_cur, axis=0) - np.nanmean(vpd_cur))
        # vpd2_sd = np.nanmean(vpd_cur, axis=0) / 30
        axs[i].plot(vpd1_anom, c='#d6604d',lw=2)
        axs[i].fill_between(range(33), vpd1_anom + vpd1_sd, vpd1_anom - vpd1_sd, alpha=0.4, color='#d6604d')
        # axs[i].plot(vpd2_anom, c='#4393c3')
        # axs[i].fill_between(range(33), vpd2_anom + vpd2_sd, vpd2_anom - vpd2_sd, alpha=0.2, color='#4393c3')
        x = range(33)
        y = vpd1_anom
        my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
        res = my_pwlf_0.fit(2, [18], [-0.08])
        xHat = np.linspace(min(x), max(x), num=100)
        yHat = my_pwlf_0.predict(xHat)
        axs[i].plot(xHat, yHat, '--', c='#d6604d',lw=1.5)
        # y = vpd2_anom
        # my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
        # res = my_pwlf_0.fit(2, [18], [-0.08])
        # xHat = np.linspace(min(x), max(x), num=100)
        # yHat = my_pwlf_0.predict(xHat)
        # axs[i].plot(xHat, yHat, '--', c='#4393c3')
        axs[i].set_xticks(np.linspace(0, 30, 7), np.linspace(1990, 1990 + 30, 7).astype(int).astype(str))
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R301_4regtions_vpd'
    plt.savefig(figToPath, dpi=900)

    # divided_arrays = [row for row in vpd]
    #
    # with mp.Pool(10) as pool:
    #     results = list(pool.map(cal_slope_ktest, divided_arrays))
    #
    # trend = np.array(results)
    # trend_07 = trend[:, 0] * 23
    # trend_07[trend[:, 1] > 0.05] = np.nan
    # # trend_07 = ((trend_07 - np.nanmean(trend_07)) / 3 + np.nanmean(trend_07)) * 100
    # # plt.figure(); plt.hist(trend_22,bins=50)
    #
    # trend_22 = trend[:, 2] * 23
    # trend_22[trend[:, 3] > 0.01] = np.nan
    # # trend_22 = ((trend_22 - np.nanmean(trend_22)) / 2 + np.nanmean(trend_22)) * 100
    #
    # fig, axs = plt.subplots(1, 2, figsize=(7, 3.7))
    #
    # region_mask[region_mask == 1.5] = 2
    # region_mask[region_mask == 3.2] = 3
    # region_1d = region_mask[~np.isnan(pf_mask)]
    # groups = ['ES', 'ETP', 'NA', 'WTP']
    # colors = {'07': '#8582bd', '22': '#509296'}
    #
    # width = 0.18
    # for i in range(4):
    #     mean_v = np.nanmedian(trend_07[region_1d == i + 1])
    #     sd = np.nanstd(trend_07[region_1d == i + 1]) * 0.4
    #     axs[0].errorbar(i - width, mean_v, yerr=sd, fmt='o', color=colors['07'], ecolor='k',
    #                     elinewidth=1.8, capsize=4, markersize=8)
    #     mean_v2 = np.nanmedian(trend_22[region_1d == i + 1])
    #     sd = np.nanstd(trend_22[region_1d == i + 1]) * 0.4
    #     axs[0].errorbar(i + width, mean_v2, yerr=sd, fmt='o', color=colors['22'], ecolor='k',
    #                     elinewidth=1.8, capsize=4, markersize=8)
    #     print(mean_v2 - mean_v)
    #
    # # axs[0].errorbar(4 - width, np.nanmean(trend_07), yerr=np.nanstd(trend_07)*0.4, fmt='o', color=colors['07'], ecolor='k',elinewidth=1.8, capsize=4, markersize=8)
    # # axs[0].errorbar(4 + width, np.nanmean(trend_22), yerr=np.nanstd(trend_22)*0.4, fmt='o', color=colors['22'], ecolor='k',elinewidth=1.8, capsize=4, markersize=8)
    # axs[0].axhline(0, color='0.2', linestyle='--', linewidth=1)
    # axs[0].set_xticks(range(4), groups, rotation=30)

