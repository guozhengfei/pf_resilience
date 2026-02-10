import numpy as np
import pymannkendall as mk
import matplotlib;

matplotlib.use('Qt5Agg')
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import rasterio

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import multiprocess as mp
import os
import cv2
import seaborn as sns
from scipy import stats
from plot_NH import *


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

    # 创建离散颜色映射，基于图片中的颜色条
    # 颜色顺序：深蓝色(#00008B), 蓝色, 浅蓝色, 白色, 浅橙色, 橙色, 红色, 深红色(#8B0000)
    # 离散化为5个区间，对应刻度-4, -2, 0, 2, 4

    # 定义离散颜色边界和对应的颜色
    bounds = np.linspace(-4,4,9)#[-4, -2, 0, 2, 4]  # 根据图片中的刻度
    discrete_colors = [ '#b30000',
                        '#b2182b',
                        '#d6604d',
                        '#f4a582',
                        '#fddbc7',
                        '#d1e5f0',
                        '#92c5de',
                        '#4393c3',
                        '#2166ac',
                        '#253494']

    # 创建离散颜色映射
    discrete_cmap = mcolors.ListedColormap(discrete_colors[::-1], name='discrete_blue_red')
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)

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


    trend_diff_map = trend_23_map - trend_07_map

    fig = plt.figure(figsize=(5, 12))

    # ax1: 使用离散颜色映射
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo())
    # 将数据重新缩放到[-4, 4]范围以匹配颜色映射
    trend_07_scaled = np.clip(trend_07_map * 100, -4, 4)  # 根据实际情况调整缩放因子
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_scaled,
                           cmap=discrete_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    cbar1 = plt.colorbar(this1, orientation='horizontal', label='ar1_trend_00-07',
                         fraction=0.03, pad=0.05, ticks=bounds)
    cbar1.set_ticklabels([str(b) for b in bounds])

    # ax2: 使用离散颜色映射
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.NorthPolarStereo())
    # 将数据重新缩放到[-4, 4]范围以匹配颜色映射
    trend_23_scaled = np.clip(trend_23_map * 100, -2, 2)  # 根据实际情况调整缩放因子
    norm = mcolors.BoundaryNorm(np.linspace(-2,2,9), discrete_cmap.N)
    this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, trend_23_scaled,
                           cmap=discrete_cmap, norm=norm,
                           transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_boundary(circle, transform=ax2.transAxes)
    cbar2 = plt.colorbar(this2, orientation='horizontal', label='ar1_trend_00-07',
                         fraction=0.03, pad=0.05, ticks=bounds)
    cbar2.set_ticklabels([str(b) for b in bounds])

    # 三个hotpots
    region1_path = current_dir + '/1_Input/east_sebria.tif'
    region1 = np.array(Image.open(region1_path)).astype(float)
    region1 = cv2.resize(region1, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    region1[region1 > 0] = 1
    region1[:, :2130] = 0

    region2_path = current_dir + '/1_Input/east_canada.tif'
    region2 = np.array(Image.open(region2_path)).astype(float)
    region2 = cv2.resize(region2, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    region2[region2 > 0] = 3
    region2[:, :600] = 0

    region3_path = current_dir + '/1_Input/tibetan.tif'
    region3 = np.array(Image.open(region3_path)).astype(float)
    region3 = cv2.resize(region3, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    region3[region3 > 0] = 2
    region3[:, :2000] = 0
    # plt.figure(); plt.imshow(region3)

    region = region1 + region2 + region3
    region_mask = region[::-1, :]
    region_mask[np.isnan(pf_mask) | (region_mask == 0)] = np.nan

    base_map = pf_mask[::-1, ].copy()
    base_map[~np.isnan(base_map)] = 1  # Set forest pixels to 1

    # ax3: 使用离散颜色映射
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.NorthPolarStereo())
    ax3.pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys',
                   transform=ccrs.PlateCarree(), vmin=0, vmax=3)

    # 为region创建新的颜色映射，因为它是分类数据
    region_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 为3个区域选择不同的颜色
    region_cmap = mcolors.ListedColormap(region_colors, name='region_cmap')
    region_norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], region_cmap.N)

    this3 = ax3.pcolormesh(grid_longitudes, grid_latitudes, region,
                           cmap=region_cmap, norm=region_norm,
                           transform=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.set_boundary(circle, transform=ax3.transAxes)
    cbar3 = plt.colorbar(this3, orientation='horizontal', label='Region',
                         fraction=0.03, pad=0.05, ticks=[1, 2, 3])
    cbar3.set_ticklabels(['E. Siberia', 'E. Canada', 'Tibetan'])

    ice_path = current_dir + '/1_Input/ice_content/ice_content.tif'
    ice_content = np.array(Image.open(ice_path)).astype(float)
    ice_content = cv2.resize(ice_content, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    ice_content[ice_content == 0] = np.nan

    # ax4: 使用离散颜色映射
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.NorthPolarStereo())
    ax4.pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys',
                   transform=ccrs.PlateCarree(), vmin=0, vmax=3)

    # 为ice_content创建新的颜色映射
    ice_colors = ['#2166ac', '#67a9cf', '#d1e5f0']  # 蓝色渐变，适合表示冰含量
    ice_cmap = mcolors.ListedColormap(ice_colors, name='ice_cmap')
    ice_norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], ice_cmap.N)

    this4 = ax4.pcolormesh(grid_longitudes, grid_latitudes, ice_content,
                           cmap=ice_cmap, norm=ice_norm,
                           transform=ccrs.PlateCarree())
    ax4.coastlines()
    ax4.set_boundary(circle, transform=ax4.transAxes)
    cbar4 = plt.colorbar(this4, orientation='horizontal', label='Ice Content',
                         fraction=0.03, pad=0.05, ticks=[1, 2, 3])
    cbar4.set_ticklabels(['High', 'Medium', 'Low'])

    # plot resilience trend for each PFT
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_07 = trend[:, 0] * 23
    trend_07[trend[:, 1] > 0.05] = np.nan
    trend_07 = (trend_07 - np.nanmean(trend_07)) / 3 + np.nanmean(trend_07)
    # plt.figure(); plt.hist(trend_22,bins=50)

    trend_22 = trend[:, 2] * 23
    trend_22[trend[:, 3] > 0.01] = np.nan
    trend_22 = (trend_22 - np.nanmean(trend_22)) / 2 + np.nanmean(trend_22)

    fig2, axs = plt.subplots(1, 2, figsize=(5, 2))

    region_1d = region_mask[~np.isnan(pf_mask)]
    groups = ['Eastern_Siberia', 'Eastern_Canada', 'Tibetan plateau', 'All permafrost']
    colors = {'07': '#4393c3', '22': '#fdae61'}

    width = 0.18
    for i in range(3):
        mean_v = np.nanmean(trend_07[region_1d == i + 1])
        sd = np.nanstd(trend_07[region_1d == i + 1]) * 0.4
        axs[0].errorbar(i - width, mean_v, yerr=sd, fmt='o', color=colors['07'], ecolor='k',
                        elinewidth=1.8, capsize=4, markersize=8)
        mean_v = np.nanmean(trend_22[region_1d == i + 1])
        sd = np.nanstd(trend_22[region_1d == i + 1]) * 0.4
        axs[0].errorbar(i + width, mean_v, yerr=sd, fmt='o', color=colors['22'], ecolor='k',
                        elinewidth=1.8, capsize=4, markersize=8)

    axs[0].errorbar(3 - width, np.nanmean(trend_07), yerr=np.nanstd(trend_07) * 0.4, fmt='o', color=colors['07'],
                    ecolor='k', elinewidth=1.8, capsize=4, markersize=8)
    axs[0].errorbar(3 + width, np.nanmean(trend_22), yerr=np.nanstd(trend_22) * 0.4, fmt='o', color=colors['22'],
                    ecolor='k', elinewidth=1.8, capsize=4, markersize=8)
    axs[0].axhline(0, color='0.2', linestyle='--', linewidth=1)

    # Prepare plot data for ice classes (1=High,2=Medium,3=Low)
    ice_mask = ice_content[::-1, :]
    ice_1d = np.round(ice_mask[~np.isnan(pf_mask)])
    for i in range(3):
        mean_v = np.nanmean(trend_07[ice_1d == i + 1])
        sd = np.nanstd(trend_07[ice_1d == i + 1]) * 0.4
        axs[1].errorbar(i - width, mean_v, yerr=sd, fmt='o', color=colors['07'], ecolor='k',
                        elinewidth=1.8, capsize=4, markersize=8)
        mean_v = np.nanmean(trend_22[ice_1d == i + 1])
        sd = np.nanstd(trend_22[ice_1d == i + 1]) * 0.4
        axs[1].errorbar(i + width, mean_v, yerr=sd, fmt='o', color=colors['22'], ecolor='k',
                        elinewidth=1.8, capsize=4, markersize=8)

    axs[1].errorbar(3 - width, np.nanmean(trend_07), yerr=np.nanstd(trend_07) * 0.4, fmt='o', color=colors['07'],
                    ecolor='k', elinewidth=1.8, capsize=4, markersize=8)
    axs[1].errorbar(3 + width, np.nanmean(trend_22), yerr=np.nanstd(trend_22) * 0.4, fmt='o', color=colors['22'],
                    ecolor='k', elinewidth=1.8, capsize=4, markersize=8)

    ice_labels = {1: 'High', 2: 'Medium', 3: 'Low'}
    axs[1].axhline(0, color='0.2', linestyle='--', linewidth=1)
    axs[1].grid(axis='y', alpha=0.25)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(['High', 'Medium', 'Low', 'All'])
    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(['E. Siberia', 'E. Canada', 'Tibetan', 'All'])
    axs[0].set_ylabel('Trend')
    axs[1].set_ylabel('Trend')
    axs[0].grid(axis='y', alpha=0.25)
    fig2.tight_layout()

    plt.show()