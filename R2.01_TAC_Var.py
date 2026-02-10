import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib import cm, pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
from plot_NH import *
import os
from PIL import Image
import rasterio
import cv2
import copy

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr

def resilience_temporal_plot(ax, tac, band_years=23):
    year_num = int(tac.shape[1]/band_years)
    tac_yr = tac.reshape(tac.shape[0], year_num, band_years)
    tac_yr = np.nanmean(tac_yr, axis=2)[:, 2:-2]
    tac_yr_rpt = np.tile(tac_yr[:, 0], (tac_yr.shape[1], 1)).T
    tac2 = tac_yr #- tac_yr_rpt

    # all pixel trend of ar1
    time_label_modis = np.linspace(2002, 2021, 20)
    se = np.nanstd(tac2, axis=0) / tac2.shape[0]**0.5*20
    mean_val_modis = np.nanmean(tac2, axis=0)
    ax.plot(time_label_modis, mean_val_modis, color='k', lw=3)
    ax.fill_between(time_label_modis, mean_val_modis - se, mean_val_modis + se, color='k', alpha=0.2, label='Standard Deviation')
    
    # 添加两段线性拟合线
    # 2002-2008 (蓝色)
    mask_2002_2008 = (time_label_modis >= 2002) & (time_label_modis <= 2008)
    x_2002_2008 = time_label_modis[mask_2002_2008]
    y_2002_2008 = mean_val_modis[mask_2002_2008]
    z_2002_2008 = np.polyfit(x_2002_2008, y_2002_2008, 1)
    p_2002_2008 = np.poly1d(z_2002_2008)
    ax.plot(x_2002_2008, p_2002_2008(x_2002_2008), color='C0', lw=2.5, linestyle='--', label='2002-2008 trend')

    # 2009-2021 (红色)
    mask_2009_2021 = (time_label_modis >= 2007) & (time_label_modis <= 2021)
    x_2009_2021 = time_label_modis[mask_2009_2021]
    y_2009_2021 = mean_val_modis[mask_2009_2021]
    z_2009_2021 = np.polyfit(x_2009_2021, y_2009_2021, 1)
    p_2009_2021 = np.poly1d(z_2009_2021)
    ax.plot(x_2009_2021[1:], p_2009_2021(x_2009_2021[1:]), color='C3', lw=2.5, linestyle='--', label='2009-2021 trend')

    ax.axvline(x=2008, color='gray', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Year')
    # ax.legend(fontsize=10)

def resilience_spatial_plot(ax1, ax2, trend, scale=23,k_size=int(3)):
    trend_07 = trend[:, 0] * scale
    trend_07[trend[:, 1] > 0.05] = np.nan

    trend_22 = trend[:, 2] * scale
    trend_22[trend[:, 3] > 0.01] = np.nan
    np.sum(trend_22[~np.isnan(trend_22)] > 0) / trend_22[~np.isnan(trend_22)].shape[0]

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
    trend_07_map[~np.isnan(trend_07_map)] = trend[:, 0] * scale
    trend_07_map = trend_07_map[::-1, :]
    trend_07_map[np.isnan(trend_07_map)] = 0
    kernel = np.ones((k_size, k_size), np.float32) / k_size**2
    trend_07_map = cv2.filter2D(trend_07_map, -1, kernel)
    trend_07_map[trend_07_map<np.nanpercentile(trend_07, 1)]=0
    trend_07_map[trend_07_map==0] = np.nan

    trend_23_map = pf_mask * 1
    trend_23_map[~np.isnan(trend_23_map)] = trend[:, 2] * scale
    trend_23_map = trend_23_map[::-1, :]
    trend_23_map[np.isnan(trend_23_map)] = 0
    trend_23_map = cv2.filter2D(trend_23_map, -1, kernel)
    trend_23_map[trend_23_map==0] = np.nan
    # trend_07_map = np.ma.masked_where(~np.isnan(pf_mask[::-1,:]) & (trend_07_map[::-1,:]==0), trend_07_map)
    # cmap = copy.copy(matplotlib.colormaps['RdBu_r'])
    # cmap.set_bad('grey', 0.5)

    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map,
                           cmap='RdBu_r', vmin=np.nanpercentile(trend_07,80)*-1, vmax=np.nanpercentile(trend_07,80),
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='Trend: 00-08', fraction=0.03, pad=0.05)

    this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, trend_23_map,
                           cmap='RdBu_r', vmin=np.nanpercentile(trend_22,80)*-1, vmax=np.nanpercentile(trend_22,80),
                           transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_boundary(circle, transform=ax2.transAxes)
    plt.colorbar(this2, orientation='horizontal', label='Trend: 09-23', fraction=0.03, pad=0.05)
    return [trend_07_map,trend_23_map]

if __name__ == '__main__':
    # load ar1 data: 5years window
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')*10**2
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo())
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.NorthPolarStereo())
    tac_all = resilience_spatial_plot(ax1, ax2, trend, scale=23, k_size=int(3))

    trend2 = np.load(current_dir + '/2_Output/spatial_resilience/VR_trend_variance_5yr_kndvi_modis_sg_rolling.npy')*10**4
    np.sum(trend2[:,2]<0)/196518

    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.NorthPolarStereo())
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.NorthPolarStereo())
    var_all = resilience_spatial_plot(ax3, ax4, trend2, scale=23, k_size=int(3))
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R0201_TAC_var_all_spatial'
    plt.savefig(figToPath, dpi=900)

    ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:, -3:]))
    tac[tac == 0] = np.nan;
    fig, axs = plt.subplots(2,1,figsize=(3, 5))
    resilience_temporal_plot(axs[0], tac, band_years=23)

    # Var
    ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/variance_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:, -3:]))
    tac[tac == 0] = np.nan;
    resilience_temporal_plot(axs[1], tac, band_years=23)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R0201_all_temporal'
    plt.savefig(figToPath, dpi=900)
    ###############################
    # #consistent of TAC and Var
    ###############################
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
    # mask1 = trend[:,0]*trend2[:,0]>0
    mask1 = tac_all[0]*var_all[0]>0 # consistent mask
    tac_all[0][~mask1]=np.nan
    var_all[0][~mask1]=np.nan
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, tac_all[0],
                           cmap='RdBu_r', vmin=-2.26, vmax=2.26,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    plt.colorbar(this1, orientation='horizontal', label='Trend: 00-08', fraction=0.03, pad=0.05)
    196518 - np.sum(~mask1)

    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.NorthPolarStereo())
    this3 = ax3.pcolormesh(grid_longitudes, grid_latitudes, var_all[0],
                           cmap='RdBu_r', vmin=-0.489,
                           vmax=0.489,
                           transform=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.set_boundary(circle, transform=ax3.transAxes)
    plt.colorbar(this3, orientation='horizontal', label='Trend: 00-08', fraction=0.03, pad=0.05)

    mask2 = tac_all[1]*var_all[1]>0
    tac_all[1][~mask2] = np.nan
    var_all[1][~mask2] = np.nan

    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.NorthPolarStereo())
    this2 = ax2.pcolormesh(grid_longitudes, grid_latitudes, tac_all[1],
                           cmap='RdBu_r', vmin=-1.27,
                           vmax=1.27,
                           transform=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.set_boundary(circle, transform=ax2.transAxes)
    plt.colorbar(this2, orientation='horizontal', label='Trend: 00-08', fraction=0.03, pad=0.05)

    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.NorthPolarStereo())
    this4 = ax4.pcolormesh(grid_longitudes, grid_latitudes, var_all[1],
                           cmap='RdBu_r', vmin=-0.558,
                           vmax=0.558,
                           transform=ccrs.PlateCarree())
    ax4.coastlines()
    ax4.set_boundary(circle, transform=ax4.transAxes)
    plt.colorbar(this4, orientation='horizontal', label='Trend: 00-08', fraction=0.03, pad=0.05)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R0201_TAC_var_consistent_spatial'
    plt.savefig(figToPath, dpi=900)

    ##temporal
    mask1 = trend[:,0]*trend2[:,0]>0
    mask2 = trend[:, 2] * trend2[:, 2] > 0

    mask3 = (~mask1) | (~mask2)
    ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:, -3:]))
    tac[tac == 0] = np.nan;
    tac = tac[mask3, :]
    fig, axs = plt.subplots(2, 1, figsize=(3, 5))
    resilience_temporal_plot(axs[0], tac, band_years=23)


    # Var
    ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/variance_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:, -3:]))
    tac[tac == 0] = np.nan;
    tac = tac[mask3, :]
    resilience_temporal_plot(axs[1], tac, band_years=23)
    fig.tight_layout()
    figToPath = current_dir + '/4_Figures/R0201_consistent_temporal'
    plt.savefig(figToPath, dpi=900)