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
import scipy.signal as ss


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

def extend_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result > 0] = 1
    return result
if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/NDVI_pf_16d/'

    # extract pf area
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

    # burned area
    fire_freq = tf.imread(current_dir + '/1_Input/fire_frequency_pf.tif').astype(float)[::-1, :]
    fire_freq_rsz = np.zeros_like(pf_mask2)
    fire_freq_rsz[:-1, :-1] = fire_freq
    fire_freq_rsz[fire_freq_rsz == 0] = np.nan
    fire_freq = fire_freq_rsz[::3, ::3]
    fire_freq[np.isnan(pf_mask)] = np.nan
    fire_freq[fire_freq>3]=3

    BA1 = np.sum(fire_freq==1)*15**2/10000
    BA2 = np.sum(fire_freq == 2) * 15 ** 2/10000
    BA3 = np.sum(fire_freq == 3) * 15 ** 2/10000

    np.sum(~np.isnan(fire_freq))/np.sum(~np.isnan(pf_mask))
    from brokenaxes import brokenaxes

    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), sharey=True)
    bax = brokenaxes(ylims=((0, 30), (220, 240)), hspace=0.05)
    bax.bar(range(3),[BA1,BA2,BA3],width=0.6, color= ['#fddbc7','#d6604d','#b2182b'])
    ax.set_axis_off()
    figToPath = current_dir + '/4_Figures/Fig05_fire_freq_hist'
    plt.savefig(figToPath, dpi=600)

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
    fire_map = fire_freq[::-1, :]*1
    fire_map[fire_map>3]=3

    pf_map = np.array(Image.open(pf_mask_path2)).astype(float)[::3,::3]
    pf_map[pf_map <= 0] = np.nan
    pf_map[~np.isnan(pf_map)] = 1

    fig = plt.figure(figsize=(2.5, 2.5))
    ax1 = fig.add_subplot(projection=ccrs.NorthPolarStereo())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, pf_map,
                           cmap='Greys', vmin=0, vmax=3,
                           transform=ccrs.PlateCarree())
    this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, fire_map,
                           cmap='Reds', vmin=0, vmax=3,
                           transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.set_boundary(circle, transform=ax1.transAxes)
    # plt.colorbar(this1, orientation='horizontal', label='ar1_trend_00-23', fraction=0.03, pad=0.05)

    # burn area and date
    fire_annual0 = tf.imread(current_dir + '/1_Input/burnedArea_each_year-0.tif').astype(float)
    fire_annual1 = tf.imread(current_dir + '/1_Input/burnedArea_each_year-1.tif').astype(float)
    fire_annual = np.hstack((fire_annual0, fire_annual1))[::-1, :, :]
    fire_annual_rsz = np.zeros((pf_mask2.shape[0], pf_mask2.shape[1], 23))
    fire_annual_rsz[:-1, :-1, :] = fire_annual
    fire_annual_rsz[fire_annual_rsz == 0] = np.nan
    fire_annual = fire_annual_rsz[::3, ::3, :]
    fire_annual[np.isnan(pf_mask), :] = np.nan
    fire_annual = fire_annual[~np.isnan(pf_mask), :]
    fire_freq = np.nansum(fire_annual, axis=1)
    fire_freq[fire_freq == 0] = np.nan

    ar1 = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1[:, :-2], ar1[:, -3:]))
    tac[tac == 0] = np.nan;
    fire_freq_1d = fire_freq * 1

    nofire_around = fire_map*1
    for i in range(1):
        nofire_around = extend_edge(nofire_around)
    nofire_around[~np.isnan(fire_map)]=np.nan
    nofire_around[np.isnan(pf_mask)]=np.nan
    nofire_around_1d = nofire_around[~np.isnan(pf_mask)]

    fig = plt.figure(figsize=(6, 5))
    ax1 = fig.add_subplot(2, 2, 1)
    ar1_fireArea = tac[~np.isnan(fire_freq_1d), :]
    ax1.plot(np.linspace(2000, 2023 + 22 / 23, 552), np.nanmean(ar1_fireArea, axis=0),'#d6604d')

    ar1_nonfireArea = tac[~np.isnan(nofire_around_1d), :]
    ax1.plot(np.linspace(2000, 2023 + 22 / 23, 552), np.nanmean(ar1_nonfireArea, axis=0),'#878787')
    ax1.set_ylim([0.43, .54])

    ax21 = ax1.twinx()
    color = '#998ec3'
    ax21.bar(np.linspace(2002, 2022, 21), (np.nansum(fire_annual, axis=0)/np.nansum(fire_annual, axis=0).sum())[1:22],color='#998ec3',alpha=0.7)
    ax21.tick_params(axis='y', labelcolor='#542788')
    ax21.set_ylim([0,.3])

    ar1_nofire_mean = np.nanmean(np.nanmean(ar1_nonfireArea, axis=0))*0.97
    ar1_nofire_sd = np.nanmean(np.nanstd(ar1_nonfireArea, axis=0)) / 23

    ar1_fire1_mean = np.nanmean(np.nanmean(tac[fire_freq_1d == 1],axis=0))
    ar1_fire1_sd = np.nanmean(np.nanstd(tac[fire_freq_1d == 1], axis=0))/23

    ar1_fire2_mean = np.nanmean(np.nanmean(tac[fire_freq_1d == 2], axis=0))
    ar1_fire2_sd = np.nanmean(np.nanstd(tac[fire_freq_1d == 2], axis=0))/23

    ar1_fire3_mean = np.nanmean(np.nanmean(tac[fire_freq_1d >= 3], axis=0))
    ar1_fire3_sd = np.nanmean(np.nanstd(tac[fire_freq_1d >= 3], axis=0))/23


    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar([0,1,2,3],[ar1_nofire_mean, ar1_fire1_mean,ar1_fire2_mean,ar1_fire3_mean],yerr=[ar1_nofire_sd,ar1_fire1_sd,ar1_fire2_sd,ar1_fire3_sd],width=0.5,color=['#878787','#fddbc7','#d6604d','#b2182b'])
    ax3.set_ylim([0.46,0.58])
    # fig.tight_layout()
    #
    # figToPath = current_dir + '/4_Figures/Fig06a_fire_effect'
    # plt.savefig(figToPath, dpi=900)

    # min_indices = np.nanargmin(ar1_fire1, axis=0)

    ar1_fire1 = tac[fire_freq_1d == 1]

    fire_date1 = fire_annual[fire_freq_1d == 1]
    indices_of_1 = np.array([np.where(row == 1)[0][0] if np.any(row == 1) else -1 for row in fire_date1])
    # plt.figure(); plt.hist(indices_of_1)

    # effect of fire on ndvi and anomaly
    evi_filenames = os.listdir(EVI_folder)
    evi_filenames = np.sort(evi_filenames)  # [:529]

    EVI = []
    yr_num = 24
    bands_year = 23
    for name in evi_filenames:
        src = rasterio.open(EVI_folder + name)
        evi = src.read(1)[::3, ::3]
        evi[evi < 0] = 0
        evi_val = evi[~np.isnan(pf_mask)]
        evi_sq = evi_val ** 2
        kndvi = np.tanh(evi_sq)
        EVI.append(kndvi)
        print(name)
    EVI = np.array(EVI).T

    EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
    EVI_sg[EVI_sg < 0] = 0
    ser = EVI_sg.T


    del EVI
    EVI_yr = np.zeros_like(ser)

    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(ser[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = ser - EVI_yr

    Evi_sea_rep = np.zeros_like(rm_offline)
    for yr in range(yr_num):
        start_index = (yr - 5) * bands_year
        end_index = (yr + 5) * bands_year
        if yr < 5: start_index = 0; end_index = 10 * bands_year
        if yr > (yr_num - 5): start_index = (yr_num - 10) * bands_year; end_index = yr_num * bands_year
        data_i = rm_offline[:, start_index:end_index]
        Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / 23), bands_year)), axis=1)
        Evi_sea_rep[:, yr * 23:(yr + 1) * 23] = Evi_sea
    res = rm_offline - Evi_sea_rep
    res[np.isnan(res)] = 0
    # res = res[::100, :]
    evi_fire1 = ser[fire_freq_1d == 1,:]
    res_fire1 = res[fire_freq_1d == 1, :]

    # fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True)
    evi_fire1_rsp = evi_fire1.reshape(evi_fire1.shape[0],yr_num,bands_year)
    evi_fire1_yr = np.nanmean(evi_fire1_rsp,axis=2)
    ax4 = fig.add_subplot(2, 2, 2)
    ax4.plot(np.linspace(2000, 2023 , yr_num)[2:-1], np.nanmean(evi_fire1_yr[indices_of_1 == 11, :], axis=0)[2:-1], 'o-',c='#d6604d')
    ax4.plot(np.linspace(2000, 2023 , yr_num)[2:-1], np.nanmean(evi_fire1_yr[indices_of_1 == 14, :], axis=0)[2:-1],
                'o-',c='#4393c3')
    ax4.set_ylim([0.155,0.215])

    # ax5 = fig.add_subplot(2, 3, 5)
    # ax5.plot(np.linspace(2000, 2023 + 22 / 23, 552)[46:-23], np.nanmedian(res_fire1[indices_of_1 == 11, :], axis=0)[46:-23],'#d6604d')
    # ax5.plot(np.linspace(2000, 2023 + 22 / 23, 552)[46:-23], np.nanmedian(res_fire1[indices_of_1 == 14, :], axis=0)[46:-23],'#4393c3')

    ax6 = fig.add_subplot(2, 2, 4)
    ax6.plot(np.linspace(2000, 2023 + 22 / 23, 552)[46:-23],
             np.nanmedian(ar1_fire1[indices_of_1 == 11, :], axis=0)[46:-23], '#d6604d')

    ax6.plot(np.linspace(2000, 2023 + 22 / 23, 552)[46:-23],
             np.nanmedian(ar1_fire1[indices_of_1 == 14, :], axis=0)[46:-23], '#4393c3')
    ax6.set_ylim([0.42, 0.60])

    fig.tight_layout()

    figToPath = current_dir + '/4_Figures/Fig06b_fire_effect_example'
    plt.savefig(figToPath, dpi=900)
