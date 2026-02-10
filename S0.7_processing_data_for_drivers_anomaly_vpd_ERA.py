import numpy as np
import tifffile as tf
import matplotlib;matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
from PIL import Image
import cv2
warnings.filterwarnings("ignore")


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

    # plt.figure(); plt.imshow(pf_mask2)
    pf_mask_1d = pf_mask.flatten()

    EVI = []
    yr_num = 41
    bands_year = 12
    Data_folder = current_dir+ '/1_Input/data for drivers/'
    T01 = tf.imread(Data_folder + 'temperature_2m_1982_2022-0000000000-0000000000.tif')
    T02 = tf.imread(Data_folder + 'temperature_2m_1982_2022-0000000000-0000001280.tif')
    T03 = tf.imread(Data_folder + 'temperature_2m_1982_2022-0000000000-0000002560.tif')
    T04 = tf.imread(Data_folder + 'temperature_2m_1982_2022-0000000000-0000003840.tif')
    T = np.concatenate((T01, T02, T03, T04), axis=1)[::3, ::3, ]
    T = T[:, :-1, :]
    T = T[::-1, :, :].astype(float)-273.15


    Td01 = tf.imread(Data_folder + 'dewpoint_temperature_2m_1982_2022-0000000000-0000000000.tif')
    Td02 = tf.imread(Data_folder + 'dewpoint_temperature_2m_1982_2022-0000000000-0000001280.tif')
    Td03 = tf.imread(Data_folder + 'dewpoint_temperature_2m_1982_2022-0000000000-0000002560.tif')
    Td04 = tf.imread(Data_folder + 'dewpoint_temperature_2m_1982_2022-0000000000-0000003840.tif')
    Td = np.concatenate((Td01, Td02, Td03, Td04), axis=1)[::3, ::3, ]
    Td = Td[:, :-1, :]
    Td = Td[::-1, :, :].astype(float)-273.15
    # data_reshaped = Td.reshape(Td.shape[0], Td.shape[1], 41, 12)
    # Td_yearly = np.nanmean(data_reshaped, axis=3)
    #
    # T_diff = T-Td
    # data_reshaped = T_diff.reshape(T_diff.shape[0], T_diff.shape[1], 41, 12)
    # Ta_yearly = np.nanmean(data_reshaped, axis=3)
    #
    # plt.figure(); plt.plot(np.nanmean(np.nanmean(Ta_yearly,axis=0),axis=0))

    ea = 0.61078 * np.exp(17.27 * Td / (Td + 237.3))
    data_reshaped = ea.reshape(ea.shape[0], ea.shape[1], 41, 12)
    ea_yearly = np.nanmean(data_reshaped[:,:,:,4:9], axis=3)

    # plt.figure(); plt.plot(np.nanmean(np.nanmean(ea_yearly,axis=0),axis=0))

    ea_star = 0.61078 * np.exp(17.27 * T / (T + 237.3))
    data_reshaped = ea_star.reshape(ea_star.shape[0], ea_star.shape[1], 41, 12)
    ea_star_yearly = np.nanmean(data_reshaped[:,:,:,4:9], axis=3)
    # ea = 2.1718e10 * np.exp(-4157/ (Td - 33.91));  # [Pa] (Henderson-Sellers, 1984)
    # ea_star = 2.1718e10 * np.exp(-4157 / (T - 33.91));
    vpd =  ea_star_yearly - ea_yearly
    pf_mask_rsz = cv2.resize(pf_mask, (vpd.shape[1], vpd.shape[0]))
    vpd_pf = vpd[~np.isnan(pf_mask_rsz),:]

    # plt.figure(); plt.plot(np.nanmean(vpd_pf,axis=0))
    EVI0 = np.reshape(vpd,(vpd.shape[0]*vpd.shape[1],vpd.shape[2]))
    EVI = EVI0#[~np.isnan(pf_mask2),:]
    ser = EVI
    rm_offline = ser# - EVI_yr

    # plt.figure();plt.plot(np.nanmean(rm_offline, axis=0))

    Evi_sea = []
    for month in range(bands_year):
        month_index = (np.linspace(0, rm_offline.shape[1], yr_num + 1)[0:-1] + month).astype(int)
        evi_month_mean = np.nanmean(rm_offline[:, month_index], axis=1)
        Evi_sea.append(evi_month_mean)

    Evi_sea = np.array(Evi_sea).T

    Evi_sea_rep = np.tile(Evi_sea, (1, yr_num))
    res = rm_offline - Evi_sea_rep
    # plt.figure();plt.plot(np.nanmean(res, axis=0))

    srad = res
    srad_annual_mean = []
    srad_annual_cv = []
    for i in range(yr_num):
        srad_mean_i = np.nanmean(srad[:, i * bands_year:(i + 1) * bands_year], axis=1)
        srad_cv_i = np.nanstd(srad[:, i * bands_year:(i + 1) * bands_year], axis=1)
        # srad_cv_i = srad_std_i / srad_mean_i
        srad_annual_mean.append(srad_mean_i)
        srad_annual_cv.append(srad_cv_i)
    srad_annual_mean = np.stack(srad_annual_mean, axis=1)
    srad_annual_cv = np.stack(srad_annual_cv, axis=1)

    fig, axs = plt.subplots(3, figsize=(6, 9))
    axs[0].plot(np.linspace(1982, 2023 - 1 / 12, 492), np.nanmean(res, axis=0))
    axs[1].plot(np.linspace(1982, 2022, 41), np.nanmean(srad_annual_mean, axis=0))
    axs[2].plot(np.linspace(1982, 2022, 41), np.nanmean(srad_annual_cv, axis=0))
    axs[0].set_title('vpd_ear5l_anomaly')