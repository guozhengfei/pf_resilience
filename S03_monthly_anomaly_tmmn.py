import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
matplotlib.use('Qt5Agg')
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")
import cv2
from scipy.interpolate import interp1d

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
    # plt.figure(); plt.imshow(pf_mask)
    pf_mask_1d = pf_mask.flatten()

    EVI = []
    yr_num = 41
    bands_year = 23
    Data_folder = current_dir + '/1_Input/data for drivers/'
    pr01 = tf.imread(Data_folder + 'tmmn_1982_2022-0000000000-0000000000.tif')
    pr02 = tf.imread(Data_folder + 'tmmn_1982_2022-0000000000-0000002304.tif')
    pr = np.concatenate((pr01, pr02), axis=1)
    # plt.figure(); plt.imshow(pr_rsz[:,:,6])
    pr = pr.astype(float)
    pr_rsz = []
    for i in range(pr.shape[2]):
        pr_i = cv2.resize(pr[:,:,i], (pf_mask.shape[1],pf_mask.shape[0]), cv2.INTER_NEAREST)
        pr_rsz.append(pr_i)
    pr_rsz = np.stack(pr_rsz, axis=2)
    # pr_rsz[pr_rsz == 0] = np.nan

    EVI0 = np.reshape(pr_rsz, (pr_rsz.shape[0] * pr_rsz.shape[1], pr_rsz.shape[2]))
    EVI1 = EVI0[~np.isnan(pf_mask_1d), :]

    interpolated_array = np.empty((EVI1.shape[0], 41*23))
    x = np.arange(EVI1.shape[1])
    x_new = np.linspace(0, EVI1.shape[1] - 1, 41*23)
    f = interp1d(x, EVI1, kind='linear', axis=1)
    interpolated_values = f(x_new)

    EVI1 = interpolated_values
    EVI1[EVI1 == 0] = np.nan

    frac_QA = np.sum(~np.isnan(EVI1), axis=1) / EVI1.shape[1]
    frac_QA[frac_QA < 0.7] = np.nan
    mask = np.isnan(frac_QA)
    EVI1[mask, :] = np.nan
    EVI = EVI1[~mask, :]

    #calculate monthly anomaly
    # calculate monthly anomaly
    EVI_yr = np.zeros_like(EVI)
    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(EVI[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = EVI - EVI_yr

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

    EVI_res = EVI1
    EVI_res[~mask, :] = res

    ###
    plt.figure();
    plt.plot(np.nanmean(EVI_res, axis=0))
    np.save(current_dir + '/1_Input/data for drivers/tmmn_month_anomaly.npy', EVI_res)
