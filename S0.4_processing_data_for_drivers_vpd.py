import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
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
    # pf_mask2[pf_mask2 != pf_mask3] = np.nan
    # pf_mask2[pf_mask2 <= 0] = np.nan
    # pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]
    # plt.figure(); plt.imshow(pf_mask)
    pf_mask_1d = pf_mask.flatten()

    EVI = []
    yr_num = 41
    bands_year = 23
    Data_folder = current_dir + '/1_Input/data for drivers/'
    pr01 = tf.imread(Data_folder + 'vpd_1982_2022-0000000000-0000000000.tif')
    pr02 = tf.imread(Data_folder + 'vpd_1982_2022-0000000000-0000002304.tif')
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

    # yearly data
    EVI_rsp = EVI.reshape(EVI.shape[0], 41, 23)
    EVI_yearly = np.nanmean(EVI_rsp, axis=2)

    vpd_mean = np.nanmean(EVI_yearly,axis=0)
    plt.figure(); plt.plot(vpd_mean)

