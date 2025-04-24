import numpy as np
from PIL import Image
import scipy.signal as ss
import multiprocess as mp
import os
import rasterio



# 修改函数：计算滑动窗口的方差，保留线性回归去趋势
def ar1_series_4yr(array):
    yrs = 4  # 3,5,7
    import pandas as pd
    import numpy.ma as ma
    from sklearn.linear_model import LinearRegression

    # 定义计算方差的函数
    def calc_variance(x):
        return ma.var(ma.masked_invalid(x))  # 计算窗口内的方差，忽略无效值

    X = array[:, :-1]
    y = array[:, -1]
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_rmv = regressor.coef_[1] * X[:, 1] + regressor.coef_[2] * X[:, 2] + regressor.coef_[3] * X[:, 3]
    y_final = y - y_rmv  # 去趋势后的残差
    bands_year = 23
    t = bands_year * yrs
    variance = pd.Series(y_final).rolling(t, center=True).apply(calc_variance).values
    return variance


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/NDVI_pf_16d/'

    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]
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
    res_ndvi = rm_offline - Evi_sea_rep
    nan_frac_ndvi = np.sum(np.isnan(res_ndvi), axis=1) / res_ndvi.shape[1]

    res_pr = np.load(current_dir + '/1_Input/data for drivers/pr_month_anomaly.npy')[:, 417:]
    res_pr = np.hstack((res_pr, res_pr[:, 417:417 + 26] * 1.01))
    nan_frac_pr = np.sum(np.isnan(res_pr), axis=1) / res_pr.shape[1]

    res_srad = np.load(current_dir + '/1_Input/data for drivers/srad_month_anomaly.npy')[:, 417:]
    res_srad = np.hstack((res_srad, res_srad[:, 417:417 + 26] * 1.01))
    nan_frac_srad = np.sum(np.isnan(res_srad), axis=1) / res_srad.shape[1]

    res_tmmx = np.load(current_dir + '/1_Input/data for drivers/tmmx_month_anomaly.npy')[:, 417:]
    res_tmmx = np.hstack((res_tmmx, res_tmmx[:, 417:417 + 26] * 1.01))
    nan_frac_tmmx = np.sum(np.isnan(res_tmmx), axis=1) / res_tmmx.shape[1]
    mask = (nan_frac_ndvi > 0.75) | (nan_frac_pr > 0.3) | (nan_frac_srad > 0.3) | (nan_frac_tmmx > 0.3)

    variables_all = np.stack((res_ndvi[:, :-1], res_tmmx[:, 1:], res_srad[:, 1:], res_pr[:, 1:], res_ndvi[:, 1:]),
                             axis=2)

    variables_all = variables_all[~mask, :, :]
    variables_all[np.isnan(variables_all)] = 0
    del rm_offline, Evi_sea_rep
    divided_arrays = [row for row in variables_all]

    with mp.Pool(120) as pool:
        results = list(pool.map(ar1_series_4yr, divided_arrays))
    ar1_res_5sg = np.array(results)  # 现在是方差结果

    Result = np.empty_like(EVI_yr[:, :-1])
    Result[~mask, :] = ar1_res_5sg
    output_file = current_dir + '/2_Output/spatial_resilience/variance_5yr_kndvi_modis_sg_rolling.npy'  # 修改文件名以反映方差
    np.save(output_file, Result)