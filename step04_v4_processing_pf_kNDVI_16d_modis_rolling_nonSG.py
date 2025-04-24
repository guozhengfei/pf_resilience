import numpy as np
from PIL import Image
import rasterio
# import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Qt5Agg')
import multiprocess as mp
import os

def ar1_series_3yr(array):
    yrs = 3  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    bands_year = 23

    t = bands_year*yrs
    ar1 = pd.Series(array).rolling(t, center=True).apply(calc_ar1).values
    return ar1

def ar1_series_5yr(array):
    yrs = 5  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    bands_year = 23

    t = bands_year*yrs
    ar1 = pd.Series(array).rolling(t, min_periods=30,center=True).apply(calc_ar1).values
    return ar1

def ar1_series_4yr(array):
    yrs = 4  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    bands_year = 23

    t = bands_year*yrs
    ar1 = pd.Series(array).rolling(t, center=True).apply(calc_ar1).values
    return ar1

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir+'/1_Input/NDVI_pf_16d/'

    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1,]
    # plt.figure(); plt.imshow(pf_mask)
    # plt.figure(); plt.imshow(evi)
    # pf_mask_1d = pf_mask.flatten()
    evi_filenames = os.listdir(EVI_folder)
    evi_filenames = np.sort(evi_filenames)#[:529]

    EVI = []
    yr_num = 24
    bands_year = 23
    for name in evi_filenames:
        src = rasterio.open(EVI_folder + name)
        evi = src.read(1)[::3, ::3]
        evi[evi < 0] = 0
        evi_val = evi[~np.isnan(pf_mask)]
        evi_sq = evi_val**2
        kndvi = np.tanh(evi_sq)
        EVI.append(kndvi)
        print(name)
    EVI = np.array(EVI).T

    ser = EVI#EVI_sg.T
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
    del rm_offline, Evi_sea_rep

    divided_arrays = [row for row in res]
    with mp.Pool(240) as pool:
        results = list(pool.map(ar1_series_4yr, divided_arrays))
    ar1_res_5sg = np.array(results)

    # plt.figure(); plt.plot(np.linspace(2000,2023+22/23,552)[46:-46],np.nanmean(ar1_res_5sg,axis=0)[46:-46])
    output_file = current_dir+'/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_org_rolling.npy'
    np.save(output_file, ar1_res_5sg)

    # with mp.Pool(24) as pool:
    #     results = list(pool.map(ar1_series_4yr, divided_arrays))
    # ar1_res_7sg = np.array(results)
    #
    # output_file = r'D:\Projects\Project_pf\Data\ar1_4yr_16d_kndvi_modis.npy'
    # np.save(output_file, ar1_res_7sg)
    #
    # with mp.Pool(24) as pool:
    #     results = list(pool.map(ar1_series_3yr, divided_arrays))
    # ar1_res_3sg = np.array(results)
    #
    # output_file = r'D:\Projects\Project_pf\Data\ar1_3yr_16d_kndvi_modis.npy'
    # np.save(output_file, ar1_res_3sg)