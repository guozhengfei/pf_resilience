import numpy as np
import multiprocess as mp
import os
import cv2
import netCDF4 as nc
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

def ar1_series_4yr(array):
    yrs = 4  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    bands_year = 12

    t = bands_year*yrs
    ar1 = pd.Series(array).rolling(t, center=True).apply(calc_ar1).values
    return ar1

def find_files_with_string(root_folder, search_str):
    file_list = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if not file.startswith('.') and search_str in file:
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    root_folder = current_dir + '/1_Input/20240704_TRENDY-v11/'
    search_str = 'gpp'
    gpp_filenames = find_files_with_string(root_folder, search_str)
    for name in gpp_filenames:
        dataset = nc.Dataset(name)
        variable_name1 = 'gpp'
        data = dataset.variables[variable_name1][:]
        data = np.ma.getdata(data)
        data = data[-23*12:,::-1,:] # year 2000-2020
        months, rows, cols = np.shape(data)
        data_pf = data[:,int(np.round((90-83.7)/180*rows)):int(np.round((90-27.0)/180*rows)),:]

        pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
        pf_mask2 = rasterio.open(pf_mask_path2).read(1)[::20,::20].astype(float)
        pf_mask2[pf_mask2 <= 0] = np.nan
        pf_mask2[pf_mask2 > 12] = np.nan

        data_pf_rsz = []
        for i in range(data_pf.shape[0]):
            data_i = cv2.resize(data_pf[i, :, :], (pf_mask2.shape[1], pf_mask2.shape[0]), cv2.INTER_NEAREST)
            data_pf_rsz.append(data_i)
        data_pf_rsz = np.stack(data_pf_rsz, axis=2)
        data_pf_rsz[data_pf_rsz<0]=np.nan
        ser = data_pf_rsz[~np.isnan(pf_mask2),:]

        # calculate resilience
        yr_num = 23
        bands_year = 12
        EVI_yr = np.zeros_like(ser)

        for year in range(yr_num):
            st = year * bands_year
            ed = st + bands_year
            evi_year = np.nanmean(ser[:, st:ed], axis=1)
            evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
            EVI_yr[:, st:ed] = evi_year_rep
        rm_offline = ser - EVI_yr
        # plt.figure()
        # plt.plot(np.nanmean(rm_offline,axis=0))

        Evi_sea_rep = np.zeros_like(rm_offline)
        for yr in range(yr_num):
            start_index = (yr - 5) * bands_year
            end_index = (yr + 5) * bands_year
            if yr < 5: start_index = 0; end_index = 10 * bands_year
            if yr > (yr_num - 5): start_index = (yr_num - 10) * bands_year; end_index = yr_num * bands_year
            data_i = rm_offline[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)), axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
        res = rm_offline - Evi_sea_rep
        mask = (res[:,0]==0)
        res = res[~mask,:]
        res[np.isnan(res)] = 0
        # res = res[::3,:]
        # plt.figure(); plt.plot(np.nanmean(res, axis=0))
        divided_arrays = [row for row in res]

        with mp.Pool(6) as pool:
            results = list(pool.map(ar1_series_4yr, divided_arrays))
        ar1_res_5sg = np.array(results)
        # plt.figure();plt.plot(np.nanmean(ar1_res_5sg,axis=0))

        output = current_dir +'/2_Output/Trendy_resilience_v2/' + name.split('/')[-1].split('.')[0]+'.npy'
        np.save(output,ar1_res_5sg)
        print(output)

