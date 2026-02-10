import netCDF4 as nc
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
import os
import cv2
matplotlib.use('Qt5Agg')
from datetime import datetime, timedelta


def moving_sd(array):
    bands_yr = 12
    yr_num = int(array.shape[1]/bands_yr)
    std_yr = []
    for i in range(yr_num):
        std_i = np.nanstd(array[:,i*bands_yr:(i+1)*bands_yr],axis=1)#/np.nanmean(array[:,i*bands_yr:(i+3)*bands_yr],axis=1)
        std_yr.append(std_i)
    return np.array(std_yr)
def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr
def convert_float_array_to_dates(float_array):
    base_date = datetime(1858, 11, 17)
    dates = [(base_date + timedelta(days=float_val)).strftime('%Y-%m') for float_val in float_array]
    return dates

def fill_nan_with_climatology(EVI, bands_year=12):
    num_years = EVI.shape[1] // bands_year
    climatology = np.zeros((EVI.shape[0], EVI.shape[1]))

    for year in range(num_years):
        if year < 2:
            start = 0
            end = 5 * bands_year
        elif year > num_years - 3:
            start = (num_years - 5) * bands_year
            end = num_years * bands_year
        else:
            start = (year - 2) * bands_year
            end = (year + 3) * bands_year

        climatology[:, year * bands_year:(year + 1) * bands_year] = np.nanmean(
            EVI[:, start:end].reshape(EVI.shape[0], 5, bands_year), axis=1
        )

    EVI_filled = np.where(np.isnan(EVI), climatology, EVI)

    return EVI_filled

# Open the NetCDF file
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
filename = current_dir+'/1_Input/VODCA2GPP_v1.nc'

pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
pf_mask2 = tf.imread(pf_mask_path2)[::10, ::10].astype(float)
pf_mask2[pf_mask2 <= 0] = np.nan
pf_mask2[pf_mask2 > 12] = np.nan


dataset = nc.Dataset(filename)
# Read data from a specific variable
variable_name = 'GPP'
data = dataset.variables[variable_name][:]
data = np.ma.getdata(data) # 1988-01-21 to 2020-06-30
data2 = data[:,::-1,:]
rows = 720
data_pf = data2[:, int(np.round((90 - 83.7) / 180 * rows)):int(np.round((90 - 27.0) / 180 * rows)), :]
data_pf[data_pf<0]=0

time = dataset.variables['time'][:]
time = np.ma.getdata(time).astype(float)
dates = convert_float_array_to_dates(time)
dataset.close()

dates = np.array(dates)
dates_uniq = np.sort(list(set(dates)))

gpp_pf = np.zeros_like(data_pf[:389,:,:])
i = 0
for date in dates_uniq:
    data_i = data_pf[dates==date,:,:]
    gpp_pf[i,:,:] = np.nanmean(data_i,axis=0)
    i = i+1

gpp_pf2 = gpp_pf[137:,:,:] # 1999-07 to 2020-06

# plt.figure(); plt.imshow(pf_mask2)
# plt.figure(); plt.imshow(gpp_pf[14,:,:])

pf_mask2_rsz = cv2.resize(pf_mask2,(data_pf.shape[2],data_pf.shape[1]), cv2.INTER_NEAREST)
gpp_pf2[:,np.isnan(pf_mask2_rsz)]=np.nan


gpp = gpp_pf2
gpp_rsz = gpp.reshape((gpp.shape[0],gpp.shape[1]*gpp.shape[2])).T
gpp_rsz = gpp_rsz[~np.isnan(gpp_rsz[:,0])]
# remove the pixels, first two years are 0
# make the 0 is nan, and then fill the nan values
# then calculate the sd
gpp_rsz[gpp_rsz==0] = np.nan
mask = np.isnan(np.nanmean(gpp_rsz[:,:12],axis=1))
mask2 = np.isnan(np.nanmean(gpp_rsz[:,-12:],axis=1))

gpp_rsz = gpp_rsz[~(mask | mask2)]
gpp_rsz = fill_nan_with_climatology(gpp_rsz,12)
gpp_rsz = fill_nan_with_climatology(gpp_rsz,12)
gpp_rsz = fill_nan_with_climatology(gpp_rsz,12)

plt.figure(); plt.plot(gpp_rsz[4000,:])

yr_num = 21
bands_yr = 12

gpp_yr = np.zeros_like(gpp_rsz)
for year in range(yr_num):
    st = year * bands_yr
    ed = st + bands_yr
    evi_year = np.nanmean(gpp_rsz[:, st:ed], axis=1)
    evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_yr, axis=1)
    gpp_yr[:, st:ed] = evi_year_rep
rm_offline = gpp_rsz - gpp_yr
# plt.figure(); plt.plot(gpp[:,52,595]); plt.imshow(gpp[1,:,:])

gpp_sea = np.nanmean(np.reshape(rm_offline, (gpp_rsz.shape[0], int(rm_offline.shape[1] / bands_yr), bands_yr)), axis=1)
# plt.figure(); plt.plot(np.nanmean(gpp_sea,axis=0))

Evi_sea_rep = np.zeros_like(rm_offline)
for yr in range(yr_num):
    start_index = (yr - 3) * bands_yr
    end_index = (yr + 3) * bands_yr
    if yr < 3: start_index = 0; end_index = 6 * bands_yr
    if yr > (yr_num - 5): start_index = (yr_num - 6) * bands_yr; end_index = yr_num * bands_yr
    data_i = rm_offline[:, start_index:end_index]
    Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_yr), bands_yr)), axis=1)
    Evi_sea_rep[:, yr * bands_yr:(yr + 1) * bands_yr] = Evi_sea
res = rm_offline - Evi_sea_rep



# gpp_sea_rep = np.zeros_like(rm_offline)
# for yr in range(yr_num):
#     gpp_sea_rep[:, yr * bands_yr:(yr + 1) * bands_yr] = gpp_sea
# res = rm_offline - gpp_sea_rep

plt.figure(); plt.plot(np.nanmean(res,axis=0))

plt.figure(); plt.plot(np.linspace(2001,2021,21),np.nanmedian(moving_sd(res), axis=1))

plt.figure(); plt.plot(np.linspace(2001,2021,21),smooth_array(np.nanmean(moving_sd(res), axis=1),5))
