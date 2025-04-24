import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import cv2
import tifffile as tf

def read_nc_files(root_folder):
    nc_files = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".nc"):
                nc_files.append(os.path.join(root, file))
    return nc_files

# Specify the root folder where you want to start searching for *.nc files
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

root_folder = current_dir+"/1_Input/OceanE/"
filenames = read_nc_files(root_folder)

oceanE = []
for name in filenames:
    dataset = nc.Dataset(name)
    # Read data from a specific variable
    variable_name = 'evapr'
    data = dataset.variables[variable_name][:]
    data = np.ma.getdata(data)
    dataset.close()
    data[data > 10**3] = np.nan
    data = data[:, ::-1, :]
    evap = np.zeros_like(data)
    evap[:,:,:180]=data[:,:,180:]
    evap[:, :,180:] = data[:, :, :180]

    evap_pf = evap[:, :55, :]
    # evap_pf[evap_pf<=0]=np.nan
    oceanE.append(evap_pf)

oceanE_arr = np.concatenate(oceanE,axis=0).T
oceanE_arr_month = np.nanmean(np.nanmean(oceanE_arr,axis=0),axis=0)
oceanE_arr_yr = oceanE_arr_month.reshape(33,12)
plt.figure(); plt.plot(np.nanmean(oceanE_arr_yr,axis=1))
np.save(current_dir+'/1_Input/data for drivers/oceanE_yr_1990-2022.npy',oceanE_arr_yr)

# plt.figure(); plt.imshow(evap[11,:,:])
# oceanE_arr_rsp = oceanE_arr.reshape((oceanE_arr.shape[0],oceanE_arr.shape[1]*oceanE_arr.shape[2])).T
#
# oceanE_arr_rsp2 = oceanE_arr_rsp.reshape((oceanE_arr_rsp.shape[0],33,12))
# oceanE_yr = np.nanmean(oceanE_arr_rsp2,axis=2)
# oceanE_yr_mean = np.nanmean(oceanE_arr_rsp,axis=0)
# oceanE_yr_mean_rsp = oceanE_yr_mean.reshape((33,12))
# plt.figure();plt.plot(np.nanmean(oceanE_yr_mean_rsp,axis=1))


# SST data
import tifffile as tf
Data_folder = current_dir+'/1_Input/data for drivers/'
sst01 = tf.imread(Data_folder + 'SST_monthly_pf-0000000000-0000000000.tif')
sst02 = tf.imread(Data_folder + 'SST_monthly_pf-0000000000-0000001280.tif')
sst = np.concatenate((sst01, sst02), axis=1)[::3, ::3, ]
sst = sst[:, :-1, :].astype(float)*0.01
sst = sst[:, :, 96:]

sst_month_mean = np.nanmean(np.nanmean(sst,axis=0),axis=0)
sst_yr_mean = sst_month_mean.reshape(33,12)

plt.figure(); plt.plot(np.nanmean(sst_yr_mean,axis=1))
np.save(current_dir+'/1_Input/data for drivers/SST_yr_1990-2022.npy', sst_yr_mean)

# OceanE = tf.imread(current_dir+'/1_Input/ocean_Evap.tif').astype(float)[::3, ::3, ]
# OceanE = OceanE[:,:-1,:]
# OceanE[np.isnan(sst[:,:,3]),:]=np.nan
#
# OceanE_mean = np.nanmean(np.nanmean(OceanE,axis=0),axis=0)
# plt.figure(); plt.plot(OceanE_mean)


# sst2 = []
# for i in range(sst.shape[2]):
#     arr = sst[:,:,i]*1
#     arr_rsz = cv2.resize(arr,(360,40))
#     sst2.append(arr_rsz)
#
# sst2_arr = np.array(sst2)
# sst2_arr_1d = sst2_arr.reshape(-1)
# oceanE_arr_1d = oceanE_arr.reshape(-1)
# mask = np.isnan(sst2_arr_1d) | np.isnan(oceanE_arr_1d)
#
# x = sst2_arr_1d[~mask]
# y = oceanE_arr_1d[~mask]
# plt.figure(); plt.plot(x,y,'.')
# import scipy.stats as st
# st.linregress(x,y)


