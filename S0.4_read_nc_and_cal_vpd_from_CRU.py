import netCDF4 as nc
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
import os
import cv2

matplotlib.use('Qt5Agg')
# Open the NetCDF file
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
filenames = os.listdir(current_dir+'/1_Input/CRU/vap/')
Vap = []
for name in filenames:
    filename = current_dir+'/1_Input/CRU/vap/'+name
    dataset = nc.Dataset(filename)
    # Read data from a specific variable
    variable_name = 'vap'
    data = dataset.variables[variable_name][:]
    data = np.ma.getdata(data)
    dataset.close()
    vap = data
    vap[vap>10**10]=np.nan
    vap = vap[:,::-1,:]
    vap_pf = vap[:, 13:126, :]
    Vap.append(vap_pf)

Vap = np.concatenate(Vap, axis=0)[12:,:,:]
# plt.figure(); plt.imshow(vap_pf[1,:,:])

filenames = os.listdir(current_dir+'/1_Input/CRU/tmp/')
Tmp = []
for name in filenames:
    filename = current_dir+'/1_Input/CRU/tmp/'+name
    dataset = nc.Dataset(filename)
    # Read data from a specific variable
    variable_name = 'tmp'
    data = dataset.variables[variable_name][:]
    data = np.ma.getdata(data)
    dataset.close()
    tmp = data
    tmp[tmp>10**10]=np.nan
    tmp = tmp[:,::-1,:]
    tmp_pf = tmp[:, 13:126, :]
    Tmp.append(tmp_pf)

Tmp = np.concatenate(Tmp, axis=0)[12:,:,:]

# plt.figure(); plt.imshow(Tmp[6,:,:])
ea_star = 0.61078 * np.exp(17.27 * Tmp / (Tmp + 237.3))
VPD = ea_star*10-Vap

## pf_mask
pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
pf_mask2 = tf.imread(pf_mask_path2).astype(float)
# pf_mask3 = tf.imread(pf_mask_path3).astype(float)
# pf_mask2[pf_mask2 != pf_mask3] = np.nan
pf_mask = pf_mask2[::3, ::3]
pf_mask = cv2.resize(pf_mask,(Tmp.shape[2],Tmp.shape[1]),cv2.INTER_NEAREST)
pf_mask[pf_mask <= 0] = np.nan
pf_mask[pf_mask > 12] = np.nan

# plt.figure(); plt.imshow(pf_mask)
VPD = VPD[:,~np.isnan(pf_mask)].T

VPD_rsz = VPD.reshape(VPD.shape[0],41,12)
VPD_yr = np.nanmedian(VPD_rsz,axis=2)
plt.figure(); plt.plot(np.nanmean(VPD_yr,axis=0))

np.save(current_dir+'/1_Input/data for drivers/vpd_yr_1982-2022.npy',VPD_yr)

ea_star = ea_star[:,~np.isnan(pf_mask)].T
svap_rsz = ea_star.reshape(ea_star.shape[0],41,12)
svap_yr = np.nanmean(svap_rsz,axis=2)
plt.figure(); plt.plot(np.nanmean(svap_yr,axis=0))
np.save(current_dir+'/1_Input/data for drivers/svap_yr_1982-2022.npy',svap_yr)

Vap = Vap[:,~np.isnan(pf_mask)].T
Vap_rsz = Vap.reshape(Vap.shape[0],41,12)
Vap_yr = np.nanmedian(Vap_rsz,axis=2)
plt.figure(); plt.plot(np.nanmean(Vap_yr,axis=0))
np.save(current_dir+'/1_Input/data for drivers/vap_yr_1982-2022.npy',Vap_yr)

Tmp = Tmp[:,~np.isnan(pf_mask)].T
Tmp_rsz = Tmp.reshape(Tmp.shape[0],41,12)
Tmp_yr = np.nanmedian(Tmp_rsz,axis=2)
np.save(current_dir+'/1_Input/data for drivers/Tmp_yr_1982-2022.npy',Tmp_yr)