import netCDF4 as nc
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
import os

matplotlib.use('Qt5Agg')
# Open the NetCDF file
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

filenames = os.listdir(r'D:\Projects\Project_pf\1_Input\CRU\vap\\')
Vap = []
for name in filenames:
    filename = r'D:\Projects\Project_pf\1_Input\CRU\vap\\'+name
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


filenames = os.listdir(r'D:\Projects\Project_pf\1_Input\CRU\tmp\\')
Tmp = []
for name in filenames:
    filename = r'D:\Projects\Project_pf\1_Input\CRU\tmp\\'+name
    dataset = nc.Dataset(filename)
    # Read data from a specific variable
    variable_name = 'tmp'
    data = dataset.variables[variable_name][:]
    data = np.ma.getdata(data)
    dataset.close()
    vap = data
    vap[vap>10**10]=np.nan
    vap = vap[:,::-1,:]
    vap_pf = vap[:, 13:126, :]
    Tmp.append(vap_pf)

Tmp = np.concatenate(Tmp, axis=0)[12:,:,:]

plt.figure(); plt.imshow(Tmp[1,:,:])
ea_star = 0.61078 * np.exp(17.27 * Tmp / (Tmp + 237.3))

## pf_mask
EVI_folder = r'D:\Projects\Project_pf\1_Input\GIMMS_NDVI\\'
evi_filenames = os.listdir(EVI_folder)
pf_mask_path = r'D:\Projects\Project_pf\1_Input\Land_Cover_PF_3g.tif'
pf_mask0 = tf.imread(pf_mask_path).astype(float)
pf_mask0[pf_mask0 <= 0] = np.nan
pf_mask0[pf_mask0 > 12] = np.nan

evi1 = np.load(EVI_folder + evi_filenames[14])
evi2 = np.load(EVI_folder + evi_filenames[15])
merged_evi = np.where(np.isnan(evi1), evi2, evi1)  # Where condition True, yield x, otherwise yield y.
pf_mask0[merged_evi <= 0] = np.nan

pf_mask = pf_mask0[::6, ::6]
pf_mask=pf_mask[0:-1,:]
pf_mask_1d = pf_mask.flatten()

VPD = ea_star*10-Vap

VPD0 = np.reshape(VPD,(VPD.shape[0],VPD.shape[1]*VPD.shape[2]))
ser = VPD0[:,~np.isnan(pf_mask_1d)].T

vpd_month = np.nanmean(ser,axis=0)
vpd_yr = vpd_month.reshape(41,12)
np.save(current_dir+'/1_Input/data for drivers/vpd_yr_1982-2022.npy',vpd_yr)

svap0 = ea_star*10
svap0 = np.reshape(svap0,(svap0.shape[0],svap0.shape[1]*svap0.shape[2]))
ser = svap0[:,~np.isnan(pf_mask_1d)].T
svap_month = np.nanmean(ser,axis=0)
svap_yr = svap_month.reshape(41,12)
np.save(current_dir+'/1_Input/data for drivers/svap_yr_1982-2022.npy',svap_yr)

vap0 = Vap
vap0 = np.reshape(vap0,(vap0.shape[0],vap0.shape[1]*vap0.shape[2]))
ser = vap0[:,~np.isnan(pf_mask_1d)].T
vap_month = np.nanmean(ser,axis=0)
vap_yr = vap_month.reshape(41,12)
np.save(current_dir+'/1_Input/data for drivers/vap_yr_1982-2022.npy',vap_yr)

tmp0 = Tmp
tmp0 = np.reshape(tmp0,(tmp0.shape[0],tmp0.shape[1]*tmp0.shape[2]))
ser = tmp0[:,~np.isnan(pf_mask_1d)].T
tmp_month = np.nanmean(ser,axis=0)
tmp_yr = tmp_month.reshape(41,12)
np.save(current_dir+'/1_Input/data for drivers/Tmp_yr_1982-2022.npy',tmp_yr)

plt.figure()
plt.plot(np.nanmean(vpd_yr, axis=1))

