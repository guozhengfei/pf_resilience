import netCDF4 as nc
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
import os
import cv2
#matplotlib.use('Qt5Agg')
from PIL import Image
from plot_NH import *
import multiprocess as mp
import scipy.stats as st

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr

def cal_slope_ktest(row):
    import numpy as np
    import pymannkendall as mk
    if np.sum(np.isnan(row))>0:
        slopes = [np.nan]*2
    else:
        coef_all = mk.original_test(row)
        slopes = [coef_all.slope, coef_all.p]
    return slopes

# Open the NetCDF file
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
filenames = np.sort(os.listdir(current_dir+'/1_Input/GPP_fluxcom/'))

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

gpp = []
for name in filenames:
    filename = current_dir+'/1_Input/GPP_fluxcom/'+name
    dataset = nc.Dataset(filename)
    # Read data from a specific variable
    variable_name = 'GPP'
    data = dataset.variables[variable_name][:]
    data = np.ma.getdata(data)
    dataset.close()
    rows = 360
    data_pf = data[:, int(np.round((90 - 83.7) / 180 * rows)):int(np.round((90 - 27.0) / 180 * rows)), :]

    data_rsz = []
    for i in range(data_pf.shape[0]):
        data_i_rsz = cv2.resize(data_pf[i,:,:],(pf_mask.shape[1],pf_mask.shape[0]))
        data_rsz.append(data_i_rsz)
    data_rsz = np.array(data_rsz)
    data_rsz = data_rsz[:,::-1,:]
    # plt.figure(); plt.imshow(data_rsz[0,:,:])
    data_rsz[:,np.isnan(pf_mask)]=np.nan
    gpp.append(data_rsz)
    print(name)

gpp = np.concatenate(gpp, axis=0)
gpp_rsz = gpp.reshape((gpp.shape[0],gpp.shape[1]*gpp.shape[2])).T
gpp_rsz = gpp_rsz[~np.isnan(pf_mask_1d), :]
gpp_yr = gpp_rsz.reshape(gpp_rsz.shape[0], 21, 12)
gpp_yr = np.nanmean(gpp_yr, axis=2)

divided_arrays = [row for row in gpp_yr]

with mp.Pool(6) as pool:
    results = list(pool.map(cal_slope_ktest, divided_arrays))
trends = np.array(results)

dataset = rasterio.open(pf_mask_path2)
left, bottom, right, top = np.squeeze(dataset.bounds)
latitudes = np.linspace(top, bottom, dataset.height)
longitudes = np.linspace(left, right, dataset.width)

# Create a 2D grid using meshgrid
grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)
grid_longitudes = grid_longitudes[::3, ::3]
grid_latitudes = grid_latitudes[::3, ::3]

# make circular boundary for polar stereographic circular plots
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

trend_07_map = pf_mask * 1
trend_07_map[~np.isnan(trend_07_map)] = trends[:, 0]
trend_07_map = trend_07_map[::-1,:]

fig = plt.figure(figsize=(8*0.6, 4*0.6))
ax0 = ax1 = fig.add_subplot(1, 2, 1)
ax0.plot(np.linspace(2002,2021,20),np.nanmean(gpp_yr, axis=0)[1:],'o-')
coef = st.linregress(np.linspace(2002,2021,20),np.nanmean(gpp_yr, axis=0)[1:])
ax0.plot([2002,2021],[2002*coef.slope+coef.intercept,2021*coef.slope+coef.intercept],'r-')
ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.NorthPolarStereo())
this1 = ax1.pcolormesh(grid_longitudes, grid_latitudes, trend_07_map,
                       cmap='RdBu', vmin=np.nanpercentile(abs(trend_07_map),80)*-1, vmax=np.nanpercentile(abs(trend_07_map),80),
                       transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.set_boundary(circle, transform=ax1.transAxes)
plt.colorbar(this1, orientation='horizontal', label='GPP_trend', fraction=0.03, pad=0.05)
fig.tight_layout()
figToPath = current_dir + '/4_Figures/FigS03_GPP_trend_fluxcom'
plt.savefig(figToPath, dpi=900)

fig = plt.figure(figsize=(2*0.8, 2*0.8))
plt.hist(trends[:,0],20,density=True,ec='k')
plt.xlim([-0.02,0.04])
fig.tight_layout()
figToPath = current_dir + '/4_Figures/FigS03_GPP_trend_fluxcom_hist'
plt.savefig(figToPath, dpi=900)

sum(trends[:,0]<0)/trends.shape[0]