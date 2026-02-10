import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Qt5Agg')
import multiprocess as mp
# from plot_NH import *
import tifffile as tf
import os
from PIL import Image

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr
def smooth_2d_array(arr, window_size):
    smoothed_arr = np.zeros_like(arr)
    rows, cols = arr.shape
    # Pad the array to handle edge cases
    padding = window_size // 2
    padded_arr = np.pad(arr, ((0, 0), (padding, padding)), mode='edge')

    # Apply moving average along each row
    for i in range(rows):
        smoothed_arr[i] = np.convolve(padded_arr[i], np.ones(window_size) / window_size, mode='valid')

    return smoothed_arr

def cal_slope_ktest(row):
    import numpy as np
    import pymannkendall as mk
    if np.sum(np.isnan(row[46:-46]))>0:
        slopes = [np.nan]*6
    else:
        coef_fh = mk.original_test(row[46:184-23])

        coef_lh = mk.original_test(row[184-23:])

        coef_all = mk.original_test(row[46:-46])

        slopes = [coef_fh.slope, coef_fh.p, coef_lh.slope, coef_lh.p, coef_all.slope, coef_all.p]

    return slopes

if __name__=='__main__':
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

    # load ar1 data
    tac = np.load(current_dir+'/2_Output/spatial_resilience/ar1_5yr_vod_sg_rolling.npy')

    # plt.figure(); plt.plot(np.nanmean(tac,axis=0))
    divided_arrays = [row for row in tac]

    with mp.Pool(6) as pool:
        results = list(pool.map(cal_slope_ktest, divided_arrays))

    trends = np.array(results) # slope firstHalf, p, slope lastHalf, p, slopeAll, p
    outpath = current_dir+'/2_Output/spatial_resilience/resilience_trend_vod.npy'
    np.save(outpath, trends)

