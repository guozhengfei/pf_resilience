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
    if np.sum(np.isnan(row[24:-24]))>0:
        slopes = [np.nan]*6
    else:
        coef_fh = mk.original_test(row[24:12*8])

        coef_lh = mk.original_test(row[12*8:])

        coef_all = mk.original_test(row[24:-24])

        slopes = [coef_fh.slope, coef_fh.p, coef_lh.slope, coef_lh.p, coef_all.slope, coef_all.p]

    return slopes

if __name__=='__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    # load ar1 data
    tac = np.load(current_dir+'/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling_brdf.npy')

    # plt.figure(); plt.plot(np.nanmean(tac,axis=0))
    divided_arrays = [row for row in tac]

    with mp.Pool(12) as pool:
        results = list(pool.map(cal_slope_ktest, divided_arrays))

    trends = np.array(results) # slope firstHalf, p, slope lastHalf, p, slopeAll, p
    outpath = current_dir+'/2_Output/spatial_resilience/resilience_trend_brdf.npy'
    np.save(outpath, trends)

