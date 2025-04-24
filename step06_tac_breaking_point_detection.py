import numpy as np
# import matplotlib.pyplot as plt
import pymannkendall as mk
import pwlf
import multiprocessing as mp
import os

def obtain_break_point(y):
    y_len = y.shape[0]
    x = np.linspace(0, y_len-1, y_len)
    my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=1)
    res = my_pwlf_0.fit(2)
    return my_pwlf_0.fit_breaks[1]

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    tac0 = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((tac0[:, :-2], tac0[:, -3:]))
    # plt.figure()
    # plt.plot(np.nanmean(tac, axis=0))

    ar1_rsp = tac.reshape(tac.shape[0], 24, 23)
    ar1_yearly = np.nanmean(ar1_rsp, axis=2)[:, 2:-2]
    ar1_yearly[ar1_yearly==0] = np.nan
    ar1_yearly2 = ar1_yearly[~np.isnan(ar1_yearly).any(axis=1)]

    ar1_yearly2 = smooth_2d_array(ar1_yearly2,3)
    # plt.figure(); plt.plot(np.nanmean(ar1_yearly2,axis=0))

    divided_arrays = [row for row in ar1_yearly2]

    with mp.Pool(120) as pool:
        results = list(pool.map(obtain_break_point, divided_arrays))

    break_points = np.array(results)
    output_file = current_dir + '/2_Output/break_points_modis.npy'
    np.save(output_file, break_points)