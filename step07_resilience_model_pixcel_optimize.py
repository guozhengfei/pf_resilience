import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import xgboost as xgb
import scipy.stats as st
import shap
from numba import jit
from tqdm import tqdm

@jit(nopython=True)
def std_filter(array):
    meanV = np.nanmean(array)
    stdV = np.nanstd(array)
    for i in range(len(array)):
        if array[i] > meanV + 3 * stdV or array[i] < meanV - 3 * stdV:
            array[i] = np.nan
    return array

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    n_rows, n_cols = array.shape
    pad_width = window_size // 2
    padded = np.pad(array, ((0, 0), (pad_width, pad_width)), mode='edge')
    output = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        output[i] = np.convolve(padded[i], kernel, mode='valid')
    return output

def model_validate(array):
    array = array[0]
    X = array[:, :-1]
    y = array[:, -1]

    Pred = np.zeros(len(y))
    model_ebm = xgb.XGBRegressor(tree_method='hist')
    for n in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[n] = False
        X_train, y_train = X[mask], y[mask]
        model_ebm.fit(X_train, y_train)
        Pred[n] = model_ebm.predict(X[n:n + 1])[0]

    r = st.linregress(Pred, y).rvalue if np.all(np.isfinite(Pred)) else np.nan

    model = model_ebm.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_abs = np.mean(np.abs(shap_values), axis=0)

    sen = np.zeros(shap_values.shape[1])
    for m in range(shap_values.shape[1]):
        x2 = X[:, m]
        y2 = std_filter(shap_values[:, m].copy())
        mask = ~np.isnan(y2)
        if np.sum(mask) > 1:
            x2_masked = x2[mask]
            y2_masked = y2[mask]
            if np.all(x2_masked == x2_masked[0]):  # Check for identical values
                sen[m] = np.nan
            else:
                sen[m] = st.linregress(x2_masked, y2_masked).slope
        else:
            sen[m] = np.nan
    return np.array([r] + list(shap_abs) + list(sen) + list(y) + list(Pred))

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    data_files = {
        'ar1': '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy',
        'ndvi': '/1_Input/data for drivers/ndvi_yearly.npy',
        'sos': '/1_Input/data for drivers/sos.npy',
        'eos': '/1_Input/data for drivers/eos.npy',
        'vpd': '/1_Input/data for drivers/vpd_yearly.npy',
        'tmmx': '/1_Input/data for drivers/tmmx_yearly.npy',
        'tmmn': '/1_Input/data for drivers/tmmn_yearly.npy',
        'srad': '/1_Input/data for drivers/srad_yearly.npy',
        'pr': '/1_Input/data for drivers/pr_yearly.npy',
        'def': '/1_Input/data for drivers/def_yearly.npy',
        'soilT': '/1_Input/data for drivers/sT_yearly.npy',
        'Alt': '/1_Input/data for drivers/Alt.npy',
        'sm': '/1_Input/data for drivers/sm_yearly.npy',
        'lai': '/1_Input/data for drivers/LAI_yearly.npy'
    }

    data = {k: np.load(current_dir + v) for k, v in data_files.items()}

    data['ar1'][data['ar1'] == 0] = np.nan
    data['ar1'] = np.hstack((data['ar1'][:, :-2], data['ar1'][:, -3:]))
    ar1_rsp = data['ar1'].reshape(-1, 24, 23)
    ar1_yearly = np.nanmean(ar1_rsp, axis=2)[:, 2:-2]
    data['tac'] = ar1_yearly

    data['ndvi'] = data['ndvi'][:, 2:-2]
    data['sos'] = data['sos'][:, 20:-1]
    data['eos'] = data['eos'][:, 20:-1]
    data['gsl'] = data['eos'] - data['sos']
    row_means = np.nanmean(data['gsl'], axis=1)
    data['gsl'][np.isnan(data['gsl'])] = row_means[np.where(np.isnan(data['gsl']))[0]]

    data['tmean'] = (data['tmmx'][:, 20:-1] + data['tmmn'][:, 20:-1]) * 0.5
    for k in ['vpd', 'srad', 'pr', 'soilT', 'Alt', 'sm', 'lai']:
        data[k] = data[k][:, 20:-1]

    mask_vars = ['sm', 'vpd', 'srad', 'pr', 'tmean', 'ndvi', 'soilT', 'Alt', 'gsl', 'tac']
    mask = np.any([np.isnan(data[v]).any(axis=1) for v in mask_vars], axis=0)

    variables = ['vpd', 'srad', 'pr', 'tmean', 'soilT', 'sm', 'Alt', 'ndvi', 'gsl', 'lai', 'tac']
    smoothed_data = {v: smooth_2d_array(data[v][~mask], 3) for v in variables}

    array = np.stack([smoothed_data[v] for v in variables], axis=2)

    with mp.Pool(min(mp.cpu_count(), 4)) as pool:
        array_split = np.split(array, array.shape[0], axis=0)
        results = list(tqdm(pool.imap(model_validate, array_split), total=len(array_split), desc="Processing"))

    np.savez(current_dir + '/2_Output/Temporal/Temporal_r_shap_obs_pre_opt.npy',
             array1=np.array(results), array2=mask)