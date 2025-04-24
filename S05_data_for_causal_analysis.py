import numpy as np
import pandas as pd
import os
import multiprocess as mp
import warnings
warnings.filterwarnings("ignore")

def std_filter(array):
    meanV = np.nanmean(array)
    stdV = np.nanstd(array)
    array[array > meanV + 3 * stdV] = np.nan
    array[array < meanV - 3 * stdV] = np.nan
    return array

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

def model_validate(array):
    import numpy as np
    import xgboost as xgb
    import scipy.stats as st
    import shap
    X = array[0, :, :-1]
    y = array[0, :, -1]
    model_ebm = xgb.XGBRegressor()
    Pred = []
    for n in range(array.shape[1]):
        X1 = np.delete(X, n, axis=0)
        y1 = np.delete(y, n, axis=0)
        model_ebm.fit(X1, y1)
        pred = model_ebm.predict(X[n, :].reshape(1, X.shape[1]))[0]
        Pred.append(pred)
    try:
        r = st.linregress(Pred, y).rvalue
    except ValueError:
        r = np.nan
    # explain the GAM model with SHAP
    model = model_ebm.fit(X, y)
    explainer_ebm = shap.Explainer(model)
    shap_values_ebm = explainer_ebm(X)
    shap_abs = np.mean(abs(shap_values_ebm.values), axis=0)
    sen = []
    for m in range(shap_values_ebm.shape[1]):
        x2 = shap_values_ebm.data[:, m]
        y2 = shap_values_ebm.values[:, m]
        # remove the outlier with 3*std
        meanV = np.nanmean(y2)
        stdV = np.nanstd(y2)
        y2[y2 > meanV + 3 * stdV] = np.nan
        y2[y2 < meanV - 3 * stdV] = np.nan
        mask = np.isnan(std_filter(y2))
        try:
            slope = st.linregress(x2[~mask], y2[~mask]).slope
        except ValueError:
            slope = np.nan
        sen.append(slope)
    return np.array([r] + list(shap_abs) + sen + list(y) + list(Pred))


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    ar1_5yr_modis[ar1_5yr_modis==0] = np.nan
    ar1_5yr_modis = np.hstack((ar1_5yr_modis[:,:-2],ar1_5yr_modis[:,-3:]))
    # all pixel trend of ar1
    ar1_rsp = ar1_5yr_modis.reshape(ar1_5yr_modis.shape[0], 24, 23)
    ar1_yearly = np.nanmean(ar1_rsp, axis=2)[:,2:-2]

    # vegetation variables
    ndvi = np.load(current_dir + '/1_Input/data for drivers/ndvi_yearly.npy')[:,2:-2]
    sos = np.load(current_dir + '/1_Input/data for drivers/sos.npy')[:,20:-1]
    eos = np.load(current_dir + '/1_Input/data for drivers/eos.npy')[:,20:-1]
    gsl = eos - sos
    row_means = np.nanmean(gsl, axis=1)
    nan_indices = np.isnan(gsl)
    gsl[nan_indices] = np.take(row_means, np.where(nan_indices)[0])

    # climate variables
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:,20:-1]
    tmmx = np.load(current_dir + '/1_Input/data for drivers/tmmx_yearly.npy')[:,20:-1]
    tmmn = np.load(current_dir + '/1_Input/data for drivers/tmmn_yearly.npy')[:,20:-1]
    tmean = (tmmn+tmmx)*0.5
    srad = np.load(current_dir + '/1_Input/data for drivers/srad_yearly.npy')[:,20:-1]
    pr = np.load(current_dir + '/1_Input/data for drivers/pr_yearly.npy')[:,20:-1]
    def_data = np.load(current_dir + '/1_Input/data for drivers/def_yearly.npy')[:,20:-1]

    # soil variables
    soilT = np.load(current_dir + '/1_Input/data for drivers/sT_yearly.npy')[:,20:-1]
    Alt = np.load(current_dir + '/1_Input/data for drivers/Alt.npy')[:,20:-1]
    sm = np.load(current_dir + '/1_Input/data for drivers/sm_yearly.npy')[:,20:-1]
    lai = np.load(current_dir+'/1_Input/data for drivers/LAI_yearly.npy')[:,20:-1]

    # remove the rows with nan values
    mask = np.isnan(sm).any(axis=1) | np.isnan(vpd).any(axis=1) | np.isnan(srad).any(axis=1) | np.isnan(pr).any(axis=1) | np.isnan(tmean).any(axis=1) | np.isnan(ndvi).any(axis=1) | np.isnan(ar1_yearly).any(axis=1) | np.isnan(soilT).any(axis=1)  | np.isnan(Alt).any(axis=1)# | np.isnan(sos_mean)#

    # np.save(r'D:\Projects\Project_pf\Data\Output\temporal_resilience_attribution\mask_temporal.npy',mask)
    vpd_annual_data = smooth_2d_array(vpd[~mask, :],3)
    srad_annual_data = smooth_2d_array(srad[~mask, :],3)
    pr_annual_data = smooth_2d_array(pr[~mask, :],3)
    tmean_annual_data = smooth_2d_array(tmean[~mask, :],3)
    sm_annual_data = smooth_2d_array(sm[~mask, :],3)
    sT_annual_data = smooth_2d_array(soilT[~mask, :],3)
    alt_annual_data = smooth_2d_array(Alt[~mask, :],3)
    ndvi_annual_data = smooth_2d_array(ndvi[~mask, :],3)
    sos_annual_data = smooth_2d_array(sos[~mask, :],3)
    eos_annual_data = smooth_2d_array(eos[~mask, :],3)
    gsl_annual_data = smooth_2d_array(gsl[~mask, :],3)
    lai_annual_data = smooth_2d_array(lai[~mask, :],3)
    tac_annual_data = smooth_2d_array(ar1_yearly[~mask, :],3)
    # construct a xgboost model to predict tac

    co2_df = pd.read_csv(current_dir + '/1_Input/data for drivers/co2_mm_mlo_v2.csv').groupby('year').mean()
    co2 = co2_df['average'][-20:].values
    co2_2d = np.tile(co2, (ndvi_annual_data.shape[0], 1))

    array = np.stack(
        [vpd_annual_data, srad_annual_data, pr_annual_data, tmean_annual_data, tac_annual_data], axis=2)  # remove phenology

    output_file = current_dir+'/2_Output/data_for_causal_abalysis.npy'
    np.save(output_file, array)