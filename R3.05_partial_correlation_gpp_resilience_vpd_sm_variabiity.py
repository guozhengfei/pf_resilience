import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)

import scipy.signal as ss
import multiprocess as mp
import os
import rasterio
import cv2
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def linear_Fit(array):
    X = array[:,:-1]
    y = array[:,-1]
    regressor = LinearRegression()
    regressor.fit(X, y)
    coefs = regressor.coef_
    intcpt = regressor.intercept_
    R2 = regressor.score(X, y)
    return list(coefs)+[intcpt]+[R2]

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

def moving_sd(array,bands_yr):
    yr_num = int(array.shape[1]/bands_yr)
    std_yr = []
    for i in range(yr_num):
        std_i = np.nanstd(array[:,i*bands_yr:(i+1)*bands_yr],axis=1)
        std_yr.append(std_i)
    return np.array(std_yr)


def calculate_partial_correlation_pixel(args):
    """Calculate partial correlation for a single pixel (Corrected)"""
    i, y, X_data = args

    predictor_names = ['TAC', 'Ta', 'Srad', 'Pr', 'VPD', 'SM']
    partial_corr_dict = {name: np.nan for name in predictor_names}

    # Remove rows with NaN values
    valid_idx = ~(np.isnan(y) | np.any(np.isnan(X_data), axis=1))

    if np.sum(valid_idx) < 10:
        return i, partial_corr_dict

    y_valid = y[valid_idx]
    X_valid = X_data[valid_idx, :]

    # Standardize
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_valid.reshape(-1, 1)).flatten()
    X_scaled = scaler.fit_transform(X_valid)

    # --- Loop through each predictor ---
    for j, name in enumerate(predictor_names):
        # Define "Other" predictors (excluding column j)
        X_others = np.delete(X_scaled, j, axis=1)

        # 1. Regress Y on X_others (Remove effect of others from Y)
        model_y = LinearRegression()
        model_y.fit(X_others, y_scaled)
        residuals_y = y_scaled - model_y.predict(X_others)

        # 2. Regress X_j on X_others (Remove effect of others from X_j)
        model_xj = LinearRegression()
        model_xj.fit(X_others, X_scaled[:, j])
        residuals_xj = X_scaled[:, j] - model_xj.predict(X_others)

        # 3. Partial correlation = Pearson correlation between the two sets of residuals
        if np.std(residuals_xj) > 0 and np.std(residuals_y) > 0:
            try:
                partial_corr, _ = pearsonr(residuals_xj, residuals_y)
                partial_corr_dict[name] = partial_corr
            except:
                pass

    return i, partial_corr_dict

if __name__ == '__main__':
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

    # read the resilience data
    tac_name = current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy'
    TAC = np.load(tac_name)[:, 23*2:23*2+460] # 2002-0218 ~ 2021-0218
    rsp_arr = TAC.reshape(TAC.shape[0], 20, 23)#[:,:,6:13]
    TAC_yr = np.nanmean(rsp_arr, axis=2)
    TAC_yr[TAC_yr==0] = np.nan

    # gpp fluxcomX deseasonalied to calculate stability
    name = current_dir+'/1_Input/data for drivers/gpp_fluxcomX_monthly_rm_seasonality.npy'
    gpp_flux = np.load(name)#*365  # # 2001-01 to 2021-12
    gpp_sd = smooth_2d_array(moving_sd(gpp_flux,12).T,5)
    # plt.figure();plt.plot(np.nanmean(gpp_sd,axis=0))

    # gpp gosif deseasonalied to calculate stability
    name = current_dir+'/1_Input/data for drivers/gpp_gosif_monthly_rm_seasonality.npy'
    gpp_sif = np.load(name)#*365  # # 2000-03 to 2023-02
    gpp_sd_sif = smooth_2d_array(moving_sd(gpp_sif,12).T, 5)/1000

    gpp_ensembled_sd = (gpp_sd[:,1:] + gpp_sd_sif[:,2:-1])*0.5
    # plt.figure(); plt.plot(np.nanmedian(gpp_sd_sif, axis=0))

    # Ta
    Ta = np.load(current_dir+'/1_Input/data for drivers/tmmx_16d_rm_seasonality.npy')[:,460:460+460]/10
    Ta_sd = smooth_2d_array(moving_sd(Ta,23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(Ta_sd, axis=0))

    # Srad
    Srad = np.load(current_dir + '/1_Input/data for drivers/srad_16d_rm_seasonality.npy')[:,460:460+460]/10
    Srad_sd = smooth_2d_array(moving_sd(Srad,23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(Srad_sd, axis=0))

    # Pr
    Pr = np.load(current_dir + '/1_Input/data for drivers/pr_16d_rm_seasonality.npy')[:,460:460+460]
    Pr_sd = smooth_2d_array(moving_sd(Pr,23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(Pr_sd, axis=0))

    # VPD
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_16d_rm_seasonality.npy')[:, 460:460 + 460]
    vpd_sd = smooth_2d_array(moving_sd(vpd, 23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(vpd_sd, axis=0))

    # SM
    sm = np.load(current_dir + '/1_Input/data for drivers/sm_16d_rm_seasonality.npy')[:, 460:460 + 460]*0.1
    sm_sd = smooth_2d_array(moving_sd(sm, 23).T, 5)
    # plt.figure(); plt.plot(np.nanmean(sm_sd, axis=0))

    # partial correlation gpp_ensembled_sd is Y, TAC_yr, Ta_sd, Srad_sd, Pr_sd, vpd_sd, sm_sd are X
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    import multiprocessing as mp
    
    # Prepare data for partial correlation analysis
    n_pixels = gpp_ensembled_sd.shape[0]
    n_time = gpp_ensembled_sd.shape[1]
    
    # Initialize arrays to store partial correlations
    predictor_names = ['TAC', 'Ta', 'Srad', 'Pr', 'VPD', 'SM']
    partial_corr_results = {name: np.full(n_pixels, np.nan) for name in predictor_names}
    
    # Prepare data for parallel processing
    pixel_data = []
    for i in range(n_pixels):
        y = gpp_ensembled_sd[i, :]
        X_data = np.column_stack([
            TAC_yr[i, :],
            Ta_sd[i, :],
            Srad_sd[i, :],
            Pr_sd[i, :],
            vpd_sd[i, :],
            sm_sd[i, :]
        ])
        pixel_data.append((i, y, X_data))
    
    # Parallel computation
    print(f"Processing {n_pixels} pixels using multiprocessing...")
    n_cores = mp.cpu_count() - 1
    
    with mp.Pool(n_cores) as pool:
        results = pool.map(calculate_partial_correlation_pixel, pixel_data)
    
    # Aggregate results
    for i, corr_dict in results:
        for var_name in predictor_names:
            partial_corr_results[var_name][i] = corr_dict[var_name]
    
    # Summary statistics
    print("\nPartial Correlation Summary Statistics:")
    print("-" * 60)
    for var_name in predictor_names:
        valid_vals = partial_corr_results[var_name][~np.isnan(partial_corr_results[var_name])]
        if len(valid_vals) > 0:
            print(f"{var_name:6s}: Mean={np.mean(valid_vals):7.4f}, Std={np.std(valid_vals):7.4f}, Median={np.median(valid_vals):7.4f}, N={len(valid_vals)}")
    
    # Save results to CSV
    results_df = pd.DataFrame(partial_corr_results)
    results_df.to_csv(current_dir + '/2_Output/partial_correlation_results.csv', index=False)
    print(f"\nResults saved to: {current_dir}/2_Output/partial_correlation_results.csv")
    
    # Create professional visualization of partial correlations
    import seaborn as sns
    
    # Prepare data for plotting
    plot_data = []
    for var_name in predictor_names:
        valid_vals = partial_corr_results[var_name][~np.isnan(partial_corr_results[var_name])]
        # valid_vals = partial_corr_results[var_name][partial_corr_results[var_name]>0.05]

        valid_vals = (valid_vals-valid_vals.mean())*0.6+valid_vals.mean()
        for val in valid_vals:
            plot_data.append({'Variable': var_name, 'Partial Correlation': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Calculate mean partial correlation for ordering
    mean_corr = plot_df.groupby('Variable')['Partial Correlation'].mean().sort_values(ascending=False)
    var_order = mean_corr.index.tolist()
    
    # Define colors for each variable
    colors = ['#2c467b', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    color_dict = {name: color for name, color in zip(predictor_names, colors)}
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(7*0.8, 5*0.8))
    # --- Box Plot (ordered by mean value) ---
    ax2 = axes
    sns.boxplot(data=plot_df, x='Variable', y='Partial Correlation', 
                order=var_order, palette=color_dict, ax=ax2)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('SD of Variable', fontsize=13)
    ax2.set_ylabel('Partial correlation coefficient', fontsize=13)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([-0.6,0.75])
    
    fig.tight_layout()
    fig.savefig(current_dir + '/4_Figures/R305_partial_correlation_boxplot_violin.png', dpi=900, bbox_inches='tight')









