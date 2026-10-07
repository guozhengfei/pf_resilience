import numpy as np

from scipy.stats import sem, t
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
from PIL import Image
import multiprocess as mp
import os
import pandas as pd

# Pointwise intervals assume independent sampling units (pixels or DGVMs).
def standard_error(values, axis=0):
    """Sample SD (ddof=1) / sqrt(non-missing n), along the sampling axis."""
    values = np.asarray(values, dtype=float)
    if np.isinf(values).any() or np.any(np.isfinite(values).sum(axis=axis) < 2):
        raise ValueError("SE requires at least two finite observations per estimate.")
    return sem(values, axis=axis, ddof=1, nan_policy="omit")


def mean_ci95(values, axis=0):
    """Return the mean, Student-t 95% CI half-width and non-missing sample count."""
    values = np.asarray(values, dtype=float)
    n = np.isfinite(values).sum(axis=axis)
    half_width = t.ppf(0.975, n - 1) * standard_error(values, axis=axis)
    return np.nanmean(values, axis=axis), half_width, n


def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    files = os.listdir(current_dir +'/2_Output/Trendy_resilience_v2/')
    file_list = []
    for file in files:
        if not file.startswith('.'):
            file_list.append(file)

    TAC = [];
    Trend = []
    for name in file_list:
        tac = np.load(current_dir +'/2_Output/Trendy_resilience_v2/'+name)
        tac = tac[:,24:-24]
        mask = np.isnan(tac).any(axis=1)
        tac = tac[~mask,:]
        tac08 = tac[:,:84]
        tac23 = tac[:,84:]
        tac_trend_08 = np.polyfit(np.arange(tac08.shape[1]) / 12, tac08.T, deg=1)[0,:]
        tac_trend_23 = np.polyfit(np.arange(tac23.shape[1]) / 12, tac23.T, deg=1)[0, :]
        trend08_mean = tac_trend_08.mean()
        trend08_sd = tac_trend_08.std(ddof=1)
        trend23_mean = tac_trend_23.mean()
        trend23_sd = tac_trend_23.std(ddof=1)
        tac_mean = np.nanmean(tac,axis=0)
        Trend.append([trend08_mean,trend08_sd, trend23_mean,trend23_sd])
        TAC.append([tac_mean])
        # if np.nanmean(tac_mean)>0:
        #     axs[0].plot(time_label_modis,tac_mean,lw=0.8)

    Trend = np.array(Trend)
    TAC = np.array(TAC)
    TAC_mean = np.nanmean(TAC,axis=2).reshape(-1)

    TAC = TAC[TAC_mean>0,:,:] # remove the tac <0 model results
    Trend = Trend[TAC_mean>0,:]

    # load ar1 data: 5years window
    ar1_5yr_modis = np.load(current_dir + '/2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy')
    tac = np.hstack((ar1_5yr_modis[:, :-2], ar1_5yr_modis[:, -3:]))
    tac[tac == 0] = np.nan;
    tac_yr = tac.reshape(tac.shape[0], 24, 23)
    tac_yr = np.nanmean(tac_yr, axis=2)[:, 2:-2]
    tac_yr_rpt = np.tile(tac_yr[:, 0], (tac_yr.shape[1], 1)).T
    tac2 = tac_yr #- tac_yr_rpt

    from brokenaxes import brokenaxes

    fig, axs = plt.subplots(1, figsize=(12 * 0.8, 2.5 * 0.8))
    bax = brokenaxes(ylims=((-0.025, -0.02),(-0.015, 0.015), (0.040, 0.047)), hspace=0.05)
    bax.bar(range(15),Trend[:,0],yerr=Trend[:,1],width=0.3,color='#d6604d', ec='k', hatch='//')
    # The All bar describes the spread of model-specific mean trends.
    bax.bar(15, Trend[:, 0].mean(), yerr=Trend[:, 0].std(ddof=1),width=0.3, color='#4393c3', ec='k', hatch='//')
    bax.bar(np.array(range(15))+0.3, Trend[:, 2],yerr=Trend[:,3], width=0.3,color='#d6604d', ec='k')
    bax.bar(15+0.3, Trend[:, 2].mean(), yerr=Trend[:, 2].std(ddof=1),width=0.3, color='#4393c3', ec='k', hatch='//')
    axs.set_axis_off()

    xtick=[]
    for file in file_list:
        xtick.append(file.split('.')[0].split('_')[0])
    xtick.pop(-5)
    xtick = xtick + ['All']
    bax.set_xticks(range(16),xtick,rotation=90)
    # fig.tight_layout()
    # figToPath = current_dir + '/4_Figures/Fig05_resilience_trendy01'
    # plt.savefig(figToPath, dpi=600)

    time_label_modis = np.linspace(2002, 2020, 19)

    # drivers of trendy gpp resilience
    # vegetation variables
    ndvi = np.load(current_dir + '/1_Input/data for drivers/ndvi_yearly.npy')[:, 2:-2]
    sos = np.load(current_dir + '/1_Input/data for drivers/sos.npy')[:, 20:-1]
    eos = np.load(current_dir + '/1_Input/data for drivers/eos.npy')[:, 20:-1]
    gsl = eos - sos
    row_means = np.nanmean(gsl, axis=1)
    nan_indices = np.isnan(gsl)
    gsl[nan_indices] = np.take(row_means, np.where(nan_indices)[0])

    # climate variables
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:, 20:-1]
    tmmx = np.load(current_dir + '/1_Input/data for drivers/tmmx_yearly.npy')[:, 20:-1]
    tmmn = np.load(current_dir + '/1_Input/data for drivers/tmmn_yearly.npy')[:, 20:-1]
    tmean = (tmmn + tmmx) * 0.5
    srad = np.load(current_dir + '/1_Input/data for drivers/srad_yearly.npy')[:, 20:-1]
    pr = np.load(current_dir + '/1_Input/data for drivers/pr_yearly.npy')[:, 20:-1]
    def_data = np.load(current_dir + '/1_Input/data for drivers/def_yearly.npy')[:, 20:-1]

    # soil variables
    soilT = np.load(current_dir + '/1_Input/data for drivers/sT_yearly.npy')[:, 20:-1]
    Alt = np.load(current_dir + '/1_Input/data for drivers/Alt.npy')[:, 20:-1]
    sm = np.load(current_dir + '/1_Input/data for drivers/sm_yearly.npy')[:, 20:-1]
    lai = np.load(current_dir + '/1_Input/data for drivers/LAI_yearly.npy')[:, 20:-1]

    # remove the rows with nan values
    mask = np.isnan(sm).any(axis=1) | np.isnan(vpd).any(axis=1) | np.isnan(srad).any(axis=1) | np.isnan(pr).any(
        axis=1) | np.isnan(tmean).any(axis=1) | np.isnan(ndvi).any(axis=1) | np.isnan(tac_yr).any(
        axis=1) | np.isnan(soilT).any(axis=1) | np.isnan(Alt).any(axis=1)  # | np.isnan(sos_mean)#

    # np.save(r'D:\Projects\Project_pf\Data\Output\temporal_resilience_attribution\mask_temporal.npy',mask)
    vpd_annual_data = np.nanmean(smooth_2d_array(vpd[~mask, :], 3),axis=0)
    srad_annual_data = np.nanmean(smooth_2d_array(srad[~mask, :], 3),axis=0)
    pr_annual_data = np.nanmean(smooth_2d_array(pr[~mask, :], 3),axis=0)
    tmean_annual_data = np.nanmean(smooth_2d_array(tmean[~mask, :], 3),axis=0)
    sm_annual_data = np.nanmean(smooth_2d_array(sm[~mask, :], 3),axis=0)
    sT_annual_data = np.nanmean(smooth_2d_array(soilT[~mask, :], 3),axis=0)
    alt_annual_data = np.nanmean(smooth_2d_array(Alt[~mask, :], 3),axis=0)
    ndvi_annual_data = np.nanmean(smooth_2d_array(ndvi[~mask, :], 3),axis=0)
    sos_annual_data = np.nanmean(smooth_2d_array(sos[~mask, :], 3),axis=0)
    eos_annual_data = np.nanmean(smooth_2d_array(eos[~mask, :], 3),axis=0)
    gsl_annual_data = np.nanmean(smooth_2d_array(gsl[~mask, :], 3),axis=0)
    lai_annual_data = np.nanmean(smooth_2d_array(lai[~mask, :], 3),axis=0)

    X = np.stack(
        [vpd_annual_data, srad_annual_data, pr_annual_data, tmean_annual_data, sT_annual_data, sm_annual_data,
         alt_annual_data, ndvi_annual_data, gsl_annual_data, lai_annual_data
         ], axis=1)[:-1,:]
    
    y = np.mean(TAC[0,:,:].reshape(19,12),axis=1)

    # Add XGBoost and SHAP analysis
    import xgboost as xgb
    import shap
    # Store SHAP values for each model
    all_shap_values = []
    
    # Loop through all TAC models
    for i in range(TAC.shape[0]):
        y = np.mean(TAC[i,:,:].reshape(19,12), axis=1)
        
        # Create and train XGBoost model
        model = xgb.XGBRegressor().fit(X, y)
        
        # Calculate SHAP values for this model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        all_shap_values.append(np.abs(shap_values).mean(0))  # Store mean absolute SHAP values
    
    # Convert to numpy array
    all_shap_values = np.array(all_shap_values)
    
    # Caption: one sample SD across DGVM-specific mean absolute SHAP values.
    mean_shap = np.mean(all_shap_values, axis=0)
    sd_shap = np.std(all_shap_values, axis=0, ddof=1)
    
    # Create feature names list
    feature_names = ['VPD', 'Srad', 'Pr', 'Ta','Ts', 'SM', 'ALT', 'kNDVI','GSL', 'LAI']
    colors = ['#878787', '#878787', '#878787', '#878787', '#d6604d', '#d6604d', '#d6604d', '#4393c3', '#4393c3','#4393c3']
    # Sort features by mean importance
    sort_idx = np.argsort(mean_shap)
    mean_shap = mean_shap[sort_idx]
    sd_shap = sd_shap[sort_idx]
    feature_names = [feature_names[i] for i in sort_idx]
    colors = [colors[i] for i in sort_idx]

    fig, axs = plt.subplots(1, 2, figsize=(12 * 0.8*0.8, 3.5 * 0.8*0.9))
    # Each DGVM contributes an annual mean, not twelve monthly replicates.
    model_annual_tac = TAC[:, 0, :].reshape(TAC.shape[0], 19, 12).mean(axis=2)
    tac_i_yr, trendy_ci95_half_width, trendy_n = mean_ci95(model_annual_tac, axis=0)
    mean_val_modis, modis_ci95_half_width, modis_n = mean_ci95(tac2[:, :-1], axis=0)
    axs[0].plot(time_label_modis, tac_i_yr, color='#4393c3', lw=2)
    axs[0].fill_between(time_label_modis, tac_i_yr - trendy_ci95_half_width,
                        tac_i_yr + trendy_ci95_half_width, color='#4393c3', alpha=0.3,
                        label='TRENDY mean 95% CI')
    axs[0].plot(time_label_modis, mean_val_modis, color='k', lw=2)
    axs[0].fill_between(time_label_modis, mean_val_modis - modis_ci95_half_width,
                        mean_val_modis + modis_ci95_half_width, color='k', alpha=0.2,
                        label='MODIS mean 95% CI')
    axs[0].set_xticks([2005,2010,2015,2020],['2005','2010','2015','2020'])
    # Create bar plot with error bars
    y_pos = np.arange(len(feature_names))
    axs[1].barh(y_pos, mean_shap, xerr=sd_shap, align='center',
            color=colors, ecolor='black')
    axs[1].set_yticks(y_pos, feature_names)
    axs[1].set_xlabel('Mean |SHAP value|')
    fig.tight_layout()
    plt.show()
    plt.savefig(current_dir + '/4_Figures/Fig05_resilience_trendy_shap_summary', dpi=900, bbox_inches='tight')

    # After creating the bar plot, add pie chart
    fig_pie = plt.figure(figsize=(2.5, 2.5))
    
    # Group variables by category based on colors
    climate_vars = mean_shap[[i for i, c in enumerate(colors) if c == '#878787']].sum()
    soil_vars = mean_shap[[i for i, c in enumerate(colors) if c == '#d6604d']].sum()
    veg_vars = mean_shap[[i for i, c in enumerate(colors) if c == '#4393c3']].sum()
    
    # Create pie chart data
    sizes = [climate_vars, soil_vars, veg_vars]
    labels = ['Climate', 'Soil', 'Vegetation']
    colors_pie = ['#878787', '#d6604d', '#4393c3']
    plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12})
    plt.tight_layout()
    plt.savefig(current_dir + '/4_Figures/Fig05_resilience_trendy_shap_pie', 
                dpi=900, bbox_inches='tight')
    plt.close()
    
    # Export data from Fig05_resilience_trendy01
    # Get the minimum length between xtick and Trend arrays
    n_models = min(len(xtick), Trend.shape[0])

    # Create DataFrame with truncated arrays
    trendy_data = pd.DataFrame({
        'Model': xtick[:n_models],
        'Trend_2002_2007': Trend[:n_models, 0],
        'Trend_2002_2007_StdDev': Trend[:n_models, 1],
        'Trend_2008_2022': Trend[:n_models, 2],
        'Trend_2008_2022_StdDev': Trend[:n_models, 3]
    })
    trendy_data.to_csv(current_dir + '/4_Figures/Fig05_trendy_trends.csv', index=False)

    # Export data from Fig05_resilience_trendy_shap_summary
    # Panel 1 - Temporal patterns
    temporal_data = pd.DataFrame({
        'Year': time_label_modis,
        'TRENDY_TAC_Anomaly': tac_i_yr, #- tac_i_yr[0],
        'TRENDY_TAC_CI95_HalfWidth': trendy_ci95_half_width,
        'TRENDY_N_Models': trendy_n,
        'MODIS_TAC_Anomaly': mean_val_modis,
        'MODIS_TAC_CI95_HalfWidth': modis_ci95_half_width,
        'MODIS_N_Pixels': modis_n
    })
    temporal_data.to_csv(current_dir + '/4_Figures/Fig.5/Fig05_temporal_patterns_2026.csv', index=False)

    # # Panel 2 - SHAP values
    # shap_data = pd.DataFrame({
    #     'Feature': feature_names,
    #     'Mean_SHAP_Value': mean_shap,
    #     'SHAP_StdDev': sd_shap,
    #     'Category': ['Climate' if c == '#878787' else 'Soil' if c == '#d6604d' else 'Vegetation' for c in colors]
    # })
    # shap_data.to_csv(current_dir + '/4_Figures/Fig05_shap_values.csv', index=False)
    #
    # # Export data from Fig05_resilience_trendy_shap_pie
    # pie_data = pd.DataFrame({
    #     'Category': ['Climate', 'Soil', 'Vegetation'],
    #     'Total_SHAP_Value': [climate_vars, soil_vars, veg_vars],
    #     'Percentage': [s/sum(sizes)*100 for s in sizes]
    # })
    # pie_data.to_csv(current_dir + '/4_Figures/Fig05_category_contributions.csv', index=False)


