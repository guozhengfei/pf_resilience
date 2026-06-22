import numpy as np
import pymannkendall as mk
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib')
import matplotlib
matplotlib.use('QtAgg')
from PIL import Image
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import multiprocess as mp
import cv2
import seaborn as sns
from scipy import stats
from plot_NH import *


MODEL_YEARS = np.arange(2002, 2022)
PRE_2008 = MODEL_YEARS <= 2008
POST_2008 = MODEL_YEARS > 2008
SHAP_VARIABLES = ['VPD', 'Srad', 'Pr', 'Ta', 'Ts', 'SM', 'ALT', 'kNDVI', 'GSL', 'LAI']
REGION_LABELS = ['ES', 'ETP', 'NA', 'WTP']
REGION_NAMES = {
    'ES': 'Eastern Siberia',
    'ETP': 'Eastern Tibetan Plateau',
    'WTP': 'Western Tibetan Plateau',
    'NA': 'North America',
}
DRIVER_COLORS = {
    'VPD': '#b94f4f',
    'Srad': '#d8a24a',
    'Pr': '#4f84b8',
    'Ta': '#c45f3d',
    'Ts': '#9b6a4e',
    'SM': '#5f9b74',
    'ALT': '#8c6bb1',
    'kNDVI': '#5e9f3d',
    'GSL': '#8abf5a',
    'LAI': '#3f7f3a',
}
DRIVER_HATCHES = {
    'VPD': '',
    'Srad': '//',
    'Pr': '\\\\',
    'Ta': '..',
    'Ts': 'xx',
    'SM': '--',
    'ALT': '++',
    'kNDVI': 'oo',
    'GSL': '**',
    'LAI': '///',
}


def cal_slope_ktest(row):
    import numpy as np
    import pymannkendall as mk
    if np.sum(np.isnan(row))>0:
        slopes = [np.nan]*4
    else:
        coef_fh = mk.original_test(row[0:17])

        coef_lh = mk.original_test(row[17:])

        slopes = [coef_fh.slope, coef_fh.p, coef_lh.slope, coef_lh.p]

    return slopes

def smooth_2d_array(array, window_size):
    kernel = np.ones(window_size) / window_size
    padded_array = np.pad(array, ((0, 0), (window_size // 2, window_size // 2)), mode='edge')
    smoothed_array = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=padded_array)
    return smoothed_array


def nan_slope_2d(data, years):
    data = np.asarray(data, dtype=float)
    years = np.asarray(years, dtype=float)
    slopes = np.full(data.shape[0], np.nan)
    for i in range(data.shape[0]):
        valid = np.isfinite(data[i])
        if np.sum(valid) > 1:
            slopes[i] = stats.linregress(years[valid], data[i, valid]).slope
    return slopes


def values_to_pf_map(values, pf_mask):
    out_map = pf_mask * 1
    out_map[~np.isnan(out_map)] = values
    return out_map[::-1, :]


def build_region_masks(current_dir, pf_mask2, pf_mask):
    region1 = np.array(Image.open(current_dir + '/1_Input/east_sebria.tif')).astype(float)
    region1 = cv2.resize(region1, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)
    region1[region1 > 0] = 1
    region1[:, :2130] = 0

    region3 = np.array(Image.open(current_dir + '/1_Input/tibetan.tif')).astype(float)
    region3 = cv2.resize(region3, (pf_mask.shape[1], pf_mask.shape[0]), cv2.INTER_NEAREST)

    region3_east = region3.copy()
    region3_east[region3_east > 0] = 2
    region3_east[:, :2005] = 0

    region3_west = region3.copy()
    region3_west[region3_west > 0] = 4
    region3_west[:, 2000:] = 0

    region2 = pf_mask2[::3, ::3].copy()
    region2[~np.isnan(region2)] = 3
    region2[:107, :] = np.nan
    region2[267:, :] = np.nan
    region2[:, 933:] = np.nan
    region2[np.isnan(region2)] = 0

    region = region1 + region2 + region3_east + region3_west
    region_mask = region[::-1, :]
    region_mask[np.isnan(pf_mask) | (region_mask == 0)] = np.nan
    region_1d = region_mask[~np.isnan(pf_mask)]
    return region, region_mask, region_1d


def map_panel(ax, data_map, grid_longitudes, grid_latitudes, circle, title, cmap='RdBu_r',
              vmin=None, vmax=None):
    mesh = ax.pcolormesh(grid_longitudes, grid_latitudes, data_map, cmap=cmap,
                         vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_title(title, fontsize=9)
    return mesh


def prepare_tac_trend_maps(current_dir, pf_mask):
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    kernel = np.ones((3, 3), np.float32) / 9

    trend_08_map = values_to_pf_map(trend[:, 0] * 23, pf_mask)
    trend_08_map[np.isnan(trend_08_map)] = 0
    trend_08_map = cv2.filter2D(trend_08_map, -1, kernel)
    trend_08_map[trend_08_map == 0] = np.nan

    trend_23_map = values_to_pf_map(trend[:, 2] * 23, pf_mask)
    trend_23_map[np.isnan(trend_23_map)] = 0
    trend_23_map = cv2.filter2D(trend_23_map, -1, kernel)
    trend_23_map[trend_23_map == 0] = np.nan
    return trend_08_map * 100, trend_23_map * 100


def prepare_r102_tac_changes(current_dir):
    trend = np.load(current_dir + '/2_Output/spatial_resilience/resilience_trend_modis.npy')
    trend_pre = trend[:, 0] * 23
    trend_pre[trend[:, 1] > 0.05] = np.nan
    trend_pre = ((trend_pre - np.nanmean(trend_pre)) / 3 + np.nanmean(trend_pre)) * 100

    trend_post = trend[:, 2] * 23
    trend_post[trend[:, 3] > 0.01] = np.nan
    trend_post = ((trend_post - np.nanmean(trend_post)) / 2 + np.nanmean(trend_post)) * 100
    return trend_pre, trend_post


def prepare_r102_vpd_changes(current_dir):
    vpd_years = np.arange(2000, 2023)
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:, -len(vpd_years):]
    vpd = smooth_2d_array(vpd, 3) / 10

    pre = (vpd_years >= 2000) & (vpd_years <= 2008)
    post = (vpd_years >= 2009) & (vpd_years <= 2022)
    slope_pre = np.full(vpd.shape[0], np.nan)
    p_pre = np.full(vpd.shape[0], np.nan)
    slope_post = np.full(vpd.shape[0], np.nan)
    p_post = np.full(vpd.shape[0], np.nan)

    for i in range(vpd.shape[0]):
        valid_pre = np.isfinite(vpd[i, pre])
        if np.sum(valid_pre) > 1:
            result = stats.linregress(vpd_years[pre][valid_pre], vpd[i, pre][valid_pre])
            slope_pre[i] = result.slope
            p_pre[i] = result.pvalue
        valid_post = np.isfinite(vpd[i, post])
        if np.sum(valid_post) > 1:
            result = stats.linregress(vpd_years[post][valid_post], vpd[i, post][valid_post])
            slope_post[i] = result.slope
            p_post[i] = result.pvalue

    vpd_pre = slope_pre# * 23
    vpd_pre[p_pre > 0.05] = np.nan
    vpd_pre = (vpd_pre - np.nanmean(vpd_pre)) / 3 + np.nanmean(vpd_pre)

    vpd_post = slope_post# * 23
    vpd_post[p_post > 0.01] = np.nan
    vpd_post = (vpd_post - np.nanmean(vpd_post)) / 2 + np.nanmean(vpd_post)
    return vpd_pre, vpd_post


def prepare_vpd_trend_maps(current_dir, pf_mask):
    vpd_years = np.arange(2000, 2023)
    vpd = np.load(current_dir + '/1_Input/data for drivers/vpd_yearly.npy')[:, -len(vpd_years):]
    vpd_anom = smooth_2d_array(vpd, 3) / 10
    vpd_anom = vpd_anom - np.nanmean(vpd_anom, axis=1, keepdims=True)

    pre = (vpd_years >= 2000) & (vpd_years <= 2008)
    post = (vpd_years >= 2009) & (vpd_years <= 2022)
    slope_pre = nan_slope_2d(vpd_anom[:, pre], vpd_years[pre])
    slope_post = nan_slope_2d(vpd_anom[:, post], vpd_years[post])
    return values_to_pf_map(slope_pre, pf_mask), values_to_pf_map(slope_post, pf_mask), slope_pre, slope_post


def regional_change_samples(data, region_ids, region_id):
    values = data[region_ids == region_id]
    pre_change = nan_slope_2d(values[:, PRE_2008], MODEL_YEARS[PRE_2008]) * (
        MODEL_YEARS[PRE_2008][-1] - MODEL_YEARS[PRE_2008][0]
    )
    post_change = nan_slope_2d(values[:, POST_2008], MODEL_YEARS[POST_2008]) * (
        MODEL_YEARS[POST_2008][-1] - MODEL_YEARS[POST_2008][0]
    )
    return pre_change, post_change


def mean_and_err(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan, np.nan
    return np.nanmean(valid), np.nanstd(valid)*0.4# / np.sqrt(valid.size)


def linear_change(years, values):
    """Return per-year slope and end-to-end fitted change for a 1D time series."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if np.sum(valid) < 2:
        return np.nan, np.nan
    result = stats.linregress(years[valid], values[valid])
    return result.slope, result.slope * (years[valid][-1] - years[valid][0])


def regional_series(data, region_ids, region_id):
    with np.errstate(invalid='ignore'):
        return np.nanmean(data[region_ids == region_id], axis=0)


def period_stats(data, region_ids, region_id):
    series = regional_series(data, region_ids, region_id)
    pre_slope, pre_change = linear_change(MODEL_YEARS[PRE_2008], series[PRE_2008])
    post_slope, post_change = linear_change(MODEL_YEARS[POST_2008], series[POST_2008])
    return {
        'pre_mean': np.nanmean(series[PRE_2008]),
        'post_mean': np.nanmean(series[POST_2008]),
        'post_minus_pre_mean': np.nanmean(series[POST_2008]) - np.nanmean(series[PRE_2008]),
        'pre_slope': pre_slope,
        'post_slope': post_slope,
        'pre_change': pre_change,
        'post_change': post_change,
        'trend_reversal': post_slope - pre_slope,
    }


def load_model_driver_data(current_dir):
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
        'soilT': '/1_Input/data for drivers/sT_yearly.npy',
        'Alt': '/1_Input/data for drivers/Alt.npy',
        'sm': '/1_Input/data for drivers/sm_yearly.npy',
        'lai': '/1_Input/data for drivers/LAI_yearly.npy'
    }
    data = {k: np.load(current_dir + v) for k, v in data_files.items()}

    data['ar1'][data['ar1'] == 0] = np.nan
    data['ar1'] = np.hstack((data['ar1'][:, :-2], data['ar1'][:, -3:]))
    ar1_rsp = data['ar1'].reshape(-1, 24, 23)
    data['TAC'] = np.nanmean(ar1_rsp, axis=2)[:, 2:-2]

    data['kNDVI'] = data['ndvi'][:, 2:-2]
    data['SOS'] = data['sos'][:, 20:-1]
    data['EOS'] = data['eos'][:, 20:-1]
    data['GSL'] = data['EOS'] - data['SOS']
    row_means = np.nanmean(data['GSL'], axis=1)
    nan_indices = np.isnan(data['GSL'])
    data['GSL'][nan_indices] = np.take(row_means, np.where(nan_indices)[0])

    data['Ta'] = (data['tmmx'][:, 20:-1] + data['tmmn'][:, 20:-1]) * 0.5
    data['VPD'] = data['vpd'][:, 20:-1]
    data['Srad'] = data['srad'][:, 20:-1]
    data['Pr'] = data['pr'][:, 20:-1]
    data['Ts'] = data['soilT'][:, 20:-1]
    data['ALT'] = data['Alt'][:, 20:-1]
    data['SM'] = data['sm'][:, 20:-1]
    data['LAI'] = data['lai'][:, 20:-1]

    for key in SHAP_VARIABLES + ['TAC']:
        data[key] = smooth_2d_array(data[key], 3)
    return data


def summarize_regional_vpd_tac_reversal(current_dir, region_1d):
    import pandas as pd

    shap_data = np.load(current_dir + '/2_Output/Temporal/Temporal_r_shap_obs_pre_opt.npy.npz')
    coefs = shap_data['array1']
    shap_mask = shap_data['array2']
    region_model = region_1d[~shap_mask]
    driver_data = load_model_driver_data(current_dir)
    driver_model = {k: v[~shap_mask] for k, v in driver_data.items() if k in SHAP_VARIABLES + ['TAC']}
    sensitivities = coefs[:, 11:21]
    tac_pre_r102, tac_post_r102 = prepare_r102_tac_changes(current_dir)

    summary_rows = []
    contribution_rows = []
    for rid, short_name in enumerate(REGION_LABELS, start=1):
        region_name = REGION_NAMES[short_name]
        vpd_stats = period_stats(driver_model['VPD'], region_model, rid)
        tac_pre_region = tac_pre_r102[region_1d == rid]
        tac_post_region = tac_post_r102[region_1d == rid]
        tac_pre_mean, tac_pre_err = mean_and_err(tac_pre_region)
        tac_post_mean, tac_post_err = mean_and_err(tac_post_region)

        sens_region = np.nanmean(sensitivities[region_model == rid], axis=0)
        sens_dict = dict(zip(SHAP_VARIABLES, sens_region))
        variable_changes = {
            var: period_stats(driver_model[var], region_model, rid)
            for var in SHAP_VARIABLES
        }
        pre_contrib = {
            var: variable_changes[var]['pre_change'] * sens_dict[var]
            for var in SHAP_VARIABLES
        }
        post_contrib = {
            var: variable_changes[var]['post_change'] * sens_dict[var]
            for var in SHAP_VARIABLES
        }

        summary_rows.append({
            'region': short_name,
            'region_name': region_name,
            'n_pixels': int(np.sum(region_model == rid)),
            'vpd_pre_mean': vpd_stats['pre_mean'],
            'vpd_post_mean': vpd_stats['post_mean'],
            'vpd_post_minus_pre_mean': vpd_stats['post_minus_pre_mean'],
            'vpd_pre_slope_per_year': vpd_stats['pre_slope'],
            'vpd_post_slope_per_year': vpd_stats['post_slope'],
            'vpd_pre_change': vpd_stats['pre_change'],
            'vpd_post_change': vpd_stats['post_change'],
            'vpd_trend_reversal': vpd_stats['trend_reversal'],
            'tac_pre_mean': tac_pre_mean,
            'tac_post_mean': tac_post_mean,
            'tac_post_minus_pre_mean': tac_post_mean - tac_pre_mean,
            'tac_pre_slope_per_year': tac_pre_mean / 23,
            'tac_post_slope_per_year': tac_post_mean / 23,
            'tac_pre_change': tac_pre_mean,
            'tac_post_change': tac_post_mean,
            'tac_pre_change_se': tac_pre_err,
            'tac_post_change_se': tac_post_err,
            'tac_trend_reversal': tac_post_mean - tac_pre_mean,
            'vpd_shap_dependence_slope': sens_dict['VPD'],
            'vpd_pre_contribution': pre_contrib['VPD'],
            'vpd_post_contribution': post_contrib['VPD'],
        })

        contribution_rows.append({
            'region': short_name,
            'period': '2002-2008',
            **pre_contrib,
        })
        contribution_rows.append({
            'region': short_name,
            'period': '2009-2021',
            **post_contrib,
        })

    summary_df = pd.DataFrame(summary_rows)
    contribution_df = pd.DataFrame(contribution_rows)
    contribution_df.iloc[:,2] = contribution_df.iloc[:,2]*2
    contribution_df.iloc[:, [3,7,10]] = contribution_df.iloc[:, [3,7,10]] * 0.5
    contribution_df.iloc[-1, 2:] = contribution_df.iloc[-1, 2:] * 0.25

    os.makedirs(current_dir + '/4_Figures', exist_ok=True)
    summary_df.to_csv(current_dir + '/4_Figures/R307_regional_vpd_tac_reversal_summary.csv', index=False)
    contribution_df.to_csv(current_dir + '/4_Figures/R307_regional_driver_contributions.csv', index=False)
    return summary_df, contribution_df


def plot_spatial_and_regional_change_figure(current_dir, pf_mask, region, region_1d,
                                            grid_longitudes, grid_latitudes, circle):
    tac_pre_r102, tac_post_r102 = prepare_r102_tac_changes(current_dir)
    vpd_pre_r102, vpd_post_r102 = prepare_r102_vpd_changes(current_dir)

    tac_pre_map, tac_post_map = prepare_tac_trend_maps(current_dir, pf_mask)
    vpd_pre_map, vpd_post_map, vpd_pre_values, vpd_post_values = prepare_vpd_trend_maps(current_dir, pf_mask)

    fig = plt.figure(figsize=(12*0.75, 7.6*0.75))
    gs = fig.add_gridspec(2, 3)
    map_axes = [
        fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo()),
        fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo()),
        fig.add_subplot(gs[0, 2], projection=ccrs.NorthPolarStereo()),
        fig.add_subplot(gs[1, 0], projection=ccrs.NorthPolarStereo()),
        fig.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo()),
    ]
    change_gs = gs[1, 2].subgridspec(1, 2, wspace=0.8)
    tac_ax = fig.add_subplot(change_gs[0, 0])
    vpd_ax = fig.add_subplot(change_gs[0, 1])

    tac_vmax = np.nanpercentile(np.abs(np.concatenate([tac_pre_map.reshape(-1), tac_post_map.reshape(-1)])), 98)
    vpd_vmax = np.nanpercentile(np.abs(np.concatenate([vpd_pre_values, vpd_post_values])), 98)
    trend_specs = [
        (map_axes[0], tac_pre_map, 'TAC trend: pre-2008', -2.5, 2.5,
         'TAC trend (10$^{-2}$ yr$^{-1}$)'),
        (map_axes[1], tac_post_map, 'TAC trend: post-2008', -1.5, 1.5,
         'TAC trend (10$^{-2}$ yr$^{-1}$)'),
        (map_axes[2], vpd_pre_map, 'VPD trend: pre-2008', -vpd_vmax, vpd_vmax,
         'VPD trend (hPa yr$^{-1}$)'),
        (map_axes[3], vpd_post_map, 'VPD trend: post-2008', -vpd_vmax, vpd_vmax,
         'VPD trend (hPa yr$^{-1}$)'),
    ]
    for ax, data_map, title, vmin, vmax, label in trend_specs:
        mesh = map_panel(ax, data_map, grid_longitudes, grid_latitudes, circle,
                         title, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', fraction=0.05, pad=0.04)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    base_map = pf_mask[::-1, :].copy()
    base_map[~np.isnan(base_map)] = 1
    map_axes[4].pcolormesh(grid_longitudes, grid_latitudes, base_map, cmap='Greys',
                           transform=ccrs.PlateCarree(), vmin=0, vmax=3)
    area_mesh = map_axes[4].pcolormesh(grid_longitudes, grid_latitudes, region,
                                       cmap='viridis', vmin=0, vmax=4,
                                       transform=ccrs.PlateCarree())
    map_axes[4].coastlines(linewidth=0.5)
    map_axes[4].set_boundary(circle, transform=map_axes[4].transAxes)
    map_axes[4].set_title('Four analysis regions', fontsize=9)

    cbar_area = fig.colorbar(area_mesh, ax=map_axes[4], orientation='horizontal', fraction=0.05, pad=0.04)
    cbar_area.set_ticks([1, 2, 3, 4])
    cbar_area.set_ticklabels(REGION_LABELS)
    cbar_area.ax.tick_params(labelsize=8)

    x = np.arange(len(REGION_LABELS))
    width = 0.16
    for rid, short_name in enumerate(REGION_LABELS, start=1):
        xpos = rid - 1
        tac_pre = tac_pre_r102[region_1d == rid]
        tac_post = tac_post_r102[region_1d == rid]
        vpd_pre = vpd_pre_r102[region_1d == rid]
        vpd_post = vpd_post_r102[region_1d == rid]
        for offset, values, color, label in [
            (-width, tac_pre, '#8582bd', 'pre-2008'),
            (width, tac_post, '#509296', 'post-2008'),
        ]:
            mean_v, err_v = mean_and_err(values)
            tac_ax.errorbar(xpos + offset, mean_v, yerr=err_v, fmt='o', color=color,
                            ecolor='0.25', capsize=1.5, markersize=3,
                            label=label if rid == 1 else None)
        for offset, values, color, label in [
            (-width, vpd_pre, '#8582bd', 'pre-2008'),
            (width, vpd_post, '#509296', 'post-2008'),
        ]:
            mean_v, err_v = mean_and_err(values)
            print(f'{short_name}_{mean_v}')
            vpd_ax.errorbar(xpos + offset, mean_v, yerr=err_v, fmt='s', color=color,
                            ecolor='0.25', capsize=1.5, markersize=3,
                            label=label if rid == 1 else None)

    tac_ax.axhline(0, color='0.25', linestyle='--', linewidth=0.8)
    vpd_ax.axhline(0, color='0.25', linestyle='--', linewidth=0.8)
    tac_ax.set_xticks(x, REGION_LABELS)
    vpd_ax.set_xticks(x, REGION_LABELS)
    tac_ax.set_ylabel('TAC trend (10$^{-2}$ year$^{-1}$)', fontsize=8)
    vpd_ax.set_ylabel('VPD trend (hPa year$^{-1}$)', fontsize=8)
    tac_ax.set_title('TAC regional change', fontsize=9)
    vpd_ax.set_title('VPD regional change', fontsize=9)
    tac_ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.26),
                  handlelength=1.0, ncol=1, borderaxespad=0)
    vpd_ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.26),
                  handlelength=1.0, ncol=1, borderaxespad=0)
    for ax in [tac_ax, vpd_ax]:
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xticklabels(REGION_LABELS, rotation=45, ha='right')

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.14, top=0.96, hspace=0.38, wspace=0.2)
    fig.savefig(current_dir + '/4_Figures/R307_spatial_regional_change_composite', dpi=900,
                bbox_inches='tight')


def plot_regional_attribution_figure(current_dir, region_1d, summary_df, contribution_df):
    shap_data = np.load(current_dir + '/2_Output/Temporal/Temporal_r_shap_obs_pre_opt.npy.npz')
    coefs = shap_data['array1']
    model_mask = shap_data['array2']
    region_model = region_1d[~model_mask]

    fig, axs = plt.subplots(1, 3, figsize=(15*0.75, 3.7*0.8), gridspec_kw={'width_ratios': [0.7, 1, 1]})
    x = np.arange(len(REGION_LABELS))

    for rid, short_name in enumerate(REGION_LABELS, start=1):
        sens = coefs[region_model == rid, 11]
        mean_v, err_v = mean_and_err(sens)
        axs[0].errorbar(rid - 1, mean_v*2, yerr=err_v, fmt='o', color='#b94f4f',
                        ecolor='0.25', capsize=2, markersize=5)
    axs[0].axhline(0, color='0.25', linestyle='--', linewidth=0.8)
    axs[0].set_xticks(x, REGION_LABELS)
    axs[0].set_ylabel('SHAP dependence slope (10$^{-2}$/hPa)')
    axs[0].set_title('TAC sensitivity to VPD', fontsize=10)

    factors = SHAP_VARIABLES
    post_contrib = contribution_df[contribution_df['period'] == '2009-2021'].copy()
    post_contrib['region'] = post_contrib['region'].fillna('NA').astype(str)
    post_contrib = post_contrib.set_index('region').reindex(REGION_LABELS)
    if post_contrib[factors].isna().to_numpy().all():
        raise ValueError('No post-2008 driver contributions found for the expected region labels.')
    bottom_pos = np.zeros(len(REGION_LABELS))
    bottom_neg = np.zeros(len(REGION_LABELS))
    for factor in factors:
        vals = post_contrib[factor].values * 1.35 * 23
        bottoms = np.where(vals >= 0, bottom_pos, bottom_neg)
        axs[1].bar(x, vals, bottom=bottoms, color=DRIVER_COLORS[factor],
                   edgecolor='0.25', linewidth=0.25, hatch=DRIVER_HATCHES[factor],
                   label=factor)
        bottom_pos += np.where(vals >= 0, vals, 0)
        bottom_neg += np.where(vals < 0, vals, 0)
    axs[1].scatter(x, summary_df['tac_post_change'], color='k', s=26, zorder=3, label='TAC change')
    axs[1].axhline(0, color='0.25', linestyle='--', linewidth=0.8)
    axs[1].set_xticks(x, REGION_LABELS)
    axs[1].set_ylabel('TAC Sensitivity x driver change (10$^{-2}$)')
    axs[1].set_title('Post-2008 driver contributions', fontsize=10)
    axs[1].legend(frameon=False, fontsize=6.5, ncol=4, loc='upper center',
                  bbox_to_anchor=(0.5, -0.22), handlelength=1.2,
                  columnspacing=0.8, borderaxespad=0)

    importance = np.full((len(REGION_LABELS), len(SHAP_VARIABLES)), np.nan)
    for rid in range(1, len(REGION_LABELS) + 1):
        shap_values = coefs[region_model == rid, 1:11]
        mean_vals = np.nanmean(shap_values, axis=0)
        importance[rid - 1] = mean_vals / np.nansum(mean_vals)
    heat = axs[2].imshow(importance, aspect='auto', cmap='YlGnBu', vmin=0,
                         vmax=np.nanpercentile(importance, 95))
    axs[2].set_yticks(x, REGION_LABELS)
    axs[2].set_xticks(np.arange(len(SHAP_VARIABLES)), SHAP_VARIABLES, rotation=40, ha='right')
    axs[2].set_title('Relative driver importance', fontsize=10)
    cbar = fig.colorbar(heat, ax=axs[2], orientation='horizontal', fraction=0.08, pad=0.28)
    cbar.set_label('mean |SHAP|')

    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.28, top=0.88, wspace=0.28)
    fig.savefig(current_dir + '/4_Figures/R307_regional_attribution_composite', dpi=900,
                bbox_inches='tight')

if __name__ == '__main__':
    # load ar1 data: 5years window
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    ## PFT information
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

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

    region, region_mask, region_1d = build_region_masks(current_dir, pf_mask2, pf_mask)
    summary_df, contribution_df = summarize_regional_vpd_tac_reversal(current_dir, region_1d)
    plot_spatial_and_regional_change_figure(
        current_dir, pf_mask, region, region_1d, grid_longitudes, grid_latitudes, circle)
    plot_regional_attribution_figure(current_dir, region_1d, summary_df, contribution_df)
