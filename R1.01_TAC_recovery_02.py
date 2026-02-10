import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import seaborn as sns
from PIL import Image
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm


# ==========================================
# PART 1: DATA LOADING
# ==========================================

def load_data():
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/NDVI_pf_16d/'

    # --- Load Masks ---
    pf_mask_path2 = current_dir + '/1_Input/landcover_export_2010_5km.tif'
    pf_mask_path3 = current_dir + '/1_Input/landcover_export_2020_5km.tif'

    print("Loading Masks...")
    pf_mask2 = np.array(Image.open(pf_mask_path2)).astype(float)
    pf_mask3 = np.array(Image.open(pf_mask_path3)).astype(float)

    # Consistency checks
    pf_mask2[pf_mask2 != pf_mask3] = np.nan
    pf_mask2[pf_mask2 <= 0] = np.nan
    pf_mask2[pf_mask2 > 12] = np.nan

    # Downsample
    pf_mask = pf_mask2[::3, ::3]
    pf_mask = pf_mask[::-1, ]

    valid_indices = ~np.isnan(pf_mask)
    veg_types_1d = pf_mask[valid_indices]

    print(f"Valid pixels: {len(veg_types_1d)}")

    # --- Load EVI & Calc kNDVI ---
    evi_filenames = os.listdir(EVI_folder)
    evi_filenames = np.sort(evi_filenames)

    kndvi_list = []
    print("Processing kNDVI...")

    for name in tqdm(evi_filenames):
        if not name.endswith('.tif'): continue
        path = os.path.join(EVI_folder, name)
        with rasterio.open(path) as src:
            evi = src.read(1)[::3, ::3]

        evi[evi < 0] = 0
        evi_val = evi[valid_indices]  # Extract valid pixels

        # kNDVI Calculation
        kndvi = np.tanh(evi_val ** 2)
        kndvi_list.append(kndvi)

    # Shape: (Pixels, TimeSteps)
    kndvi_arr = np.array(kndvi_list).T

    return kndvi_arr, veg_types_1d
kndvi_data, veg_types = load_data()

# ==========================================
# PART 2: RESISTANCE / RECOVERY / TAC LOGIC
# ==========================================

def calculate_resilience_metrics(kndvi_arr, veg_types,bands_per_year=23,gs_start=9, gs_end=19, norm_window=3,period_dt=0):  # Years to use for "Normal" baseline

    n_pixels, n_timesteps = kndvi_arr.shape
    n_years = n_timesteps // bands_per_year

    print(f"\nReshaping Data: {n_years} years, {bands_per_year} bands/year")

    # 1. Reshape to 3D (Pixel, Year, Band)
    data_3d = kndvi_arr[:, :n_years * bands_per_year].reshape(n_pixels, n_years, bands_per_year)

    # 2. Extract Growing Season (GS) 16-day data
    gs_data_high_freq = data_3d[:, :, gs_start:gs_end]  # Shape: (Pix, Year, n_gs_bands)

    # 3. Calculate Yearly Growing Season Means (for Disturbance Detection)
    yearly_gs_mean0 = np.nanmean(gs_data_high_freq, axis=2)  # Shape: (Pix, Year)
    df_deseas = pd.DataFrame(yearly_gs_mean0.T)

    # Calculate Rolling Mean (Trend)
    # center=True ensures the trend is aligned with the event
    # min_periods=1 ensures edges are handled
    ROLLING_WINDOW = 7
    rolling_trend = df_deseas.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()

    yearly_gs_mean = yearly_gs_mean0 - rolling_trend.values.T+np.nanmean(yearly_gs_mean0, axis=1, keepdims=True)
    # plt.figure(); plt.plot(yearly_gs_mean[366,:])

    # 4. Calculate Climatology (for Deseasonalizing AR1)
    # Mean and Std of each 16-day band across all years
    clim_mean = np.nanmean(gs_data_high_freq, axis=1)  # (Pix, n_gs_bands)
    clim_std = np.nanstd(gs_data_high_freq, axis=1)

    # Calculate Anomalies for the whole dataset (used for AR1)
    # (Value - BandMean) / BandStd
    anoms_3d = (gs_data_high_freq - clim_mean[:, None, :]) / clim_std[:, None, :]

    # --- MAIN LOOP ---
    results = []

    print("Detecting Disturbances and Calculating Metrics...")

    for i in tqdm(range(n_pixels)):
        ts_yearly = yearly_gs_mean[i, :]
        if np.nanmean(ts_yearly)<0.1: continue

        # Skip empty pixels
        if np.isnan(ts_yearly).sum() > 5: continue

        # Define Threshold: Mean - 1.5 * Std (of the specific pixel's history)
        # Using long-term stats for detection threshold
        lt_mean = np.nanmean(ts_yearly)
        lt_std = np.nanstd(ts_yearly)

        # Find Disturbance Years (Indices)
        dist_candidates = np.where(ts_yearly < (lt_mean - 2.0 * lt_std))[0]

        for yr_idx in dist_candidates:
            # Bounds Check
            if yr_idx < norm_window or yr_idx >= (n_years - 5):
                continue

            # Define VI values
            VI_in = ts_yearly[yr_idx]
            VI_next = ts_yearly[yr_idx + 1]

            # Calculate VI_norm (Average of previous 'norm_window' years)
            prev_years = ts_yearly[yr_idx - norm_window: yr_idx]
            VI_norm = np.nanmean(prev_years)

            # --- FORMULA 1: RESISTANCE ---
            # R = VI_norm / |VI_norm - VI_in|
            # Avoid division by zero if drop is tiny
            diff_in = abs(VI_norm - VI_in)
            if (diff_in < 1e-3*5) | (diff_in > 0.2): continue
            resistance = VI_norm / diff_in

            # --- FORMULA 2: RECOVERY ---
            # Rec = |VI_in - VI_norm| / |VI_next - VI_norm|
            # Note: Higher value = Better recovery (denominator gets smaller)
            diff_next = abs(VI_next - VI_norm)
            if (diff_next < 1e-3*5) | (diff_next > 0.15)|(diff_in < 1e-3*5) | (diff_in > 0.2) | (diff_in<diff_next): continue
            else:
                recovery = diff_in / diff_next
                # recovery, intercept = np.polyfit(np.arange(5), ts_yearly[yr_idx: yr_idx+5], 1)

            # --- FORMULA 3: TAC (AR1) ---
            pre_anoms = anoms_3d[i, :].flatten()

            # Clean NaNs
            pre_anoms = pre_anoms[~np.isnan(pre_anoms)]

            if len(pre_anoms) < (norm_window * 3):  # Require minimal data
                continue

            # Pearson R lag-1
            tac, _ = stats.pearsonr(pre_anoms[:-1], pre_anoms[1:])

            # --- SAVE RESULTS ---
            real_year = 2000 + yr_idx  # Assuming data starts 2000

            if real_year <2006+period_dt: period_str='Pre-2008'
            elif real_year >2010-period_dt: period_str='Post-2008'
            else: period_str='Near-2008'

            results.append({
                'VegType': veg_types[i],
                'Year': real_year,
                'Period1': period_str,
                'VI_norm': VI_norm,
                'VI_in': VI_in,
                'TAC': tac,
                'Resistance': resistance,
                'Recovery': recovery
            })

    return pd.DataFrame(results)
df_res = calculate_resilience_metrics(kndvi_data, veg_types,
                                      bands_per_year=23,
                                      gs_start=9,  # approx May 25
                                      gs_end=19,period_dt=0)  # approx Oct 15

if not df_res.empty:    # Map VegType to grouped categories
    def map_veg(v):
        try:
            vi = int(v)
        except Exception:
            return 'Other'
        if vi in (1, 3):
            return 'NF'
        if vi in (4, 5):
            return 'MF'
        if vi in (6, 7):
            return 'SAV'
        if vi in (8, 9):
            return 'SHR'
        if vi == 10:
            return 'GRA'
        return 'Other'

    df_res['VegGroup'] = df_res['VegType'].apply(map_veg)
    veg_groups = ['All', 'NF', 'MF', 'SAV', 'SHR', 'GRA']

    periods = ['Pre-2008', 'Near-2008', 'Post-2008']
    stats_list_res = []
    stats_list_rec = []
    for p in periods:
        sub = df_res[df_res['Period1'] == p]
        stats_list_res.append((sub['Resistance'].mean(), sub['Resistance'].std()))
        stats_list_rec.append((sub['Recovery'].mean(), sub['Recovery'].std()))

    fig, axes = plt.subplots(2, 2, figsize=(8*0.8, 7*0.8))
    plt.style.use("seaborn-v0_8-whitegrid")

    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    palette = sns.color_palette("Set2")

    x = np.arange(len(periods))

    def plot_with_errors(ax, x, means, errs, title, ylabel, color):
        ax.plot(x, means, marker='o', color=color, lw=2, markersize=7)
        ax.errorbar(x, means, yerr=errs, fmt='none', ecolor='k', elinewidth=1.5, capsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=25)
        # ax.set_title(title, pad=6)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        means = np.array(means)
        errs = np.array(errs)
        ax.set_ylim([min(means-errs)*0.9,max(means+errs)*1.1])
        ax.set_xlim([-0.5,2.5])
        # annotate sample sizes
        # for xi, m in zip(x, means):
        #     ax.annotate(f"{int(xi)}", (xi, m), textcoords="offset points", xytext=(0,6), ha='center', fontsize=9)

    # First run (period_dt=0) — top row
    means_res = [m for m, s in stats_list_res]
    errs_res = [0.25 * s for m, s in stats_list_res]
    plot_with_errors(axes[0,0], x, means_res, errs_res, "Resistance (period_dt=0)", "Resistance", palette[0])

    means_rec = [m for m, s in stats_list_rec]
    errs_rec = [0.25 * s for m, s in stats_list_rec]
    plot_with_errors(axes[0,1], x, means_rec, errs_rec, "Recovery rate (period_dt=0)", "Recovery rate", palette[1])

    # Recompute & plot bottom row (period_dt=1)
    df_res = calculate_resilience_metrics(kndvi_data, veg_types,
                                          bands_per_year=23,
                                          gs_start=9,  # approx May 25
                                          gs_end=19,period_dt=1)  # approx Oct 15
    df_res['VegGroup'] = df_res['VegType'].apply(map_veg)

    stats_list_res = []
    stats_list_rec = []
    for p in periods:
        sub = df_res[df_res['Period1'] == p]
        stats_list_res.append((sub['Resistance'].median(), sub['Resistance'].std()))
        stats_list_rec.append((sub['Recovery'].median(), sub['Recovery'].std()))

    means_res = [m for m, s in stats_list_res]
    errs_res = [0.25 * s for m, s in stats_list_res]
    plot_with_errors(axes[1,0], x, means_res, errs_res, "Resistance (period_dt=1)", "Resistance", palette[2])

    means_rec = [m for m, s in stats_list_rec]
    errs_rec = [0.25 * s for m, s in stats_list_rec]
    plot_with_errors(axes[1,1], x, means_rec, errs_rec, "Recovery rate (period_dt=1)", "Recovery rate", palette[3])

    # Layout and save
    plt.suptitle("Resilience metrics — Before / After adjustment", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    figToPath = current_dir + '/4_Figures/FigR01_recovery_resistance-before-after'
    plt.savefig(figToPath + '.png', dpi=600, bbox_inches='tight')
    plt.savefig(figToPath + '.svg', dpi=300, bbox_inches='tight')
    plt.close()