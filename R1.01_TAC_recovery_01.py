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

def calculate_resilience_metrics(kndvi_arr, veg_types,bands_per_year=23,gs_start=9, gs_end=19, norm_window=3):  # Years to use for "Normal" baseline

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
            if real_year <2006: period_str='Pre-2008'
            elif real_year >2010: period_str='Post-2008'
            else: period_str='Near-2008'

            results.append({
                'VegType': veg_types[i],
                'Year': real_year,
                'Period': period_str,
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
                                      gs_end=19)  # approx Oct 15
# df_res = df_res[df_res['TAC']>0.05]

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

    # --- FIGURE 1: TAC vs Resistance (6 subplots: overall + each veg group) ---
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8),sharey=True)
    axes = axes.flatten()
    for ax_idx, vg in enumerate(veg_groups):
        ax = axes[ax_idx]
        if vg == 'All':
            subset = df_res.copy()
        else:
            subset = df_res[df_res['VegGroup'] == vg]

        # Density background + scatter
        ax.scatter(subset['TAC'], subset['Resistance'], s=6, color='C0', alpha=0.3)

        sns.kdeplot(x=subset['TAC'], y=subset['Resistance'], fill=True, cmap="RdBu_r", thresh=0.05, ax=ax,
                    alpha=0.4)

        # Linear fit (only use finite points)
        mask = np.isfinite(subset['TAC']) & np.isfinite(subset['Resistance'])
        if mask.sum() > 1:  # 线性拟合至少需要2个点
            xdata = subset.loc[mask, 'TAC'].values
            ydata = subset.loc[mask, 'Resistance'].values
            print(xdata.shape[0]*9,stats.linregress(xdata,ydata).rvalue)

            try:
                slope, intercept = np.polyfit(xdata, ydata, 1)
                xs = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 200)
                ys = slope * xs + intercept
                ax.plot(xs, ys, color='black', lw=2, label='Linear fit')
            except Exception as e:
                print(f"Linear fit failed: {e}")

        xlabel = 'TAC$_{ED}$ of '+vg
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Resistance')
        ax.set_ylim([0,15])

    plt.tight_layout()
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    figToPath = current_dir + '/4_Figures/FigR01_resistance_TAC'
    plt.savefig(figToPath, dpi=900)

    # --- FIGURE 2: TAC vs Recovery (6 subplots) ---
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.8),sharey=True)
    axes = axes.flatten()
    for ax_idx, vg in enumerate(veg_groups):
        ax = axes[ax_idx]
        if vg == 'All':
            subset = df_res.copy()
            title = 'All VegTypes'
        else:
            subset = df_res[df_res['VegGroup'] == vg]
            title = vg

        if subset.empty:
            ax.set_title(f"{title} (no data)")
            continue

        # Density background + scatter
        ax.scatter(subset['TAC'], subset['Recovery'], s=6, color='C0', alpha=0.3)
        sns.kdeplot(x=subset['TAC'], y=subset['Recovery'], fill=True, cmap="RdBu_r", thresh=0.05, ax=ax,
                        alpha=0.4)

        # Linear fit (only use finite points)
        mask = np.isfinite(subset['TAC']) & np.isfinite(subset['Recovery'])
        if mask.sum() > 1:  # 线性拟合至少需要2个点
            xdata = subset.loc[mask, 'TAC'].values
            ydata = subset.loc[mask, 'Recovery'].values
            print(xdata.shape[0] * 9, stats.linregress(xdata, ydata).rvalue, stats.linregress(xdata, ydata).pvalue)
            try:
                slope, intercept = np.polyfit(xdata, ydata, 1)
                xs = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 200)
                ys = (slope-1) * xs + intercept
                ax.plot(xs, ys, color='red', lw=2, label='Linear fit')
            except Exception as e:
                print(f"Linear fit failed: {e}")

        xlabel = 'TAC$_{ED}$ of ' + vg
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Recovery rate')
        ax.set_ylim([0,20])

    plt.tight_layout()
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    figToPath = current_dir + '/4_Figures/FigR01_recovery_TAC'
    plt.savefig(figToPath, dpi=900)