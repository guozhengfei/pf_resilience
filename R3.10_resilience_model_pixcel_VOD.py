import os

# Set before importing numpy/xgboost to avoid CPU oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import multiprocessing as mp

import numpy as np
import xgboost as xgb
from tqdm import tqdm


# ============================================================
# Settings
# ============================================================
N_WORKERS = min(os.cpu_count() or 1, 8)   # Mac Studio: try 4, 6, 8, or 10
CHUNK_PIXELS = 256                       # try 128/256/512
N_ESTIMATORS = 100

DRIVER_VARS = [
    "vpd", "srad", "pr", "tmean", "soilT",
    "sm", "Alt", "ndvi", "gsl", "lai"
]

RESPONSE_VAR = "tac"
VARIABLES = DRIVER_VARS + [RESPONSE_VAR]
N_FEATURES = len(DRIVER_VARS)

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.3,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "nthread": 1,
    "verbosity": 0,
    "seed": 0,
}


# ============================================================
# Helper functions
# ============================================================
def smooth_2d_array_fast(array, window_size=3):
    """
    Fast row-wise moving average with edge padding.
    Equivalent to your original smooth_2d_array() for window_size=3.
    """
    pad_width = window_size // 2
    padded = np.pad(array, ((0, 0), (pad_width, pad_width)), mode="edge")

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=window_size,
        axis=1
    )

    return windows.mean(axis=-1).astype(np.float32, copy=False)


def fast_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) < 2:
        return np.nan

    x = x[mask]
    y = y[mask]

    dx = x - x.mean()
    dy = y - y.mean()

    denom = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))

    if denom == 0:
        return np.nan

    return np.sum(dx * dy) / denom


def fast_slope(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if np.sum(mask) < 2:
        return np.nan

    x = x[mask]
    y = y[mask]

    dx = x - x.mean()
    dy = y - y.mean()

    denom = np.sum(dx * dx)

    if denom == 0:
        return np.nan

    return np.sum(dx * dy) / denom


def model_validate_pixel(pixel_array):
    """
    Fast version:
    - train once using all years
    - predict fitted values on the same years
    - calculate fitted r
    - calculate native XGBoost SHAP values
    """

    X = pixel_array[:, :-1]
    y = pixel_array[:, -1]

    dtrain = xgb.DMatrix(X, label=y, missing=np.nan)

    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        verbose_eval=False
    )

    # Fitted prediction, not cross-validated prediction
    pred = model.predict(dtrain).astype(np.float32)

    # This r is optimistic because prediction is in-sample
    r = fast_corr(pred, y)

    # Native TreeSHAP from XGBoost
    # Last column is bias term, so remove it
    shap_values = model.predict(dtrain, pred_contribs=True)[:, :-1]

    shap_abs = np.mean(np.abs(shap_values), axis=0).astype(np.float32)

    sen = np.empty(N_FEATURES, dtype=np.float32)

    shap_mean = np.nanmean(shap_values, axis=0)
    shap_std = np.nanstd(shap_values, axis=0)

    for m in range(N_FEATURES):
        x2 = X[:, m]
        y2 = shap_values[:, m].copy()

        # Same 3-sigma filter logic as your std_filter()
        bad = (y2 > shap_mean[m] + 3 * shap_std[m]) | (
            y2 < shap_mean[m] - 3 * shap_std[m]
        )
        y2[bad] = np.nan

        sen[m] = fast_slope(x2, y2)

    return np.concatenate(
        [
            np.array([r], dtype=np.float32),
            shap_abs,
            sen,
            y.astype(np.float32),
            pred.astype(np.float32),
        ]
    )


def process_chunk(chunk):
    n_pixels = chunk.shape[0]
    n_years = chunk.shape[1]

    n_outputs = 1 + N_FEATURES + N_FEATURES + n_years + n_years

    out = np.empty((n_pixels, n_outputs), dtype=np.float32)

    for i in range(n_pixels):
        out[i] = model_validate_pixel(chunk[i])

    return out


# ============================================================
# Main script
# ============================================================
if __name__ == "__main__":

    current_dir = Path(os.path.dirname(os.getcwd()))

    data_files = {
        "ar1": "2_Output/spatial_resilience/ar1_5yr_vod_sg_rolling.npy",
        "ndvi": "1_Input/data for drivers/ndvi_yearly.npy",
        "sos": "1_Input/data for drivers/sos.npy",
        "eos": "1_Input/data for drivers/eos.npy",
        "vpd": "1_Input/data for drivers/vpd_yearly.npy",
        "tmmx": "1_Input/data for drivers/tmmx_yearly.npy",
        "tmmn": "1_Input/data for drivers/tmmn_yearly.npy",
        "srad": "1_Input/data for drivers/srad_yearly.npy",
        "pr": "1_Input/data for drivers/pr_yearly.npy",
        "soilT": "1_Input/data for drivers/sT_yearly.npy",
        "Alt": "1_Input/data for drivers/Alt.npy",
        "sm": "1_Input/data for drivers/sm_yearly.npy",
        "lai": "1_Input/data for drivers/LAI_yearly.npy",
    }

    print("Loading data...")

    data = {
        k: np.load(current_dir / v).astype(np.float32, copy=False)
        for k, v in data_files.items()
    }

    print("Preparing variables...")

    data["ar1"][data["ar1"] == 0] = np.nan

    ar1_rsp = data["ar1"].reshape(-1, 22, 23)
    ar1_yearly = np.nanmean(ar1_rsp, axis=2)[:, 2:-2]
    data["tac"] = ar1_yearly.astype(np.float32, copy=False)

    data["ndvi"] = data["ndvi"][:, 2:-4]

    data["sos"] = data["sos"][:, 20:-3]
    data["eos"] = data["eos"][:, 20:-3]

    data["gsl"] = data["eos"] - data["sos"]

    row_means = np.nanmean(data["gsl"], axis=1)
    row_idx, col_idx = np.where(np.isnan(data["gsl"]))
    data["gsl"][row_idx, col_idx] = row_means[row_idx]

    data["tmean"] = (data["tmmx"][:, 20:-3] + data["tmmn"][:, 20:-3]) * 0.5

    for k in ["vpd", "srad", "pr", "soilT", "Alt", "sm", "lai"]:
        data[k] = data[k][:, 20:-3]

    mask_vars = [
        "sm", "vpd", "srad", "pr", "tmean",
        "ndvi", "soilT", "Alt", "gsl", "tac"
    ]

    mask = np.any(
        np.vstack([np.isnan(data[v]).any(axis=1) for v in mask_vars]),
        axis=0
    )

    print(f"Valid pixels: {np.sum(~mask):,} / {len(mask):,}")

    print("Smoothing variables...")

    smoothed_data = {
        v: smooth_2d_array_fast(data[v][~mask], window_size=3)
        for v in VARIABLES
    }

    array = np.stack([smoothed_data[v] for v in VARIABLES], axis=2)
    array = array.astype(np.float32, copy=False)

    print(f"Input array shape: {array.shape}")

    chunks = [
        array[i:i + CHUNK_PIXELS]
        for i in range(0, array.shape[0], CHUNK_PIXELS)
    ]

    print(f"Running with {N_WORKERS} workers and {len(chunks)} chunks...")

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=N_WORKERS) as pool:
        result_chunks = list(
            tqdm(
                pool.imap(process_chunk, chunks, chunksize=1),
                total=len(chunks),
                desc="Processing"
            )
        )

    results = np.vstack(result_chunks)

    out_file = current_dir / "2_Output/Temporal/Temporal_r_shap_obs_pre_opt_vod_fast_fitted.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_file,
        array1=results,
        array2=mask
    )

    print(f"Saved to: {out_file}")