# Permafrost Resilience Analysis

This repository contains Python and R scripts used to process permafrost-domain vegetation, climate, fire, flux-site, and moisture-pathway data, calculate temporal autocorrelation based resilience metrics, and generate manuscript figures.

The code is organized as analysis scripts rather than an installable Python package. Most scripts expect the repository to sit beside large input and output data folders.

## System Requirements

### Operating system

Tested on:

- macOS 26.1, arm64
- Python 3.13.2
- R 4.5.2

The Python scripts should also run on recent Linux and macOS systems with equivalent geospatial libraries. Windows may require path edits, especially for scripts that contain hard-coded Windows drive paths in the fire-analysis R workflow.

### Hardware

No non-standard hardware is required.

For a small downstream demo, a normal desktop or laptop with 16 GB RAM is sufficient. For the full raw-data processing workflow, a high-memory workstation is recommended because several scripts read hundreds of GB of raster inputs and use large multiprocessing pools. Some scripts currently request `mp.Pool(120)` or `mp.Pool(240)`; reduce these values to match the number of available CPU cores on your machine.

The local development dataset used with this repository is approximately:

- `1_Input`: 202 GB
- `2_Output`: 13 GB
- `4_Figures`: 549 MB

These data directories are not stored in this Git repository.

### Python dependencies

The following versions were tested on the development workstation:

| Package | Tested version |
| --- | --- |
| numpy | 2.2.5 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| matplotlib | 3.10.3 |
| Pillow | 11.2.1 |
| rasterio | 1.4.3 |
| tifffile | 2025.5.24 |
| opencv-python | 4.11.0 |
| xarray | 2025.4.0 |
| netCDF4 | 1.7.2 |
| geopandas | 1.1.1 |
| shapely | 2.1.1 |
| cartopy | 0.25.0 |
| scikit-learn | 1.7.0 |
| statsmodels | 0.14.4 |
| pymannkendall | 1.4.3 |
| pwlf | 2.5.2 |
| multiprocess | 0.70.18 |
| numba | 0.61.2 |
| pyarrow | 21.0.0 |
| xgboost | 3.0.2 |
| shap | 0.48.0 |
| earthengine-api | 1.7.20 |
| tqdm | 4.67.1 |
| psutil | 7.1.3 |
| pyhdf | installed |

Optional or script-specific Python packages:

- `causal_ccm`: used by convergent cross-mapping scripts.
- `icoscp_core`: used by ICOS flux-data download scripts.
- `pyqt`: required by scripts that set the Matplotlib backend to `Qt5Agg` or `qtAgg`.

### R dependencies

The R scripts are used mainly for fire radiative power preprocessing and conversion utilities. Tested with R 4.5.2.

R packages used by the fire workflow include:

- `pacman`
- `L3bin`
- `sf`
- `units`
- `dplyr`
- `lubridate`
- `ggplot2`
- `cowplot`
- `rnaturalearth`
- `tidyverse`
- `mgcv`
- `ggh4x`
- `ggsci`
- `hashr`
- `terra`
- `viridis`

## Installation Guide

### 1. Clone the repository

The scripts expect this repository to be the code folder inside a larger project directory:

```text
Project 2 pf resilience/
├── 1_Input/
├── 2_Output/
├── 3_Code_new/        # this repository
└── 4_Figures/
```

Recommended clone layout:

```bash
mkdir -p "Project 2 pf resilience"/{1_Input,2_Output,4_Figures}
git clone https://github.com/guozhengfei/pf_resilience.git "Project 2 pf resilience/3_Code_new"
cd "Project 2 pf resilience/3_Code_new"
```

Place the required input data in `../1_Input`. If running downstream figure scripts only, also place the required processed arrays in `../2_Output`.

### 2. Create a Python environment

Using conda or mamba is recommended because the project depends on geospatial packages such as Rasterio, Cartopy, GeoPandas, NetCDF4, and PyHDF.

```bash
conda create -n pf-resilience -c conda-forge python=3.13 \
  numpy pandas scipy matplotlib pillow rasterio tifffile opencv xarray netcdf4 \
  geopandas shapely cartopy scikit-learn statsmodels pymannkendall pwlf \
  multiprocess numba pyarrow xgboost shap earthengine-api tqdm psutil pyhdf pyqt

conda activate pf-resilience
```

If a package is unavailable from conda on your platform, install it with pip inside the activated environment:

```bash
python -m pip install causal-ccm icoscp_core
```

For Google Earth Engine scripts that import `ee`, authenticate once:

```bash
earthengine authenticate
```

Typical installation time on a normal desktop computer is 20 to 45 minutes, depending mainly on geospatial package solving and download speed.

### 3. Install R packages, if using the R fire workflow

```r
install.packages("pacman")
pacman::p_load(
  L3bin, sf, units, dplyr, lubridate, ggplot2, cowplot, rnaturalearth,
  tidyverse, mgcv, ggh4x, ggsci, hashr, terra, viridis
)
```

Typical R package installation time is 10 to 30 minutes on a normal desktop computer.

## Demo

The demo runs a downstream figure-data script using already processed resilience arrays. It is intended as a smoke test of the environment and directory layout.

### Required demo inputs

Run from `3_Code_new/`. The following files must exist:

```text
../1_Input/landcover_export_2010_5km.tif
../1_Input/landcover_export_2020_5km.tif
../1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif
../2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy
../2_Output/spatial_resilience/resilience_trend_modis.npy
../4_Figures/Fig.1/
```

### Run the demo

```bash
cd "Project 2 pf resilience/3_Code_new"
conda activate pf-resilience
python Fig.01_resilience_patterns_v3.py
```

Some scripts use an interactive Qt Matplotlib backend. On a headless server, change the backend line in the script from `Qt5Agg` or `qtAgg` to `Agg`, or run on a machine with a graphical display.

### Expected output

The script creates or overwrites:

```text
../4_Figures/Fig.1/Fig01_temporal_patterns_2026.csv
```

The CSV contains annual temporal autocorrelation summaries by vegetation class with columns:

```text
Year, Needle_Forest, Mixed_Forest, Savanna, Shrubland, Grassland, All_Mean, Standard_Error
```

Expected demo runtime on a normal desktop computer is approximately 1 to 5 minutes, assuming the processed `.npy` files are already present. Runtime is dominated by loading an approximately 826 MB resilience array.

## Instructions for Use

Always run scripts from the repository directory, `3_Code_new/`, because most scripts define the project root as:

```python
current_dir = os.path.dirname(os.getcwd()).replace("\\", "/")
```

This means inputs are read from `../1_Input`, intermediate outputs from `../2_Output`, and figure outputs from `../4_Figures`.

### Main MODIS kNDVI resilience workflow

1. Prepare input rasters and driver arrays in `../1_Input`, including:
   - `NDVI_pf_16d/`
   - `landcover_export_2010_5km.tif`
   - `landcover_export_2020_5km.tif`
   - `data for drivers/pr_month_anomaly.npy`
   - `data for drivers/srad_month_anomaly.npy`
   - `data for drivers/tmmx_month_anomaly.npy`

2. Compute rolling temporal autocorrelation:

```bash
python step04_v4_processing_pf_kNDVI_16d_modis_final.py
```

Expected output:

```text
../2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy
```

3. Compute resilience trends:

```bash
python step05_resilience_trend_modis.py
```

Expected output:

```text
../2_Output/spatial_resilience/resilience_trend_modis.npy
```

4. Estimate breakpoint timing:

```bash
python step06_tac_breaking_point_detection.py
```

Expected output:

```text
../2_Output/break_points_modis.npy
```

5. Generate selected figure data and figures:

```bash
python Fig.01_resilience_patterns_v3.py
python Fig.01b_resilience_patterns_spatial_v2.py
python Fig.02a_driver_relative_importance.py
python Fig.02b_variable_sensitivity.py
python Fig.03_fire_frequency_plot.py
python Fig.04ab_gpp_resilience_relationship_stability.py
python Fig.05a_trendy_resilience_v3.py
```

Each figure script reads a different subset of `../1_Input` and `../2_Output` and writes figures or CSVs under `../4_Figures` or `../2_Output`.

### Other workflows

- `S0.*` scripts preprocess raw input products such as climate drivers, GPP, VOD, BRDF, GIMMS, and flux-site data.
- `R1.*`, `R2.*`, and `R3.*` scripts contain revision, sensitivity, and mechanism analyses.
- `Fig.*` scripts generate manuscript and supplementary figures.
- `download_era5_vimd_1990_2024.py` and `download_xbase_from_icos.py` download external data and may require credentials or service-specific authentication.
- `Summarising global FRP v3.R` processes global fire radiative power data. Update its `setwd()` and `source()` paths before use on a new machine.

### Running on your own data

To run the software on new data:

1. Reproduce the expected project layout with `1_Input`, `2_Output`, `3_Code_new`, and `4_Figures` as sibling directories.
2. Put your input rasters and arrays into the same relative paths and file names used by the target script, or edit the path variables near the top of that script.
3. Match spatial grids, orientation, and resolution to the existing workflow. Many scripts assume the permafrost mask grid, reversed north-south orientation, and fixed temporal lengths such as 24 years and 23 observations per year.
4. Adjust multiprocessing pool sizes to your hardware before running full processing scripts.
5. Run preprocessing scripts first, then resilience calculation scripts, then figure scripts.

For large raw-data runs, expected runtime ranges from several hours to multiple days depending on storage speed, memory, CPU count, and which products are being processed. Downstream figure scripts usually run in minutes once the intermediate `.npy`, `.npz`, `.csv`, and `.tif` products are available.

## Notes

- The repository currently does not include a formal test suite.
- Many scripts are manuscript-analysis scripts with fixed file names and assumptions. When adapting the workflow, inspect the target script's input paths before running.
- If Matplotlib reports that its default cache directory is not writable, set `MPLCONFIGDIR` to a writable temporary directory before running:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```
