# Permafrost Resilience Analysis

Python and R scripts for processing permafrost-domain vegetation, climate, fire, flux-site, and moisture-pathway data; calculating temporal-autocorrelation resilience metrics; and generating manuscript figures.

This is a script-based analysis repository, not an installable package. Most scripts expect large data folders beside this code folder.

## System Requirements

Tested on:

- macOS 26.1, arm64
- Python 3.13.2
- R 4.5.2

Expected to work on recent Linux or macOS systems with equivalent geospatial libraries. Windows may require path edits in scripts with hard-coded drive paths.

No non-standard hardware is required. A normal desktop is enough for downstream figure scripts. Full raw-data processing is memory and CPU intensive; several scripts use `mp.Pool(120)` or `mp.Pool(240)`, so reduce these values on smaller machines.

### Dependencies

Tested Python packages:

```text
numpy 2.2.5, pandas 2.2.3, scipy 1.15.3, matplotlib 3.10.3,
Pillow 11.2.1, rasterio 1.4.3, tifffile 2025.5.24,
opencv-python 4.11.0, xarray 2025.4.0, netCDF4 1.7.2,
geopandas 1.1.1, shapely 2.1.1, cartopy 0.25.0,
scikit-learn 1.7.0, statsmodels 0.14.4, pymannkendall 1.4.3,
pwlf 2.5.2, multiprocess 0.70.18, numba 0.61.2,
pyarrow 21.0.0, xgboost 3.0.2, shap 0.48.0,
earthengine-api 1.7.20, tqdm 4.67.1, psutil 7.1.3, pyhdf
```

Optional/script-specific Python packages:

```text
causal_ccm, icoscp_core, pyqt
```

R packages used by the fire workflow:

```text
pacman, L3bin, sf, units, dplyr, lubridate, ggplot2, cowplot,
rnaturalearth, tidyverse, mgcv, ggh4x, ggsci, hashr, terra, viridis
```

## Installation

Use this directory layout:

```text
Project 2 pf resilience/
├── 1_Input/
├── 2_Output/
├── 3_Code_new/     # this repository
└── 4_Figures/
```

Clone:

```bash
mkdir -p "Project 2 pf resilience"/{1_Input,2_Output,4_Figures}
git clone https://github.com/guozhengfei/pf_resilience.git "Project 2 pf resilience/3_Code_new"
cd "Project 2 pf resilience/3_Code_new"
```

Create the Python environment:

```bash
conda create -n pf-resilience -c conda-forge python=3.13 \
  numpy pandas scipy matplotlib pillow rasterio tifffile opencv xarray netcdf4 \
  geopandas shapely cartopy scikit-learn statsmodels pymannkendall pwlf \
  multiprocess numba pyarrow xgboost shap earthengine-api tqdm psutil pyhdf pyqt

conda activate pf-resilience
python -m pip install causal-ccm icoscp_core
```

If using Google Earth Engine scripts:

```bash
earthengine authenticate
```

If using the R fire workflow:

```r
install.packages("pacman")
pacman::p_load(L3bin, sf, units, dplyr, lubridate, ggplot2, cowplot,
               rnaturalearth, tidyverse, mgcv, ggh4x, ggsci, hashr,
               terra, viridis)
```

Typical install time on a normal desktop: 20-45 minutes for Python, plus 10-30 minutes for R packages if needed.

## Demo

This demo runs one downstream figure-data script using processed resilience arrays.

Required files:

```text
../1_Input/landcover_export_2010_5km.tif
../1_Input/landcover_export_2020_5km.tif
../1_Input/permaice_CDSI_dissolved/permaice_CDSI.tif
../2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy
../2_Output/spatial_resilience/resilience_trend_modis.npy
../4_Figures/Fig.1/
```

Run from `3_Code_new/`:

```bash
conda activate pf-resilience
python Fig.01_resilience_patterns_v3.py
```

Expected output:

```text
../4_Figures/Fig.1/Fig01_temporal_patterns_2026.csv
```

Expected demo runtime on a normal desktop: 1-5 minutes, assuming processed `.npy` files are already present.

## Instructions for Use

Run scripts from `3_Code_new/`. Most scripts define the project root as the parent directory:

```python
current_dir = os.path.dirname(os.getcwd()).replace("\\", "/")
```

So inputs are read from `../1_Input`, intermediate files from `../2_Output`, and figures from `../4_Figures`.

Main MODIS kNDVI resilience workflow:

```bash
python step04_v4_processing_pf_kNDVI_16d_modis_final.py
python step05_resilience_trend_modis.py
```

Main outputs:

```text
../2_Output/spatial_resilience/ar1_5yr_kndvi_modis_sg_rolling.npy
../2_Output/spatial_resilience/resilience_trend_modis.npy
```

Selected figure scripts:

```bash
python Fig.01_resilience_patterns_v3.py
python Fig.01b_resilience_patterns_spatial_v2.py
```

To run on your own data:

1. Put your rasters and arrays in the paths expected by the script, or edit the path variables.
2. Match the expected grid, orientation, and temporal dimensions where possible.
3. Reduce multiprocessing pool sizes to fit your hardware.
4. Run preprocessing scripts first, then resilience scripts, then figure scripts.

Full raw-data runs can take several hours to multiple days. Downstream figure scripts usually finish in minutes after intermediate files are available.

## Notes

- On a headless server, replace Matplotlib `Qt5Agg`/`qtAgg` backends with `Agg`.
- If Matplotlib cache warnings appear, run:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```
