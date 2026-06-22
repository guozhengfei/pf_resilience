import os
import warnings
import netCDF4 as nc
import numpy as np
import pwlf
from scipy.linalg import LinAlgWarning
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,
})

YEAR_SPLIT = 2008


def add_piecewise_fit(ax, years, values, color):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(years) & np.isfinite(values)
    if valid.sum() < 4:
        return

    year_valid = years[valid]
    x = year_valid - YEAR_SPLIT
    y = values[valid]
    anchor = float(np.interp(YEAR_SPLIT, year_valid, y))-0.8

    model = pwlf.PiecewiseLinFit(x, y, degree=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        model.fit(2, [0], [anchor])
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = model.predict(x_hat)
    ax.plot(x_hat + YEAR_SPLIT, y_hat, "--", color=color, lw=1.5)


def read_nc_files(root_folder):
    nc_files = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".nc") and not file.startswith("."):
                nc_files.append(os.path.join(root, file))
    return nc_files

# Specify the root folder where you want to start searching for *.nc files
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

root_folder = current_dir+"/1_Input/OceanE/"
filenames = read_nc_files(root_folder)
labels = [
    "Latitude: 30-90°N",
    "Latitude: 35-90°N",
    "Latitude: 40-90°N"
]
fig, ax = plt.subplots(figsize=(5.*0.8, 3.2*0.85))
for i, label in zip([60,55,50], labels):
    oceanE = []
    for name in filenames:

        dataset = nc.Dataset(name)
        # Read data from a specific variable
        variable_name = 'evapr'
        data = dataset.variables[variable_name][:]
        data = np.ma.getdata(data)
        dataset.close()
        data[data > 10**3] = np.nan
        data = data[:, ::-1, :]
        evap = np.zeros_like(data)
        evap[:,:,:180]=data[:,:,180:]
        evap[:, :,180:] = data[:, :, :180]

        evap_pf = evap[:, :i, :]
        # evap_pf[evap_pf<=0]=np.nan
        oceanE.append(evap_pf)
        # plt.figure();plt.imshow(evap_pf[1,:,:])

    oceanE_arr = np.concatenate(oceanE,axis=0).T
    oceanE_arr_month = np.nanmean(np.nanmean(oceanE_arr,axis=0),axis=0)
    oceanE_arr_yr = oceanE_arr_month.reshape(33,12)
    yearly_se = np.std(oceanE_arr_yr,axis=1)/50
    yearly_mean = np.nanmean(oceanE_arr_yr,axis=1)
    yearly_anom = yearly_mean - np.nanmean(yearly_mean)

    years = np.arange(1990, 2023)
    line, = ax.plot(
        years,
        yearly_anom,
        lw=2.0,
        label=label
    )

    ax.fill_between(
        years,
        yearly_anom - yearly_se,
        yearly_anom + yearly_se,
        color=line.get_color(),
        alpha=0.18,
        linewidth=0
    )
    add_piecewise_fit(ax, years, yearly_anom, line.get_color())

ax.axhline(0, color="0.45", lw=1.0, ls="--")
ax.axvline(2008, color="0.55", lw=1.0, ls=":")

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Evaporation anomaly (mm)", fontsize=11)
ax.set_title("Ocean evaporation sensitivity analysis", fontsize=11)

ax.legend(
    frameon=True,
    facecolor="white",
    framealpha=0.6,
    edgecolor="none",
    fontsize=10,
    loc="best"
)

ax.grid(alpha=0.25, linewidth=0.6)
ax.tick_params(labelsize=10)
ax.set_ylim([-6,6])
fig.tight_layout()
plt.show()
fig.savefig('/Volumes/Zhengfei_01/Project 2 pf resilience/4_Figures/R313_OceanE_sensitivity_01_anomaly_se.png', dpi=600, bbox_inches="tight")

