#!/usr/bin/env python3
"""
Download ERA5 monthly vertically integrated moisture divergence (VIMD).

Default output:
  1_Input/ERA5_VIMD/monthly/era5_vimd_monthly_global_YYYY.nc

Optional yearly output:
  1_Input/ERA5_VIMD/yearly/era5_vimd_yearly_global_1990_2024.nc

Requirements:
  pip install cdsapi xarray netCDF4

CDS credentials:
  Create ~/.cdsapirc following the CDS API instructions before running.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path


DATASET = "reanalysis-era5-single-levels-monthly-means"
VARIABLE = "vertical_integral_of_divergence_of_moisture_flux"
MONTHS = [f"{month:02d}" for month in range(1, 13)]
TIMES = ["00:00"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download monthly ERA5 VIMD and optionally aggregate to yearly means."
    )
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--mode",
        choices=("monthly", "yearly", "both"),
        default="monthly",
        help="monthly downloads only; yearly aggregates existing/downloaded monthly files; both does both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("1_Input") / "ERA5_VIMD",
        help="Base output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download or overwrite existing output files.",
    )
    return parser.parse_args()


def monthly_request(year: int) -> dict:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [VARIABLE],
        "year": [str(year)],
        "month": MONTHS,
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_monthly(start_year: int, end_year: int, monthly_dir: Path, overwrite: bool) -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: cdsapi. Install it with `python3 -m pip install cdsapi`."
        ) from exc

    monthly_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    for year in range(start_year, end_year + 1):
        target = monthly_dir / f"era5_vimd_monthly_global_{year}.nc"
        if target.exists() and target.stat().st_size > 0 and not overwrite:
            print(f"Skip existing: {target}")
            continue

        print(f"Downloading ERA5 VIMD monthly means for {year} -> {target}")
        client.retrieve(DATASET, monthly_request(year), str(target))


def aggregate_yearly(start_year: int, end_year: int, monthly_dir: Path, yearly_dir: Path, overwrite: bool) -> None:
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: xarray. Install it with `python3 -m pip install xarray netCDF4`."
        ) from exc

    yearly_dir.mkdir(parents=True, exist_ok=True)
    out_file = yearly_dir / f"era5_vimd_yearly_global_{start_year}_{end_year}.nc"
    if out_file.exists() and out_file.stat().st_size > 0 and not overwrite:
        print(f"Skip existing yearly file: {out_file}")
        return

    yearly_datasets = []
    for year in range(start_year, end_year + 1):
        monthly_file = monthly_dir / f"era5_vimd_monthly_global_{year}.nc"
        if not monthly_file.exists():
            raise FileNotFoundError(f"Missing monthly file for yearly aggregation: {monthly_file}")

        print(f"Aggregating annual mean for {year}")
        ds = xr.open_dataset(monthly_file)
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        weights = xr.DataArray(
            [calendar.monthrange(year, month)[1] for month in range(1, 13)],
            coords={time_name: ds[time_name]},
            dims=(time_name,),
        )
        annual = ds.weighted(weights).mean(dim=time_name, keep_attrs=True)
        annual = annual.expand_dims(year=[year])
        yearly_datasets.append(annual)

    combined = xr.concat(yearly_datasets, dim="year")
    combined.attrs.update(
        {
            "source_dataset": DATASET,
            "source_variable": VARIABLE,
            "aggregation": "Annual mean from monthly means weighted by days in month",
        }
    )
    combined.to_netcdf(out_file)
    print(f"Wrote yearly file: {out_file}")


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    monthly_dir = args.output_dir / "monthly"
    yearly_dir = args.output_dir / "yearly"

    if args.mode in ("monthly", "both"):
        download_monthly(args.start_year, args.end_year, monthly_dir, args.overwrite)

    if args.mode in ("yearly", "both"):
        aggregate_yearly(args.start_year, args.end_year, monthly_dir, yearly_dir, args.overwrite)


if __name__ == "__main__":
    main()
