import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from cycler import cycler
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import linregress
from scipy.interpolate import griddata
from matplotlib.markers import MarkerStyle
import json
from pprint import pprint
import matplotlib.gridspec as gridspec
from io import StringIO
import os
import matplotlib.ticker as mticker
import geopandas as gpd
import glob
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
#import psyplot.project as psy


### FUNCTIONS (initially taken from Justine Charrel on SpiritX, 28/07/2026) ###

def load_dataset_by_prefix(path, prefix):
    pattern = os.path.join(path, f"{prefix}_*.nc")
    files = sorted(glob.glob(pattern))
    print(f"[INFO] {len(files)} {prefix} files found in {path}.")

    if not files:
        return None

    valid_files = []
    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                ds.load()  # force lecture
                if "time" in ds.coords or "time" in ds.dims:
                    valid_files.append(f)
                else:
                    print(f"[WARNING] Skipping {f}: no 'time' coordinate.")
        except Exception as e:
            print(f"[ERROR] Skipping unreadable file {f}: {e}")

    print(f"[INFO] {len(valid_files)} valid {prefix} files remaining after check.")

    if not valid_files:
        return None
    elif len(valid_files) == 1:
        ds = xr.open_dataset(valid_files[0])
    else:
        ds = xr.open_mfdataset(valid_files, combine='nested', concat_dim='time', parallel=False)

    ds = ds.compute()
    
    # Manage duplicates in the time coordinate
    if "time" in ds.coords:
        original_len = ds.sizes['time']
        var = 'RECORD'
        if prefix == 'SPC' :
            var = 'temperature'

        # Masks null values if the variable exists
        if var in ds:
            ds = ds.where(ds[var].notnull(), drop=True)
        # Remove duplicates in timestamps
        _, index = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=sorted(index))
        ds = ds.sortby('time')

        n_dup = original_len - ds.sizes['time']
        if n_dup > 0:
            print(f"[INFO] Removed {n_dup} duplicated timestamps from {prefix} dataset.")

    return ds

def reindex_to_regular_time_auto(obj, freq, start, end):
    if 'time' not in obj.dims:
        raise ValueError("L'objet n'a pas de dimension 'time'.")
    times = pd.to_datetime(obj.time.values)
    offset_seconds = pd.Series(times.second).mode().iloc[0]
    offset = pd.Timedelta(seconds=offset_seconds)
    full_time = pd.date_range(start + offset, end, freq=freq)
    return obj.reindex(time=full_time)
    
def resolve_column(dataset, candidates):
    """Return first corresponding variable present in xarray.Dataset"""
    for name in candidates:
        if name in dataset.data_vars:
            return dataset[name]
    return None

def plot_outliers(ax, time_series, mask, marker='+', color='red', y_val=None):
    if mask.any():
        masked_times = time_series[mask]

        if y_val is None:
            y_min, y_max = ax.get_ylim()
            y_val = y_min

        ax.scatter(masked_times, np.full(len(masked_times), y_val),
                   color=color, s=20, marker=marker, label="Outliers")
        ax.legend(loc='upper right')


## MY FUNCTIONS ##

def open_single_dataset(
    prefix: str,
    data_folder_path: str,
    site: str,
    year: str,
    month: str,
    day: str,
    **xr_open_kwargs,
    ) -> Optional[xr.Dataset]:
    """
    Open a single NetCDF file for a specific site and day, with optional checks.

    Args:
        data_folder_path: Base path to the data folder (e.g., "/path/to/data").
        site: Site name (e.g., "d17").
        year: Year as a string (e.g., "2025").
        month: Month as a string (e.g., "01").
        day: Day as a string (e.g., "01").
        **xr_open_kwargs: Additional keyword arguments to pass to `xr.open_dataset`.
    """
    # Construct the filename
    path = Path(data_folder_path) / site / "L0" / year / month / day

    ds = load_dataset_by_prefix(path, prefix)

    return ds

def open_and_concatenate_datasets(
    prefix: str,
    data_folder_path: str,
    site_list: List[str],
    begin_year: str,
    end_year: str,
    begin_month: str,
    end_month: str,
    begin_day: str,
    end_day: str,
    prefix_to_freq: Dict[str, str],
    **xr_open_kwargs,
    ) -> Dict[str, Optional[xr.Dataset]]:
    """
    Open and concatenate multiple NetCDF files for each site in `site_list`.
    Returns a dictionary where keys are site names and values are concatenated datasets.

    Args:
        prefix: Prefix for the filenames (e.g., "data_").
        data_folder_path: Base path to the data folder (e.g., "/path/to/data").
        site_list: List of site names (e.g., ["d17", "d18"]).
        begin_year: Start year (e.g., "2025").
        end_year: End year (e.g., "2025").
        begin_month: Start month (e.g., "01").
        end_month: End month (e.g., "12").
        begin_day: Start day (e.g., "01").
        end_day: End day (e.g., "31").
        prefix_to_freq: Dictionary mapping prefixes to their corresponding frequencies (e.g., {'prefix1': '10min', 'prefix2': '1H'}).
        **xr_open_kwargs: Additional keyword arguments for `open_single_dataset`.

    Returns:
        Dictionary mapping site names to concatenated xarray.Dataset objects.
        Sites with no valid files will map to None.
    """
    datasets = {}

    # Get the frequency for the current prefix
    freq = prefix_to_freq.get(prefix, '10min')  # Default to '10min' if prefix not found

    for site in site_list:
        site_datasets = []

        # Generate all dates in the range
        start_date = datetime(int(begin_year), int(begin_month), int(begin_day))
        end_date = datetime(int(end_year), int(end_month), int(end_day))

        current_date = start_date
        while current_date <= end_date:
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            day = current_date.strftime("%d")

            try:
                ds = open_single_dataset(
                    prefix=prefix,
                    data_folder_path=data_folder_path,
                    site=site,
                    year=year,
                    month=month,
                    day=day,
                    **xr_open_kwargs,
                )
                if ds is not None:
                    site_datasets.append(ds)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {site}/{year}/{month}/{day}: {e}")
            except Exception as e:
                print(f"Unexpected error for {site}/{year}/{month}/{day}: {e}")

            # Move to the next day
            current_date += timedelta(days=1)

        # Concatenate datasets for the site
        if site_datasets:
            try:
                concatenated_ds = xr.concat(site_datasets, dim="time")
                # Reindex to regular time using the frequency from prefix_to_freq
                datasets[site] = reindex_to_regular_time_auto(
                    concatenated_ds,
                    freq=freq,
                    start=start_date,
                    end=end_date
                )
                print(f"Successfully concatenated {len(site_datasets)} files for site {site}.")
            except Exception as e:
                print(f"Failed to concatenate datasets for {site}: {e}")
                datasets[site] = None
        else:
            print(f"No valid datasets found for site {site}.")
            datasets[site] = None

    return datasets



