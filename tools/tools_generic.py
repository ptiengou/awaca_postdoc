from info import *
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


### FUNCTIONS initially taken from Justine Charrel (on SpiritX, 28/07/2026) ###
    

def reindex_to_regular_time_auto(obj, freq, start, end):
    print(f'Reindexing start : {start}')
    print(f'Reindexing end : {end}')
    if 'time' not in obj.dims:
        raise ValueError("L'objet n'a pas de dimension 'time'.")
    times = pd.to_datetime(obj.time.values)
    offset_seconds = pd.Series(times.second).mode().iloc[0]
    offset = pd.Timedelta(seconds=offset_seconds)
    full_time = pd.date_range(start + offset, end, freq=freq)
    print('REINDEXED TIME')
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


## DATASET MANAGEMENT ##

def open_single_dataset(
    prefix: str,
    data_folder_path: str,
    site: str,
    year: str,
    month: str,
    day: str,
    drop_spc_bins=True,
    resample_mean=False,
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

    
    if ds is not None:
        # remove bins for SPC
        if prefix == 'SPC' and drop_spc_bins:
            ds = ds.drop('particle_count')
            ds = ds.drop('bins')

        # resample
        resample_freq = prefix_to_resample_freq.get(prefix, None)
        if resample_mean :
            ds = ds.resample(time=resample_freq).mean()
        


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
    resample_mean=False,
    reindex=False,
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

    for site in site_list:
        site_datasets = []

        # Generate all dates in the range
        start_date = datetime(int(begin_year), int(begin_month), int(begin_day))
        end_date = datetime(int(end_year), int(end_month), int(end_day)) + timedelta(days=1)

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
                    resample_mean =resample_mean,
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
        
        if site_datasets:
            # Concatenate datasets for the site
            try:
                concatenated_ds = xr.concat(site_datasets, dim="time")
                datasets[site] = concatenated_ds
                print(f"Successfully concatenated {len(site_datasets)} files for site {site}.")
            except Exception as e:
                print(f"Failed to concatenate datasets for {site}: {e}")
                datasets[site] = None

            # Reindex datasets for the site
            if reindex:
                try:
                    resample_freq = prefix_to_resample_freq.get(prefix, None)
                    acq_freq = prefix_to_acquisition_freq.get(prefix, None)
                    if resample_mean : 
                        index_freq = resample_freq
                    else:
                        index_freq = acq_freq
                    if index_freq is not None:
                        datasets[site] = clear_duplicates_reindex(
                            datasets[site],
                            prefix=prefix,
                            begin_year=begin_year,
                            begin_month=begin_month,
                            begin_day=begin_day,
                            end_year=end_year,
                            end_month=end_month,
                            end_day=end_day
                        )
                except Exception as e:
                    print(f"Failed reindexation for {site}: {e}")
                    # datasets[site] = None

        else:
            print(f"No valid datasets found for site {site}.")
            datasets[site] = None

    return datasets

def clear_duplicates_reindex( 
    ds: xr.Dataset,
    prefix: str,
    begin_year: str,
    begin_month: str,
    begin_day: str,
    end_year: str,
    end_month: str,
    end_day: str,
    keep_first: bool = True,
 ) -> xr.Dataset:
    """
    Remove duplicate timestamps, reindex to a regular time grid, and return the cleaned dataset.
    The frequency for reindexing is retrieved from the `prefix_to_resample_freq` dictionary.

    Args:
        ds: Input xarray Dataset with a 'time' dimension.
        prefix: Prefix to look up the resampling frequency in `prefix_to_resample_freq`.
        begin_year: Start year (e.g., "2023").
        begin_month: Start month (e.g., "01").
        begin_day: Start day (e.g., "01").
        end_year: End year (e.g., "2023").
        end_month: End month (e.g., "12").
        end_day: End day (e.g., "31").
        keep_first: If True, keep the first occurrence of duplicates. Defaults to True.

    Returns:
        xarray Dataset with duplicates removed and reindexed to the specified time range.

    Raises:
        ValueError: If the `prefix` is not found in `prefix_to_resample_freq`.
    """
    # Retrieve the frequency from the dictionary
    freq = prefix_to_resample_freq.get(prefix)
    if freq is None:
        raise ValueError(f"No frequency found for prefix '{prefix}' in prefix_to_resample_freq.")

    # Step 1: Check for duplicates and print their count and dates
    time_series = ds.time.to_series()
    has_duplicates = time_series.duplicated().any()
    if has_duplicates:
        duplicate_counts = time_series.value_counts()
        duplicates = duplicate_counts[duplicate_counts > 1]
        print(f"Number of duplicate timestamps: {len(duplicates)}")
        print(f"Duplicate dates: {list(duplicates.index.strftime('%Y-%m-%d %H:%M:%S'))}")

    # Step 2: Remove duplicates
    if keep_first:
        df = ds.to_dataframe().reset_index()
        df_unique = df.drop_duplicates(subset=["time"], keep='first')
        ds_unique = df_unique.set_index("time").to_xarray()

    # Step 3: Reindex to the full time range
    start_date = datetime(int(begin_year), int(begin_month), int(begin_day))
    end_date = datetime(int(end_year), int(end_month), int(end_day)) + timedelta(days=1)
    full_time = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Reindex the dataset
    ds_reindexed = ds_unique.reindex(time=full_time)

    return ds_reindexed

## STATS ##
def compute_variable_stats(datasets: dict, variable: str) -> pd.DataFrame:
    stats = []
    for name, ds in datasets.items():
        if variable in ds:
            data = ds[variable].values
            n_total = data.size
            n_nan = np.isnan(data).sum()
            stats.append({
                "Dataset": name,
                "Mean": np.nanmean(data),
                "Min": np.nanmin(data),
                "Max": np.nanmax(data),
                "Std": np.nanstd(data),
                "Data Points": n_total - n_nan,
                "NaN Count": n_nan,
                "NaN Fraction": n_nan / n_total if n_total > 0 else 0.0
            })
    return pd.DataFrame(stats)

## PLOTTING ##
def plot_multiple_datasets_separately(
    datasets: Dict[str, xr.Dataset],
    variables: List[str],
    figsize: tuple = (15, 3),
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    **plot_kwargs
 ) -> None:
    """
    Plot time series for each dataset in a single column, with all specified variables on each plot.
    All plots share the same y-axis scale and the same time range.

    Args:
        datasets: Dictionary of xarray.Dataset objects (keys are dataset names).
        variables: List of variable names to plot for each dataset.
        figsize: Figure size for each subplot (width, height). Default: (15, 3).
        ymin: Manually set the minimum y-axis limit. If None, it is computed from the data.
        ymax: Manually set the maximum y-axis limit. If None, it is computed from the data.
        **plot_kwargs: Additional keyword arguments for xarray's plot() method (e.g., color, linestyle).
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    n_datasets = len(datasets)

    # Create a figure with n_datasets subplots in a single column
    fig, axes = plt.subplots(n_datasets, 1, figsize=(figsize[0], figsize[1] * n_datasets), sharey=True)

    # Handle case where only one dataset is provided (axes is not a list)
    if n_datasets == 1:
        axes = [axes]

    # Find global min/max for y-axis scaling
    global_min, global_max = None, None
    for ds in datasets.values():
        for var in variables:
            if var in ds:
                var_min, var_max = ds[var].min().values, ds[var].max().values
                if global_min is None or var_min < global_min:
                    global_min = var_min
                if global_max is None or var_max > global_max:
                    global_max = var_max

    # Override with user-provided ymin/ymax if specified
    if ymin is not None:
        global_min = ymin
    if ymax is not None:
        global_max = ymax

    # Find global time range across all datasets
    global_time_min = None
    global_time_max = None
    for ds in datasets.values():
        if "time" in ds.coords:
            time_min, time_max = ds.time.min().values, ds.time.max().values
            if global_time_min is None or time_min < global_time_min:
                global_time_min = time_min
            if global_time_max is None or time_max > global_time_max:
                global_time_max = time_max

    # Plot each dataset
    for ax, (name, ds) in zip(axes, datasets.items()):
        for var in variables:
            if var in ds:
                ds[var].plot(ax=ax, label=var, **plot_kwargs)
        ax.set_title(f"Dataset: {name}")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        # Set consistent y-axis limits
        if global_min is not None and global_max is not None:
            ax.set_ylim(global_min, global_max)

        # Set consistent x-axis limits (time range)
        if global_time_min is not None and global_time_max is not None:
            ax.set_xlim(global_time_min, global_time_max)

    plt.tight_layout()