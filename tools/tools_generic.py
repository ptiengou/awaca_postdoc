from info import *
from imports import *

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
    Largely taken from Justine Charrel's code.

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
                concatenated_ds = xr.concat(site_datasets, dim="time")#, combine_attrs="drop_conflicts")
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
            
            try:
                for var in datasets[site].data_vars:
                    if var in site_datasets[0].data_vars:
                        datasets[site][var].attrs = site_datasets[0][var].attrs.copy()
            except Exception as e:
                    print(f"Lost attributes")

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

def create_dict_data_datadaily(
    sites: list,
    sensors: list,
    start_date: str,
    end_date: str,
) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Load and resample data for all sensors and sites.
    Returns two dictionaries: `data` and `daily_data`.
    """
    data = {}

    for sensor in sensors:
        sensor_datasets = {}
        for site in sites:
            file_path = f'../../data/{sensor}_{site}_{start_date}_{end_date}.netcdf'

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            try:
                ds = xr.open_dataset(file_path, engine='netcdf4')
                sensor_datasets[site] = ds
            except Exception as e:
                print(f"Failed to open {file_path}: {e}")
                continue

        if sensor_datasets:
            data[f"{sensor}"] = sensor_datasets
            daily_datasets = {
                name: ds.resample(time='1D').mean()
                for name, ds in sensor_datasets.items()
            }
            daily_data[f"{sensor}"] = daily_datasets

    return data, daily_data

def create_data(
    sites: list,
    sensors: list,
    start_date: str,
    end_date: str,
) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Load raw data for all sensors and sites.
    Returns a nested dictionary: {sensor: {site: xr.Dataset}}.
    """
    data = {}

    for sensor in sensors:
        sensor_datasets = {}
        for site in sites:
            file_path = f'../../data/{sensor}_{site}_{start_date}_{end_date}.netcdf'

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            try:
                ds = xr.open_dataset(file_path, engine='netcdf4')
                sensor_datasets[site] = ds
            except Exception as e:
                print(f"Failed to open {file_path}: {e}")
                continue

        if sensor_datasets:
            data[sensor] = sensor_datasets

    return data
    
def create_daily_data(
    data: Dict[str, Dict[str, xr.Dataset]]
) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Resample raw data into daily averages.
    Returns a nested dictionary: {sensor: {site: xr.Dataset}}.
    """
    daily_data = {}

    for sensor, sensor_datasets in data.items():
        daily_datasets = {
            site: ds.resample(time='1D').mean()
            for site, ds in sensor_datasets.items()
        }
        daily_data[sensor] = daily_datasets

    return daily_data

def create_resampled_data(
    data: Dict[str, Dict[str, xr.Dataset]],
    sampling: str,
) -> Dict[str, Dict[str, xr.Dataset]]:
    """
    Resample raw data into averages.
    Returns a nested dictionary: {sensor: {site: xr.Dataset}}.
    """
    sampled_data = {}

    for sensor, sensor_datasets in data.items():
        sampled_datasets = {
            site: ds.resample(time=sampling).mean()
            for site, ds in sensor_datasets.items()
        }
        sampled_data[sensor] = sampled_datasets

    return sampled_data

def filter_data_by_max_values(data: dict, variables: list):
    """
    Filter datasets in the data dictionary by replacing values above the max value
    defined in var_to_maxval with NaN for each variable in the list.
    Existing NaN values are preserved.

    Args:
        data (dict): Dictionary of the form {sensor: {site: xarray.Dataset}}.
        variables (list): List of variables to filter.

    Returns:
        dict: Filtered data dictionary with values above maxval replaced by NaN.
    """
    filtered_data = {}

    for sensor, sites in data.items():
        filtered_data[sensor] = {}
        for site, ds in sites.items():
            filtered_ds = ds.copy()
            for variable in variables:
                if variable in var_to_maxval and variable in filtered_ds:
                    max_val = var_to_maxval[variable]
                    # Replace values > max_val with NaN, preserving existing NaNs
                    filtered_ds[variable] = filtered_ds[variable].where(
                        (filtered_ds[variable] <= max_val) | (np.isnan(filtered_ds[variable]))
                    )
            filtered_data[sensor][site] = filtered_ds

    return filtered_data

def filter_datasets_golden(
    ds_dict: dict[str, dict[str, xr.Dataset]], start_date: str, end_date: str
) -> dict[str, dict[str, xr.Dataset]]:
    """Slice a nested dictionary of xarray Datasets {sensor: {site: ds}} between two dates.

    Parameters
    ----------
    ds_dict : dict[str, dict[str, xr.Dataset]]
        Nested dictionary where top-level keys are sensors and inner keys are sites.
    start_date : str
        Start date formatted as 'YYYY-MM-DD' (or ISO string).
    end_date : str
        End date formatted as 'YYYY-MM-DD'.

    Returns
    -------
    dict[str, dict[str, xr.Dataset]]
        Nested dictionary containing the temporally sliced Datasets.
    """
    return {
        sensor: {
            site: ds.sel(
                {
                    ("time" if "time" in ds.coords else "t"): slice(
                        start_date, end_date
                    )
                }
            )
            for site, ds in sites.items()
        }
        for sensor, sites in ds_dict.items()
    }

def extract_variable_series(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    variable: str,
    site: str,
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
) -> Optional[xr.DataArray]:
    """
    Extracts a specific variable DataArray for a given site from the nested
    sensor_datasets structure {sensor: {site: Dataset}}.
    """
    sensor = variable_to_sensor.get(variable)
    if sensor and site in sensor_datasets.get(sensor, {}):
        ds = sensor_datasets[sensor][site]
        if variable in ds:
            return ds[variable]
    return None

def extract_site_dataset(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    site: str,
    trigger_variable: str = "FluxMean2",
    variables: Optional[List[str]] = None,
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
    tolerance: Optional[str] = "10min",  # Max allowable shift when matching timestamps
) -> xr.Dataset:
    """
    Extracts variables for a site while strictly preserving all timesteps of 
    the primary trigger_variable. Secondary variables are aligned to the trigger 
    time axis via nearest-neighbor matching.
    """
    target_vars = variables if variables is not None else list(variable_to_sensor.keys())
    
    # 1. Retrieve the primary trigger DataArray first
    trigger_da = extract_variable_series(
        sensor_datasets=sensor_datasets,
        variable=trigger_variable,
        site=site,
        variable_to_sensor=variable_to_sensor,
    )
    
    if trigger_da is None:
        return xr.Dataset()

    aligned_das = [trigger_da]

    # 2. Reindex secondary variables to match the trigger's exact time coordinate
    for var in target_vars:
        if var == trigger_variable:
            continue
            
        da = extract_variable_series(
            sensor_datasets=sensor_datasets,
            variable=var,
            site=site,
            variable_to_sensor=variable_to_sensor,
        )
        if da is not None:
            # Match nearest timestamps within tolerance without dropping trigger timesteps
            da_reindexed = da.reindex(
                time=trigger_da.time, 
                method="nearest", 
                tolerance=pd.Timedelta(tolerance) if tolerance else None
            )
            aligned_das.append(da_reindexed)

    # Combine into a single Dataset anchored on trigger_da's time dimension
    return xr.merge(aligned_das)

def _format_var_label(var: str, site: str, var_to_units: Optional[Dict[str, str]], var_to_longname: Optional[Dict[str, str]]) -> str:
    """Formats standard variable labels with long names, sites, and units."""
    long_name = (var_to_longname or {}).get(var, var)
    unit = (var_to_units or {}).get(var, "")
    return f"{long_name} ({site})" + (f" [{unit}]" if unit else "")

def resample_dbz(
    obj: xr.Dataset | xr.DataArray,
    var_name: str = "Zea",
    freq: str = "10min",
    min_count: int = 1,
) -> xr.DataArray | xr.Dataset:
    """Resamples a radar reflectivity field in dBZ to a specified time frequency

    using linear reflectivity averaging.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        The input MRR xarray object.
    var_name : str, default 'Zea'
        The target dBZ variable name if `obj` is a Dataset.
    freq : str, default '10min'
        Pandas/xarray time frequency string (e.g., '10min', '15min', '1H').
    min_count : int, default 1
        Minimum number of valid time steps required per bin; returns NaN if
        fewer.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The time-resampled object in dBZ.
    """
    # 1. Extract target DataArray
    if isinstance(obj, xr.Dataset):
        da = obj[var_name]
    elif isinstance(obj, xr.DataArray):
        da = obj
        var_name = da.name or "Zea"
    else:
        raise TypeError("Input must be an xarray DataArray or Dataset.")

    # 2. Linearize reflectivity (dBZ -> mm^6 m^-3)
    da_linear = 10.0 ** (da / 10.0)

    # 3. Resample & average in linear space
    da_lin_resampled = da_linear.resample(time=freq).mean(
        dim="time"#, min_count=min_count
    )

    # 4. Convert back to dBZ safely (ignoring warnings for zeros/NaNs)
    with np.errstate(invalid="ignore", divide="ignore"):
        da_dbz = 10.0 * np.log10(da_lin_resampled.where(da_lin_resampled > 0))

    # Preserve metadata
    da_dbz.name = var_name
    da_dbz.attrs = da.attrs
    da_dbz.attrs["units"] = "dBZ"

    # Return same data structure as input
    if isinstance(obj, xr.Dataset):
        resample_ds = obj.resample(time=freq).first()
        resample_ds[var_name] = da_dbz
        return resample_ds

    return da_dbz

def compute_vertical_over_threshold_fraction(
    ds_or_da: xr.Dataset | xr.DataArray,
    var_name: str = "Zea",
    vertical_dim: str = "range",
    threshold: float = -5.0,
    height_slice: tuple[float, float] | None = None,
) -> xr.DataArray:
    """Calculates the fraction of vertical levels exceeding a given threshold

    for each time step, with an optional vertical height slice.

    Parameters
    ----------
    ds_or_da : xr.Dataset or xr.DataArray
        The input xarray Dataset or DataArray.
    var_name : str, default 'Zea'
        The target variable name if `ds_or_da` is a Dataset.
    vertical_dim : str, default 'range'
        The name of the vertical dimension over which to compute the fraction.
    threshold : float, default -5.0
        The minimum threshold value (e.g. -5 dBZ).
    height_slice : tuple of (float, float), optional
        Optional vertical height bounds as (min_height, max_height), e.g., (300, 1200).
        If provided, the dataset will be sliced along `vertical_dim` before computing.

    Returns
    -------
    xr.DataArray
        A 1D DataArray indexed by time containing the vertical fraction (0.0 to 1.0)
        exceeding the threshold for each profile.
    """
    # 1. Extract DataArray
    if isinstance(ds_or_da, xr.Dataset):
        if var_name not in ds_or_da:
            raise KeyError(
                f"Variable '{var_name}' not found in the provided Dataset."
            )
        da = ds_or_da[var_name]
    elif isinstance(da_or_ds, xr.DataArray):
        da = ds_or_da
        var_name = da.name or var_name
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray.")

    if vertical_dim not in da.dims:
        raise KeyError(
            f"Vertical dimension '{vertical_dim}' not found in DataArray dimensions {da.dims}."
        )

    # 2. Apply optional vertical slicing
    if height_slice is not None:
        if len(height_slice) != 2:
            raise ValueError(
                "`height_slice` must be a tuple of length 2: (min_height, max_height)."
            )
        da = da.sel({vertical_dim: slice(height_slice[0], height_slice[1])})

    # 3. Boolean mask of values exceeding threshold
    over_thresh = da > threshold

    # 4. Calculate fraction of vertical levels exceeding threshold
    fraction = over_thresh.mean(dim=vertical_dim, skipna=True)

    # 5. Set metadata attributes
    slice_str = (
        f" between {height_slice[0]} and {height_slice[1]}"
        if height_slice
        else ""
    )
    fraction.name = f"{var_name}_fraction_above_{threshold}dBZ"
    fraction.attrs["units"] = "1"
    fraction.attrs[
        "description"
    ] = f"Fraction of vertical levels along {vertical_dim}{slice_str} where {var_name} > {threshold}"

    return fraction

## STATS ##
def compute_variable_stats(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    variable: str,
    sites: Optional[List[str]] = sites
) -> pd.DataFrame:
    """
    Compute statistics for a specific variable across all sites and sensors.

    Args:
        sensor_datasets: Nested dictionary of datasets structured as {sensor: {site: dataset}}.
        variable: Variable name to compute statistics for.
        variable_to_sensor: Dictionary mapping variables to their respective sensors.
        sites: Optional list of site names to include. If None, all sites are included.

    Returns:
        A pandas DataFrame with statistics for the variable across all specified sites.
    """
    stats = []

    # Find the sensor for the variable
    sensor = variable_to_sensor.get(variable)
    if sensor is None:
        raise ValueError(f"Variable '{variable}' not found in variable_to_sensor.")

    # Iterate over all sites in the sensor's datasets
    for site, ds in sensor_datasets.get(sensor, {}).items():
        if sites is not None and site not in sites:
            continue  # Skip if site is not in the provided list

        if variable in ds:
            data = ds[variable].values
            n_total = data.size
            n_nan = np.isnan(data).sum()

            stats.append({
                "Site": site,
                "Mean": np.nanmean(data),
                "Min": np.nanmin(data),
                "Max": np.nanmax(data),
                "Std": np.nanstd(data),
                "Data Points": n_total - n_nan,
                "NaN Count": n_nan,
                "NaN Fraction": n_nan / n_total if n_total > 0 else 0.0
            })

    return pd.DataFrame(stats)

def plot_binned_distribution(
    data: dict,
    variable: str,
    variable_to_sensor: dict,
    bin_number: int = 20,
    min_value: float = None,
    max_value: float = None,
 ):
    """
    Plot a binned distribution for each site in the data dictionary.

    Args:
        data (dict): Dictionary of the form {sensor: {site: xarray.Dataset}}.
        variable (str): The variable to plot.
        variable_to_sensor (dict): Dictionary mapping variables to sensors.
        bin_number (int): Number of bins (default: 20).
        min_value (float): Minimum value for bins (default: rounded min of data).
        max_value (float): Maximum value for bins (default: rounded max of data).
    """
    # Determine the sensor for the variable
    sensor = variable_to_sensor.get(variable)
    if sensor is None:
        raise ValueError(f"No sensor found for variable: {variable}")

    # Get the datasets for the sensor
    sensor_data = data.get(sensor)
    if sensor_data is None:
        raise ValueError(f"No data found for sensor: {sensor}")

    # Extract non-NaN values for the variable across all sites
    all_values = []
    for site, ds in sensor_data.items():
        if variable in ds:
            values = ds[variable].values.flatten()
            all_values.extend(values[~np.isnan(values)])

    if not all_values:
        raise ValueError(f"No non-NaN data found for variable {variable} in sensor {sensor}")

    # Calculate min and max if not provided
    if min_value is None:
        min_value = np.floor(np.min(all_values))
    if max_value is None:
        max_value = np.ceil(np.max(all_values))

    # Create bins
    bins = np.linspace(min_value, max_value, bin_number + 1)

    # Create subplots
    num_sites = len(sensor_data)
    fig, axes = plt.subplots(nrows=num_sites, ncols=1, figsize=(5, 3 * num_sites))
    if num_sites == 1:
        axes = [axes]

    for ax, (site, ds) in zip(axes, sensor_data.items()):
        if variable in ds:
            values = ds[variable].values.flatten()
            values = values[~np.isnan(values)]  # Filter out NaN values

            # Plot histogram with normalized frequency (percentage)
            counts, _ = np.histogram(values, bins=bins)
            percentages = (counts / len(values)) * 100
            ax.bar(bins[:-1], percentages, width=np.diff(bins), align='edge', alpha=0.7, label=site)

            ax.set_title(f"Distribution of {variable} for {site}")
            ax.set_xlabel(variable)
            ax.set_ylabel("Frequency (%)")
            ax.set_ylim(0, 100)
            ax.legend()

    plt.tight_layout()
    plt.show()

def plot_binary_availability_for_sites(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    variables: List[str],
    sites: List[str],
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
    var_to_units: Optional[Dict[str, str]] = var_to_units,
    var_to_longname: Optional[Dict[str, str]] = var_to_longname,
    figsize: Tuple[int, int] = (15, 5),
    colors: Optional[List[str]] = None,
    **plot_kwargs
) -> None:
    """
    Plot binary (0/1) time series for each variable, indicating data availability (1) or NaN (0).
    Each subplot corresponds to a variable, and the legend indicates the site.

    Args:
        sensor_datasets: Nested dictionary of datasets structured as {sensor: {site: dataset}}.
        variables: List of variable names to plot.
        sites: List of site names to include in the plot.
        variable_to_sensor: Dictionary mapping variables to their respective sensors.
        var_to_units: Dictionary mapping variables to their unit strings.
        var_to_longname: Dictionary mapping variables to their descriptive long names.
        figsize: Figure size for each subplot (width, height). Default: (15, 5).
        colors: List of colors for each site. If None, uses default colors.
        **plot_kwargs: Additional keyword arguments for xarray's plot() method (e.g., linestyle).
    """
    if not sensor_datasets or not variables or not sites:
        raise ValueError("No datasets, variables, or sites provided.")

    if variable_to_sensor is None:
        raise ValueError("variable_to_sensor dictionary is required.")

    if colors is not None and len(colors) != len(sites):
        raise ValueError("Length of colors must match the number of sites.")

    n_variables = len(variables)
    fig, axes = plt.subplots(n_variables, 1, figsize=(figsize[0], figsize[1] * n_variables), sharex=True)

    if n_variables == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        sensor = variable_to_sensor.get(var)
        if sensor is None:
            print(f"Warning: Variable '{var}' not found in variable_to_sensor. Skipping.")
            continue

        for i, site in enumerate(sites):
            if site in sensor_datasets.get(sensor, {}):
                ds = sensor_datasets[sensor][site]
                if var in ds:
                    # Create binary mask: 1 for non-NaN, 0 for NaN
                    binary_data = (~np.isnan(ds[var])).astype(int)

                    # Plot binary data
                    site_color = colors[i] if colors else None
                    binary_data.plot(ax=ax, label=f"{site}", color=site_color, **plot_kwargs)

        # Retrieve metadata for titles and labels
        long_name = (var_to_longname or {}).get(var, var)
        unit = (var_to_units or {}).get(var, "")
        ax.set_title(f"{long_name} ({var})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Availability (1=Valid, 0=NaN)")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(-0.1, 1.1)  # Ensure binary values are clearly visible

    plt.tight_layout()

def plot_monthly_availability_table(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    variables: List[str],
    sites: List[str],
    var_to_longname: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Compute and display monthly data availability (%) in a color-coded heatmap table.

    One row per variable-site combination, one column per month (YYYY-MM).

    Colors:
        - Green:  > 90%
        - Yellow: 70% - 90%
        - Orange: 40% - 70%
        - Red:    < 40%

    Returns:
        pd.DataFrame: Matrix of percentage availability values.
    """
    availability_records = {}

    for var in variables:
        sensor = variable_to_sensor.get(var)
        if not sensor or sensor not in sensor_datasets:
            print(
                f"Warning: Variable '{var}' or sensor '{sensor}' missing. Skipping."
            )
            continue

        var_label = (var_to_longname or {}).get(var, var)

        for site in sites:
            if site not in sensor_datasets[sensor]:
                continue

            ds = sensor_datasets[sensor][site]
            if var not in ds:
                continue

            da = ds[var]
            time_coord = "time" if "time" in da.coords else "t"

            # Create boolean mask (1 = valid, 0 = NaN)
            valid_mask = ~np.isnan(da)

            # Resample monthly: compute ratio of valid timesteps / total expected timesteps
            # Using 'count' for valid timesteps and 'size' for total expected timesteps in month
            monthly_valid = valid_mask.resample({time_coord: "1MS"}).sum()
            monthly_total = valid_mask.resample({time_coord: "1MS"}).count()

            monthly_pct = (monthly_valid / monthly_total) * 100.0

            # Convert to Series indexed by YYYY-MM
            s = monthly_pct.to_series()
            s.index = s.index.strftime("%Y-%m")

            row_name = f"{site} — {var_label}"
            availability_records[row_name] = s

    if not availability_records:
        raise ValueError("No matching data found to construct table.")

    # Combine into single DataFrame (Rows: Variable per Site, Columns: Months)
    df_avail = pd.DataFrame(availability_records).T.sort_index()

    # Define custom discrete colormap matching specified thresholds
    # Thresholds: [<40%, 40-70%, 70-90%, >90%]
    bounds = [0, 40, 70, 90, 100]
    hex_colors = ["#d7191c", "#fdae61", "#ffffbf", "#1a9641"]  # Red, Orange, Yellow, Green
    cmap = mcolors.ListedColormap(hex_colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot Setup
    if figsize is None:
        figsize = (max(12, len(df_avail.columns) * 0.8), len(df_avail) * 0.6 + 2)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_avail,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": "Availability (%)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title("Data Availability Matrix by Site and Variable (%)", fontsize=14, pad=15)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Site — Variable", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    #return df_avail

## PLOTTING ##

## Time plots
def format_time_plot(
    plot_data: Union[xr.DataArray, Tuple[Any, Any]],
    ax: plt.Axes,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    marker: Optional[str] = None,
    # Titles and Labels
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    unit: Optional[str] = None,
    # Axis Limits
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    xlim: Optional[Tuple[Any, Any]] = None,
    # Time Formatting
    time_format: Optional[str] = None,  # e.g., '%H:%M', '%Y-%m-%d', '%b %d'
    show_date_stamp: bool = False,      # Adds year-month-day annotation
    # Grid & Spacing
    grid: bool = True,
    grid_style: Dict[str, Any] = None,
    # Legend Controls
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_outside: bool = False,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    # Spans & Reference Lines
    vlines: Optional[List[Dict[str, Any]]] = None,
    hlines: Optional[List[Dict[str, Any]]] = None,
    vspans: Optional[List[Dict[str, Any]]] = None,
    **plot_kwargs: Any,
) -> plt.Axes:
    """
    Generic, standardized core routine for all temporal plots (time series, 
    diurnal cycles, normalized event progressions).

    Args:
        plot_data: xarray.DataArray or a tuple of (x, y) arrays.
        ax: matplotlib Axes instance to plot onto.
        label: Legend label string.
        color: Line color.
        linestyle: Line style (e.g., '-', '--', ':').
        linewidth: Line width.
        marker: Point marker style.
        title: Subplot title. Pass 'off' or '' to clear.
        xlabel: Custom x-axis label.
        ylabel: Custom y-axis label.
        unit: Optional unit string appended as '[unit]' to ylabel if provided.
        ylim: Tuple of (ymin, ymax).
        xlim: Tuple of (xmin, xmax).
        time_format: Strftime format for x-axis dates (e.g., '%H:%M').
        show_date_stamp: If True, adds a 'YYYY-MMM-DD' date tag at the bottom-left.
        grid: Whether to draw grid lines.
        grid_style: Dictionary of kwargs for grid formatting.
        show_legend: Toggle legend display.
        legend_loc: Legend position string.
        legend_outside: If True, places legend outside plot to the right.
        legend_kwargs: Extra parameters passed to ax.legend().
        vlines: List of dicts specifying axvline parameters (e.g., [{'x': peak, 'color': 'red'}]).
        hlines: List of dicts specifying axhline parameters.
        vspans: List of dicts specifying axvspan parameters for highlighted regions.
        **plot_kwargs: Keyword arguments passed to xarray.plot() or ax.plot().

    Returns:
        plt.Axes: The formatted matplotlib axes object.
    """
    grid_style = grid_style or {"linestyle": ":", "alpha": 0.6}
    legend_kwargs = legend_kwargs or {}

    # -------------------------------------------------------------------------
    # 1. EXECUTE PLOT
    # -------------------------------------------------------------------------
    if isinstance(plot_data, xr.DataArray):
        # Extract metadata automatically if available
        if ylabel is None and "long_name" in plot_data.attrs:
            ylabel = plot_data.attrs["long_name"]
        if unit is None and "units" in plot_data.attrs:
            unit = plot_data.attrs["units"]

        plot_data.plot(
            ax=ax,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            **plot_kwargs,
        )
    elif isinstance(plot_data, tuple) and len(plot_data) == 2:
        x, y = plot_data
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            **plot_kwargs,
        )
    else:
        raise ValueError("plot_data must be an xarray.DataArray or a tuple of (x, y).")

    # -------------------------------------------------------------------------
    # 2. TITLES & LABELS
    # -------------------------------------------------------------------------
    if title == "off":
        ax.set_title("")
    elif title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

    if ylabel is not None:
        label_text = f"{ylabel} [{unit}]" if unit else ylabel
        ax.set_ylabel(label_text, fontsize=10)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10)

    # -------------------------------------------------------------------------
    # 3. DATE & TIME FORMATTING
    # -------------------------------------------------------------------------
    if time_format is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))

    if show_date_stamp:
        try:
            if isinstance(plot_data, xr.DataArray):
                start_val = pd.to_datetime(plot_data.coords[plot_data.dims[0]].values[0])
            else:
                start_val = pd.to_datetime(plot_data[0][0])
            
            date_str = start_val.strftime("%Y-%b-%d")
            ax.text(
                0.01,
                -0.18,
                date_str,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="gray",
            )
        except Exception:
            pass  # Fall back gracefully if x-axis is numeric rather than datetime

    # -------------------------------------------------------------------------
    # 4. REFERENCE LINES & HIGHLIGHT SPANS
    # -------------------------------------------------------------------------
    if vlines:
        for vl in vlines:
            ax.axvline(**vl)
    if hlines:
        for hl in hlines:
            ax.axhline(**hl)
    if vspans:
        for span in vspans:
            ax.axvspan(**span)

    # -------------------------------------------------------------------------
    # 5. AXIS LIMITS & GRID
    # -------------------------------------------------------------------------
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)

    if grid:
        ax.grid(True, **grid_style)

    # -------------------------------------------------------------------------
    # 6. LEGEND MANAGEMENT
    # -------------------------------------------------------------------------
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            # Deduplicate labels while preserving order
            by_label = dict(zip(labels, handles))
            if legend_outside:
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    bbox_to_anchor=(1.02, 1.0),
                    loc="upper left",
                    frameon=True,
                    **legend_kwargs,
                )
            else:
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc=legend_loc,
                    frameon=True,
                    **legend_kwargs,
                )

    return ax

def plot_per_site_multiple_vars(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    variables: List[str],
    sites: List[str],
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
    var_to_units: Optional[Dict[str, str]] = None,
    var_to_longname: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (15, 3),
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    **plot_kwargs: Any,
) -> None:
    """
    Plots time series for each site on a dedicated subplot with all specified 
    variables overlaid, utilizing format_time_plot for standardized formatting.
    """
    if not sensor_datasets or not variables or not sites:
        raise ValueError("No datasets, variables, or sites provided.")

    if variable_to_sensor is None:
        raise ValueError("variable_to_sensor dictionary is required.")

    n_sites = len(sites)
    fig, axes = plt.subplots(n_sites, 1, figsize=(figsize[0], figsize[1] * n_sites), sharex=True, sharey=True)
    if n_sites == 1:
        axes = [axes]

    # Compute global y-axis limits independently if not provided
    global_min, global_max = ymin, ymax

    if global_min is None:
        for site in sites:
            for var in variables:
                da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
                if da is not None:
                    v_min = float(da.min().values)
                    global_min = v_min if global_min is None else min(global_min, v_min)

    if global_max is None:
        for site in sites:
            for var in variables:
                da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
                if da is not None:
                    v_max = float(da.max().values)
                    global_max = v_max if global_max is None else max(global_max, v_max)

    # Plot variables site by site
    for ax, site in zip(axes, sites):
        for var in variables:
            da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
            if da is not None:
                long_name = (var_to_longname or {}).get(var, var)
                unit = (var_to_units or {}).get(var, "")
                label_str = f"{long_name} [{unit}]" if unit else long_name

                # Delegate line plotting and label/grid styling to format_time_plot
                format_time_plot(
                    plot_data=da,
                    ax=ax,
                    label=label_str,
                    title=f"Site: {site}",
                    ylabel="Value",
                    ylim=(global_min, global_max),
                    show_legend=True,
                    legend_loc="upper right",
                    **plot_kwargs,
                )

    plt.tight_layout()

def plot_per_var_multiple_sites(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    variables: List[str],
    sites: List[str],
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
    var_to_units: Optional[Dict[str, str]] = None,
    var_to_longname: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    ymin: Optional[List[Optional[float]]] = None,
    ymax: Optional[List[Optional[float]]] = None,
    colors: Optional[List[str]] = None,
    **plot_kwargs: Any,
) -> None:
    """
    Plots time series for each variable in a single column, displaying all specified
    sites overlaid with distinct colors. Standardises formatting via format_time_plot.
    """
    if not sensor_datasets or not variables or not sites:
        raise ValueError("No datasets, variables, or sites provided.")

    if variable_to_sensor is None:
        raise ValueError("variable_to_sensor dictionary is required.")

    if ymin is not None and len(ymin) != len(variables):
        raise ValueError("Length of ymin must match the number of variables.")
    if ymax is not None and len(ymax) != len(variables):
        raise ValueError("Length of ymax must match the number of variables.")
    if colors is not None and len(colors) != len(sites):
        raise ValueError("Length of colors must match the number of sites.")

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1] * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    y_mins = ymin or [None] * n_vars
    y_maxs = ymax or [None] * n_vars

    # Plot variable by variable across all sites
    for ax, var, user_ymin, user_ymax in zip(axes, variables, y_mins, y_maxs):
        sensor = variable_to_sensor.get(var)
        if sensor is None:
            continue

        # Compute per-variable limits independently if not user-defined
        calc_ymin, calc_ymax = user_ymin, user_ymax

        if calc_ymin is None:
            for site in sites:
                da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
                if da is not None:
                    v_min = float(da.min().values)
                    calc_ymin = v_min if calc_ymin is None else min(calc_ymin, v_min)

        if calc_ymax is None:
            for site in sites:
                da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
                if da is not None:
                    v_max = float(da.max().values)
                    calc_ymax = v_max if calc_ymax is None else max(calc_ymax, v_max)

        # Plot each site for the given variable
        for i, site in enumerate(sites):
            da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
            if da is not None:
                site_color = colors[i] if colors else None
                long_name = (var_to_longname or {}).get(var, var)
                unit = (var_to_units or {}).get(var, "")

                # Offload line plotting and axis styling to format_time_plot
                format_time_plot(
                    plot_data=da,
                    ax=ax,
                    label=site,
                    color=site_color,
                    title=f"{long_name} ({var})",
                    ylabel=long_name,
                    unit=unit,
                    ylim=(calc_ymin, calc_ymax),
                    show_legend=True,
                    legend_loc="upper right",
                    **plot_kwargs,
                )

    plt.tight_layout()

## Scatter plots
def format_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    c: Optional[np.ndarray] = None,
    # Titles and Labels
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cbar_label: Optional[str] = None,
    # Axis Limits (Independent Min/Max)
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    clim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    # Statistical Annotations and Reference Lines
    show_corr: bool = True,
    show_fit: bool = True,
    show_oneone: bool = False,
    corr_pos: Tuple[float, float] = (0.05, 0.85),
    # Styling and Grid
    cmap: str = "viridis",
    grid: bool = True,
    grid_style: Optional[Dict[str, Any]] = None,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    fig: Optional[plt.Figure] = None,
    **scatter_kwargs: Any,
) -> Tuple[plt.Axes, Optional[Any]]:
    """
    Generic core routine for scatter plots. Manages points, colorbars,
    linear regression, correlation statistics, reference lines, and limits.
    """
    grid_style = grid_style or {"linestyle": ":", "alpha": 0.7}
    
    # Configure scatter defaults
    default_kwargs = {
        "alpha": 0.8 if c is not None else 0.6,
        "edgecolors": "none",
        "s": 20,
    }
    if c is not None:
        default_kwargs["cmap"] = cmap
    default_kwargs.update(scatter_kwargs)

    # Color limits handling
    c_min, c_max = (None, None) if clim is None else clim

    # 1. Plot Scatter Points
    if c is not None:
        sc = ax.scatter(x, y, c=c, vmin=c_min, vmax=c_max, **default_kwargs)
    else:
        sc = ax.scatter(x, y, **default_kwargs)

    # 2. Optional Colorbar
    cbar = None
    if c is not None and fig is not None:
        cbar = fig.colorbar(sc, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, fontweight="bold")

    # 3. Best Fit Line and Correlation Computation
    slope, intercept, r_val, _, _ = stats.linregress(x, y)

    if show_fit:
        x_range = np.array([x.min(), x.max()])
        y_fit = slope * x_range + intercept
        sign = "+" if intercept >= 0 else "-"
        
        # Build regression label appending r value at the end
        eq_label = f"$y = {slope:.2f}x {sign} {abs(intercept):.2f}$ ($r = {r_val:.3f}$)"
        ax.plot(x_range, y_fit, color="crimson", linestyle="--", linewidth=2, label=eq_label)

    # 4. 1:1 Reference Line
    if show_oneone:
        line_min = max(x.min(), y.min())
        line_max = min(x.max(), y.max())
        ax.plot([line_min, line_max], [line_min, line_max], color="black", linestyle=":", linewidth=1.5, label="1:1 line")

    # 5. Correlation Text Annotation (Only shown if fit is off to prevent duplicating r)
    if show_corr and not show_fit:
        annotation = f"$r = {r_val:.3f}$"
        ax.text(
            corr_pos[0],
            corr_pos[1],
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    # 6. Titles, Labels, Legend & Grid
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc=legend_loc)

    if grid:
        ax.grid(True, **grid_style)

    # 7. Apply Axis Limits (Independent min/max check)
    if xlim is not None:
        left, right = xlim
        if left is not None or right is not None:
            ax.set_xlim(left=left, right=right)

    if ylim is not None:
        bottom, top = ylim
        if bottom is not None or top is not None:
            ax.set_ylim(bottom=bottom, top=top)

    return ax, cbar

def align_and_clean_data(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    vars_and_sites: Sequence[Tuple[str, str]],
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
) -> List[np.ndarray]:
    """
    Extracts, temporally aligns, flattens, and removes NaN/non-finite values
    across 2 or 3 variable-site pairs.
    """
    das = []
    for var, site in vars_and_sites:
        da = extract_variable_series(sensor_datasets, var, site, variable_to_sensor)
        if da is None:
            raise KeyError(f"Variable '{var}' at site '{site}' could not be retrieved.")
        das.append(da)

    # Temporally align all series
    aligned_das = xr.align(*das, join="inner")
    
    # Flatten arrays
    flat_arrays = [da.values.flatten() for da in aligned_das]

    # Create joint finite mask across all variables
    mask = np.ones(flat_arrays[0].shape, dtype=bool)
    for arr in flat_arrays:
        mask &= np.isfinite(arr)

    cleaned_arrays = [arr[mask] for arr in flat_arrays]

    if len(cleaned_arrays[0]) == 0:
        raise ValueError("No overlapping, valid data points found across the specified variables and sites.")

    return cleaned_arrays

def plot_bivariate_scatter(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    var1: str,
    var2: str,
    site1: str,
    site2: str,
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
    var_to_units: Optional[Dict[str, str]] = None,
    var_to_longname: Optional[Dict[str, str]] = None,
    min_val: Optional[List[Optional[float]]] = None,
    max_val: Optional[List[Optional[float]]] = None,
    show_corr: bool = True,
    show_fit: bool = True,
    show_oneone: bool = False,
    figsize: Tuple[int, int] = (7, 6),
    **scatter_kwargs: Any,
) -> None:
    """Plots var1 vs var2 using format_scatter_plot and align_and_clean_data."""
    if min_val is not None and len(min_val) != 2:
        raise ValueError("min_val must be a sequence of exactly 2 elements: [var1_min, var2_min]")
    if max_val is not None and len(max_val) != 2:
        raise ValueError("max_val must be a sequence of exactly 2 elements: [var1_max, var2_max]")

    # 1. Align and Clean
    x_clean, y_clean = align_and_clean_data(
        sensor_datasets, [(var1, site1), (var2, site2)], variable_to_sensor
    )

    # 2. Metadata Labels
    xlabel = _format_var_label(var1, site1, var_to_units, var_to_longname)
    ylabel = _format_var_label(var2, site2, var_to_units, var_to_longname)
    title = f"Site: {site1}\n{var1} vs {var2}" if site1 == site2 else f"{var1} ({site1}) vs {var2} ({site2})"

    # 3. Limit Pairs
    xlim = (min_val[0] if min_val else None, max_val[0] if max_val else None)
    ylim = (min_val[1] if min_val else None, max_val[1] if max_val else None)

    # 4. Render
    fig, ax = plt.subplots(figsize=figsize)
    format_scatter_plot(
        x=x_clean,
        y=y_clean,
        ax=ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        show_corr=show_corr,
        show_fit=show_fit,
        show_oneone=show_oneone,
        **scatter_kwargs,
    )

    plt.tight_layout()

def plot_trivariate_scatter(
    sensor_datasets: Dict[str, Dict[str, xr.Dataset]],
    var1: str,
    var2: str,
    var3: str,
    site1: str,
    site2: str,
    site3: str,
    variable_to_sensor: Dict[str, str] = variable_to_sensor,
    var_to_units: Optional[Dict[str, str]] = None,
    var_to_longname: Optional[Dict[str, str]] = None,
    min_val: Optional[List[Optional[float]]] = None,
    max_val: Optional[List[Optional[float]]] = None,
    min3: Optional[float] = None,
    max3: Optional[float] = None,
    show_corr: bool = True,
    show_fit: bool = True,
    show_oneone: bool = False,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 6),
    **scatter_kwargs: Any,
) -> None:
    """Plots var1 vs var2 colored by var3 using format_scatter_plot and align_and_clean_data."""
    if min_val is not None and len(min_val) not in (2, 3):
        raise ValueError("min_val must be a sequence of 2 or 3 elements: [var1_min, var2_min, var3_min]")
    if max_val is not None and len(max_val) not in (2, 3):
        raise ValueError("max_val must be a sequence of 2 or 3 elements: [var1_max, var2_max, var3_max]")

    # Resolve min3/max3 priorities
    if min3 is None and min_val is not None and len(min_val) == 3:
        min3 = min_val[2]
    if max3 is None and max_val is not None and len(max_val) == 3:
        max3 = max_val[2]

    # 1. Align and Clean
    x_clean, y_clean, c_clean = align_and_clean_data(
        sensor_datasets, [(var1, site1), (var2, site2), (var3, site3)], variable_to_sensor
    )

    # 2. Metadata Labels
    xlabel = _format_var_label(var1, site1, var_to_units, var_to_longname)
    ylabel = _format_var_label(var2, site2, var_to_units, var_to_longname)
    cbar_label = _format_var_label(var3, site3, var_to_units, var_to_longname)
    
    if site1 == site2 == site3:
        title = f"Site: {site1}\n{var1} vs {var2} (colored by {var3})"
    else:
        title = f"{var1} ({site1}) vs {var2} ({site2})\nColored by {var3} ({site3})"

    # 3. Limit Pairs
    xlim = (min_val[0] if min_val else None, max_val[0] if max_val else None)
    ylim = (min_val[1] if min_val else None, max_val[1] if max_val else None)

    # 4. Render
    fig, ax = plt.subplots(figsize=figsize)
    format_scatter_plot(
        x=x_clean,
        y=y_clean,
        c=c_clean,
        ax=ax,
        fig=fig,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_label=cbar_label,
        xlim=xlim,
        ylim=ylim,
        clim=(min3, max3),
        show_corr=show_corr,
        show_fit=show_fit,
        show_oneone=show_oneone,
        cmap=cmap,
        **scatter_kwargs,
    )

    plt.tight_layout()



# 