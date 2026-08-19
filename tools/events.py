from info import *
from imports import *
from tools_generic import *

class Event:
    """
    Encapsulates a single physical event at a specific site across all variables.

    Attributes:
        site (str): Name or ID of the observation site.
        event_id (int): Unique identifier for the event.
        start_time (pd.Timestamp): Start timestamp of the event.
        end_time (pd.Timestamp): End timestamp of the event.
        ds (xr.Dataset): Slice of the multi-variable Dataset covering the event window.
        trigger_variable (str): Name of the variable used to detect the event.
        time_dim (str): Name of the time dimension in the Dataset.
    """

    def __init__(
        self,
        site: str,
        event_id: int,
        ds: xr.Dataset,
        trigger_variable: str,
        buffer_window: pd.Timedelta = pd.Timedelta(0),
        time_dim: str = "time",
    ) -> None:
        self.site = site
        self.event_id = event_id
        self.trigger_variable = trigger_variable
        self.time_dim = time_dim

        # Extract core event time bounds from the provided dataset slice
        self.start_time = pd.Timestamp(ds[self.time_dim].values[0])
        self.end_time = pd.Timestamp(ds[self.time_dim].values[-1])
        self.duration = self.end_time - self.start_time

        # Store the multi-variable xarray Dataset
        self.ds = ds

    @property
    def peak_time(self) -> pd.Timestamp:
        """Timestamp corresponding to the maximum value of the trigger variable."""
        peak_idx = self.ds[self.trigger_variable].argmax(dim=self.time_dim).values
        return pd.Timestamp(self.ds[self.time_dim].values[peak_idx])

    @property
    def peak_value(self) -> float:
        """Peak value of the trigger variable during the event."""
        return float(self.ds[self.trigger_variable].max(dim=self.time_dim).values)

    def get_relative_time_ds(
        self, align_on: str = "start", normalize_duration: bool = False
    ) -> xr.Dataset:
        """
        Returns a copy of the dataset with normalized or relative time coordinates.

        Args:
            align_on (str): 'start' sets t=0 at event start; 'peak' sets t=0 at peak intensity.
            normalize_duration (bool): If True, normalises time coordinate to 0.0 - 100.0% completion.

        Returns:
            xr.Dataset: Dataset with a new coordinate 'rel_time' replacing or augmenting time.
        """
        ds_rel = self.ds.copy(deep=True)
        time_vals = pd.to_datetime(ds_rel[self.time_dim].values)

        if normalize_duration:
            if self.duration == pd.Timedelta(0):
                rel_coords = np.zeros(len(time_vals))
            else:
                rel_coords = (time_vals - self.start_time) / self.duration * 100.0
            ds_rel = ds_rel.assign_coords(norm_time=(self.time_dim, rel_coords))
            ds_rel.norm_time.attrs["units"] = "% completion"
        else:
            if align_on == "start":
                ref_time = self.start_time
            elif align_on == "peak":
                ref_time = self.peak_time
            else:
                raise ValueError("align_on must be either 'start' or 'peak'.")

            rel_coords = (time_vals - ref_time).total_seconds() / 3600.0  # Hours
            ds_rel = ds_rel.assign_coords(rel_time_hours=(self.time_dim, rel_coords))
            ds_rel.rel_time_hours.attrs["units"] = "hours"

        return ds_rel

    def summary_stats(self, variables: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculates summary statistics for specified variables during the event."""
        if variables is None:
            variables = list(self.ds.data_vars.keys())

        stats = {
            "site": self.site,
            "event_id": self.event_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_hours": self.duration.total_seconds() / 3600.0,
            "peak_time": self.peak_time,
        }

        for var in variables:
            if var in self.ds:
                data = self.ds[var].values
                stats[f"{var}_mean"] = float(np.nanmean(data))
                stats[f"{var}_max"] = float(np.nanmax(data))
                stats[f"{var}_min"] = float(np.nanmin(data))
                stats[f"{var}_std"] = float(np.nanstd(data))

        return stats

    def __repr__(self) -> str:
        return (
            f"<Event {self.event_id} | Site: {self.site} | "
            f"Duration: {self.duration} | Start: {self.start_time}>"
        )

class EventCollection:
    """
    Container class for managing, filtering, and analysing a list of Event objects.
    """

    def __init__(self, events: Optional[List[Event]] = None) -> None:
        self.events: List[Event] = events if events is not None else []

    def add_event(self, event: Event) -> None:
        """Adds an Event instance to the collection."""
        self.events.append(event)

    def filter_by_site(self, site: str) -> "EventCollection":
        """Returns a new EventCollection containing only events from the specified site."""
        filtered = [e for e in self.events if e.site == site]
        return EventCollection(filtered)

    def filter_by_duration(
        self, min_duration: Optional[pd.Timedelta] = None, max_duration: Optional[pd.Timedelta] = None
    ) -> "EventCollection":
        """Filters events based on minimum and/or maximum duration thresholds."""
        filtered = self.events
        if min_duration is not None:
            filtered = [e for e in filtered if e.duration >= min_duration]
        if max_duration is not None:
            filtered = [e for e in filtered if e.duration <= max_duration]
        return EventCollection(filtered)

    def to_catalog(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Converts all contained events into a tabular pandas DataFrame catalog.

        Args:
            variables (List[str], optional): Ancillary variables to include summary metrics for.

        Returns:
            pd.DataFrame: Table containing metadata and summary statistics for every event.
        """
        records = [e.summary_stats(variables=variables) for e in self.events]
        return pd.DataFrame(records)

    def compute_composite(
        self, variable: str, num_bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a Superposed Epoch Analysis (composite profile) across all events 
        by interpolating the target variable onto a uniform normalized time grid (0-100%).

        Args:
            variable (str): Name of the variable to construct the composite for.
            num_bins (int): Number of time points in the normalized 0-100% grid.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - grid: Relative time points (0 to 100%)
                - mean_profile: Ensemble mean values across all events
                - std_profile: Ensemble standard deviation across all events
        """
        norm_grid = np.linspace(0, 100, num_bins)
        interpolated_profiles = []

        for event in self.events:
            if variable not in event.ds:
                continue

            ds_norm = event.get_relative_time_ds(normalize_duration=True)
            x = ds_norm["norm_time"].values
            y = ds_norm[variable].values

            # Remove NaN values prior to 1D linear interpolation
            valid = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(valid) > 1:
                interp_y = np.interp(norm_grid, x[valid], y[valid])
                interpolated_profiles.append(interp_y)

        if not interpolated_profiles:
            raise ValueError(f"Variable '{variable}' not found in any events.")

        profiles_arr = np.array(interpolated_profiles)
        mean_profile = np.nanmean(profiles_arr, axis=0)
        std_profile = np.nanstd(profiles_arr, axis=0)

        return norm_grid, mean_profile, std_profile

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> Event:
        return self.events[index]

    def __repr__(self) -> str:
        return f"<EventCollection containing {len(self.events)} events>"

    def get_site(self, site: str) -> "EventCollection":
        """
        Extracts a new EventCollection containing only events belonging to the specified site.

        Parameters
        ----------
        site : str
            The site identifier to filter by (e.g., 'd17').

        Returns
        -------
        EventCollection
            A new EventCollection instance containing matching events.
        """
        site_events = [e for e in self.events if getattr(e, "site", None) == site]
        return EventCollection(site_events)


    """
    Container class for managing, filtering, and analysing a list of Event objects.
    """

    def __init__(self, events: Optional[List[Event]] = None) -> None:
        self.events: List[Event] = events if events is not None else []

    def add_event(self, event: Event) -> None:
        """Adds an Event instance to the collection."""
        self.events.append(event)

    def filter_by_site(self, site: str) -> "EventCollection":
        """Returns a new EventCollection containing only events from the specified site."""
        filtered = [e for e in self.events if e.site == site]
        return EventCollection(filtered)

    def filter_by_duration(
        self, min_duration: Optional[pd.Timedelta] = None, max_duration: Optional[pd.Timedelta] = None
    ) -> "EventCollection":
        """Filters events based on minimum and/or maximum duration thresholds."""
        filtered = self.events
        if min_duration is not None:
            filtered = [e for e in filtered if e.duration >= min_duration]
        if max_duration is not None:
            filtered = [e for e in filtered if e.duration <= max_duration]
        return EventCollection(filtered)

    def to_catalog(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Converts all contained events into a tabular pandas DataFrame catalog.

        Args:
            variables (List[str], optional): Ancillary variables to include summary metrics for.

        Returns:
            pd.DataFrame: Table containing metadata and summary statistics for every event.
        """
        records = [e.summary_stats(variables=variables) for e in self.events]
        return pd.DataFrame(records)

    def compute_composite(
        self, variable: str, num_bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a Superposed Epoch Analysis (composite profile) across all events 
        by interpolating the target variable onto a uniform normalized time grid (0-100%).

        Args:
            variable (str): Name of the variable to construct the composite for.
            num_bins (int): Number of time points in the normalized 0-100% grid.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - grid: Relative time points (0 to 100%)
                - mean_profile: Ensemble mean values across all events
                - std_profile: Ensemble standard deviation across all events
        """
        norm_grid = np.linspace(0, 100, num_bins)
        interpolated_profiles = []

        for event in self.events:
            if variable not in event.ds:
                continue

            ds_norm = event.get_relative_time_ds(normalize_duration=True)
            x = ds_norm["norm_time"].values
            y = ds_norm[variable].values

            # Remove NaN values prior to 1D linear interpolation
            valid = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(valid) > 1:
                interp_y = np.interp(norm_grid, x[valid], y[valid])
                interpolated_profiles.append(interp_y)

        if not interpolated_profiles:
            raise ValueError(f"Variable '{variable}' not found in any events.")

        profiles_arr = np.array(interpolated_profiles)
        mean_profile = np.nanmean(profiles_arr, axis=0)
        std_profile = np.nanstd(profiles_arr, axis=0)

        return norm_grid, mean_profile, std_profile

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> Event:
        return self.events[index]

    def __repr__(self) -> str:
        return f"<EventCollection containing {len(self.events)} events>"


class EventDetector:
    def __init__(
        self,
        threshold: float = 1.0,
        min_timesteps: int = 24,
        buffer_timesteps: int = 2,
        variable_to_sensor: Dict[str, str] = variable_to_sensor,
    ):
        self.threshold = threshold
        self.min_timesteps = min_timesteps
        self.buffer_timesteps = buffer_timesteps
        self.variable_to_sensor = variable_to_sensor

    def detect_events(
        self,
        data: Dict[str, Dict[str, xr.Dataset]],
        variable: str,
        additional_variables: Optional[List[str]] = None,
    ) -> "EventCollection":
        events = []
        target_vars = [variable] + (additional_variables or [])
        global_event_id = 0

        # Unique set of sites across all sensor datasets
        sites = set(site for sensor_dict in data.values() for site in sensor_dict)

        for site_id in sites:
            # 1. Merge all sensor datasets for this site into a single aligned dataset
            merged_ds = extract_site_dataset(
                sensor_datasets=data,
                site=site_id,
                variables=target_vars,
                variable_to_sensor=self.variable_to_sensor,
            )

            if merged_ds == xr.Dataset() or variable not in merged_ds:
                continue

            da_trigger = merged_ds[variable]
            is_event = da_trigger.values > self.threshold

            # 2. Find start and end indices of contiguous events
            event_starts, event_ends = self._find_event_indices(is_event)

            for start_idx, end_idx in zip(event_starts, event_ends):
                buf_start = max(0, start_idx - self.buffer_timesteps)
                buf_end = min(len(da_trigger) - 1, end_idx + self.buffer_timesteps)

                time_vals = da_trigger.time.values
                start_time = pd.to_datetime(time_vals[buf_start])
                end_time = pd.to_datetime(time_vals[buf_end])

                # Slice merged dataset across the buffered event window
                event_ds = merged_ds.sel(time=slice(start_time, end_time))

                # Instantiation matching your existing Event signature
                event = Event(
                    site=site_id,
                    event_id=global_event_id,
                    ds=event_ds,
                    trigger_variable=variable,
                )
                events.append(event)
                global_event_id += 1

        return EventCollection(events)

    def _find_event_indices(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        durations = (ends - starts) + 1
        valid = durations >= self.min_timesteps

        return starts[valid], ends[valid]

    def _find_event_indices(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        durations = (ends - starts) + 1
        valid = durations >= self.min_timesteps

        return starts[valid], ends[valid]

###############
## Functions ##
###############

def plot_single_event(event, variables=None, **kwargs):
    if variables is None:
        variables = list(event.ds.data_vars.keys())

    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 2.5 * len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        format_time_plot(
            plot_data=event.ds[var],
            ax=ax,
            label=var,
            title=f"Event #{event.event_id}: {var}",
            vlines=[{"x": event.peak_time, "color": "darkred", "linestyle": "--", "label": "Peak"}],
            vspans=[{"xmin": event.start_time, "xmax": event.end_time, "color": "red", "alpha": 0.12, "label": "Event Window"}],
            **kwargs,
        )
    plt.tight_layout()

def plot_event_collection_traces(
    collection: "EventCollection",
    variable: str,
    align_to: Literal["peak_time", "start_time"] = "peak_time",
    time_unit: str = "h",
    cmap_name: str = "viridis",
    alpha: float = 0.35,
    linewidth: float = 0.8,
    show_mean: bool = True,
    var_to_units: Optional[Dict[str, str]] = var_to_units,
    var_to_longname: Optional[Dict[str, str]] = var_to_longname,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots event collection time series using the core format_time_plot utility.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    valid_events = [e for e in collection if variable in e.ds]
    n_events = len(valid_events)

    if n_events == 0:
        raise ValueError(f"Variable '{variable}' not found in any event of the collection.")

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(1, n_events - 1))

    all_rel_times = []
    all_values = []

    long_name = (var_to_longname or {}).get(variable, variable)
    unit = (var_to_units or {}).get(variable, None)
    align_str = align_to.replace("_", " ").title()
    xlabel = f"Relative Time [{time_unit}] from {align_str}"

    # 1. Plot individual event traces via format_time_plot
    for idx, event in enumerate(valid_events):
        da = event.ds[variable]
        ref_time = getattr(event, align_to)

        time_vals = pd.to_datetime(da[event.time_dim].values)
        rel_time = (time_vals - pd.to_datetime(ref_time)) / pd.Timedelta(1, time_unit)
        values = da.values

        color = cmap(norm(idx))

        # Delegate plotting and basic trace setup to format_time_plot
        format_time_plot(
            plot_data=(rel_time, values),
            ax=ax,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            show_legend=False,  # Suppress individual trace legends
            grid=False,         # Apply grid once at the end
        )

        all_rel_times.append(rel_time)
        all_values.append(values)

    # 2. Colorbar for event sequence
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Event Index", rotation=270, labelpad=15)

    # 3. Ensemble Mean trace via format_time_plot
    if show_mean and n_events > 0:
        min_t = min(np.min(rt) for rt in all_rel_times)
        max_t = max(np.max(rt) for rt in all_rel_times)
        common_grid = np.linspace(min_t, max_t, 250)

        interp_matrix = np.array([
            np.interp(common_grid, rt, vals, left=np.nan, right=np.nan)
            for rt, vals in zip(all_rel_times, all_values)
        ])

        mean_trace = np.nanmean(interp_matrix, axis=0)

        format_time_plot(
            plot_data=(common_grid, mean_trace),
            ax=ax,
            label=f"Ensemble Mean (N={n_events})",
            color="black",
            linewidth=2.0,
            show_legend=False,
        )

    # 4. Final formatting pass to set labels, grid, and reference line
    vlines = [{"x": 0, "color": "darkred", "linestyle": "--", "linewidth": 1.0, "alpha": 0.8, "label": align_str}]

    # Dummy call to format_time_plot to apply titles, labels, vlines, and deduplicated legend
    format_time_plot(
        plot_data=(np.array([]), np.array([])),  # Empty tuple just to trigger formatting block
        ax=ax,
        title=f"Event Traces: {long_name} (Aligned on {align_str})",
        xlabel=xlabel,
        ylabel=long_name,
        unit=unit,
        vlines=vlines,
        grid=True,
        show_legend=show_mean,
    )

    plt.tight_layout()
    return fig, ax
### Composites ###
def compute_event_composite(
    events: List["Event"],
    variables: Optional[List[str]] = None,
    align_to: Literal["peak_time", "start_time"] = "peak_time",
    time_unit: str = "h",  # 'h' for hours, 'D' for days, etc.
) -> xr.Dataset:
    """
    Aligns a list of Event instances to a common relative time axis and computes 
    composite statistics (mean, std, median, q25, q75).

    Args:
        events: List of Event objects to average.
        variables: Variables to include. Defaults to all variables in the first event.
        align_to: Reference time anchor ('peak_time' or 'start_time').
        time_unit: Pandas frequency unit for the relative time offset ('h', 'D', 'm').

    Returns:
        xr.Dataset containing mean, std, median, q25, and q75 across all events.
    """
    if not events:
        raise ValueError("Event list is empty.")

    aligned_datasets = []

    for event in events:
        ds = event.ds.copy()

        # Determine reference timestamp
        ref_time = getattr(event, align_to)
        
        # Calculate relative time delta from reference
        time_vals = pd.to_datetime(ds[event.time_dim].values)
        relative_offsets = (time_vals - pd.to_datetime(ref_time)) / pd.Timedelta(1, time_unit)

        # Assign relative time as a new dimension/coordinate
        ds = ds.assign_coords(rel_time=("time", relative_offsets))
        ds = ds.swap_dims({event.time_dim: "rel_time"}).drop_vars(event.time_dim)

        # Select requested variables
        if variables:
            ds = ds[[v for v in variables if v in ds]]

        aligned_datasets.append(ds)

    # Combine along new event dimension
    composite_raw = xr.concat(aligned_datasets, dim="event_idx")

    # Calculate statistics across ensemble
    composite_ds = xr.Dataset(
        {
            f"{var}_mean": composite_raw[var].mean(dim="event_idx") for var in composite_raw.data_vars
        }
    )
    for var in composite_raw.data_vars:
        composite_ds[f"{var}_std"] = composite_raw[var].std(dim="event_idx")
        composite_ds[f"{var}_median"] = composite_raw[var].median(dim="event_idx")
        composite_ds[f"{var}_q25"] = composite_raw[var].quantile(0.25, dim="event_idx")
        composite_ds[f"{var}_q75"] = composite_raw[var].quantile(0.75, dim="event_idx")

    composite_ds.attrs["num_events"] = len(events)
    composite_ds.attrs["aligned_to"] = align_to
    composite_ds.attrs["time_unit"] = time_unit

    return composite_ds

def compute_grouped_event_composites(
    events: List["Event"],
    group_by_func: Callable[["Event"], str],
    variables: Optional[List[str]] = None,
    align_to: Literal["peak_time", "start_time"] = "peak_time",
    time_unit: str = "h",
) -> Dict[str, xr.Dataset]:
    """
    Groups a list of Event instances using a user-defined function (e.g., extracting 
    site or season) and computes composite datasets for each group independently.

    Args:
        events: List of Event objects.
        group_by_func: Function taking an Event and returning a string group label 
                       (e.g., lambda e: e.site or lambda e: get_season(e.peak_time)).
        variables: Variables to include in the composite.
        align_to: Reference time anchor ('peak_time' or 'start_time').
        time_unit: Pandas frequency unit for relative time offsets.

    Returns:
        A dictionary mapping each group label to its computed composite xr.Dataset.
    """
    # 1. Partition events into groups
    grouped_dict = defaultdict(list)
    for event in events:
        key = group_by_func(event)
        grouped_dict[key].append(event)

    # 2. Compute composite for each group using the core alignment logic
    composites = {}
    for group_name, group_events in grouped_dict.items():
        # Re-use the alignment logic for this subset
        composites[group_name] = compute_event_composite(
            events=group_events,
            variables=variables,
            align_to=align_to,
            time_unit=time_unit,
        )
        composites[group_name].attrs["group_label"] = group_name

    return composites

def plot_event_composite(
    composite_ds: xr.Dataset,
    variables: List[str],
    var_to_units: Optional[Dict[str, str]] = var_to_units,
    var_to_longname: Optional[Dict[str, str]] = var_to_longname,
    use_quantiles: bool = False,
    figsize: Tuple[int, int] = (12, 3),
    **plot_kwargs: Any,
) -> None:
    """
    Plots the composite mean and shading bounds (std or IQR) for each variable.
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1] * n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    time_unit = composite_ds.attrs.get("time_unit", "h")
    align_to = composite_ds.attrs.get("aligned_to", "peak_time")
    xlabel = f"Relative Time [{time_unit}] from {align_to.replace('_', ' ')}"

    for ax, var in zip(axes, variables):
        mean_key = f"{var}_mean"
        if mean_key not in composite_ds:
            continue

        da_mean = composite_ds[mean_key]
        x_vals = da_mean.rel_time.values

        # Determine shading bounds
        if use_quantiles:
            lower = composite_ds[f"{var}_q25"].values
            upper = composite_ds[f"{var}_q75"].values
            shade_label = "IQR (25th-75th)"
        else:
            std = composite_ds[f"{var}_std"].values
            lower = da_mean.values - std
            upper = da_mean.values + std
            shade_label = "±1 Std Dev"

        # Format line plot
        long_name = (var_to_longname or {}).get(var, var)
        unit = (var_to_units or {}).get(var, "")

        format_time_plot(
            plot_data=(x_vals, da_mean.values),
            ax=ax,
            label="Composite Mean",
            title=f"Composite: {long_name}",
            xlabel=xlabel,
            ylabel=long_name,
            unit=unit,
            vlines=[{"x": 0, "color": "darkred", "linestyle": "--", "label": f"{align_to}"}],
            show_legend=True,
            legend_loc="upper right",
            **plot_kwargs,
        )

        # Overlay uncertainty band
        ax.fill_between(x_vals, lower, upper, color="black", alpha=0.15, label=shade_label)

    plt.tight_layout()