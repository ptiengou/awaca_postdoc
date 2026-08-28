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

    def compute_integrated_flux(
        self, variable: str = "FluxMean2", format_exponential: bool = False
    ) -> Union[float, str]:
        """
        Computes time-integrated cumulative flux over the event duration in g/m^2.

        Integration of flux rate [g m^-2 s^-1] over time [s] yields total flux [g m^-2].

        Parameters
        ----------
        variable : str
            The variable to integrate (e.g. 'FluxMean2').
        format_exponential : bool
            If True, returns a string formatted as scientific notation 'A x 10^B' g m^-2.
            If False, returns the raw float in g m^-2.

        Returns
        -------
        Union[float, str]
            Integrated cumulative flux in g m^-2.
        """
        if variable not in self.ds:
            return np.nan

        da = self.ds[variable]
        times = pd.to_datetime(da[self.time_dim].values)
        values = da.values

        valid = ~np.isnan(values)
        if np.sum(valid) < 2:
            return np.nan

        # Convert timestamps to explicit total seconds from start to preserve g m^-2 units
        time_seconds = (times[valid] - times[valid][0]).total_seconds().values
        
        # Trapezoidal integration: [g m^-2 s^-1] * [s] -> [g m^-2]
        integrated_val = float(trapezoid(y=values[valid], x=time_seconds))

        if format_exponential:
            if integrated_val == 0 or np.isnan(integrated_val):
                return "0 g m^-2"
            mantissa, exponent = f"{integrated_val:.2e}".split("e")
            return f"{mantissa} \u00d7 10^{int(exponent)} g m^-2"

        return integrated_val       

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

    def get_duration_stats(self) -> Dict[str, float]:
        """Returns mean, std, median, min, and max event durations in hours."""
        if not self.events:
            return {}
        durations = np.array([e.duration.total_seconds() / 3600.0 for e in self.events])
        return {
            "mean_hours": float(np.mean(durations)),
            "std_hours": float(np.std(durations)),
            "median_hours": float(np.median(durations)),
            "min_hours": float(np.min(durations)),
            "max_hours": float(np.max(durations)),
        }

    def get_integrated_flux_stats(
        self, variable: str = "FluxMean2", print_summary: bool = True
    ) -> Dict[str, Union[float, str]]:
        """
        Calculates cumulative flux statistics (mean, std, sum, median) in g m^-2
        and provides a formatted pretty-printed summary.

        Parameters
        ----------
        variable : str
            The variable to integrate (default: 'FluxMean2').
        print_summary : bool
            If True, prints a nicely formatted summary table to the console.

        Returns
        -------
        Dict[str, Union[float, str]]
            Dictionary containing both raw floats and formatted scientific notation strings.
        """
        fluxes = [
            e.compute_integrated_flux(variable=variable, format_exponential=False)
            for e in self.events
        ]
        valid_fluxes = np.array([f for f in fluxes if not np.isnan(f)])

        if len(valid_fluxes) == 0:
            if print_summary:
                print(f"No valid data found for variable '{variable}'.")
            return {}

        mean_val = float(np.mean(valid_fluxes))
        std_val = float(np.std(valid_fluxes))
        sum_val = float(np.sum(valid_fluxes))
        median_val = float(np.median(valid_fluxes))
        min_val = float(np.min(valid_fluxes))
        max_val = float(np.max(valid_fluxes))

        def _to_exp(val: float) -> str:
            if val == 0:
                return "0"
            m, e = f"{val:.2e}".split("e")
            return f"{m} \u00d7 10^{int(e)}"

        stats = {
            "count": len(valid_fluxes),
            "mean_g_m2": mean_val,
            "std_g_m2": std_val,
            "sum_g_m2": sum_val,
            "median_g_m2": median_val,
            "min_g_m2": min_val,
            "max_g_m2": max_val,
            "mean_formatted": f"{_to_exp(mean_val)} g m\u207b\u00b2",
            "std_formatted": f"{_to_exp(std_val)} g m\u207b\u00b2",
            "sum_formatted": f"{_to_exp(sum_val)} g m\u207b\u00b2",
            "median_formatted": f"{_to_exp(median_val)} g m\u207b\u00b2",
        }

        return stats

    def get_diurnal_start_distribution(self) -> pd.Series:
        """Returns the count of event start times grouped by hour of the day (0-23)."""
        start_hours = [e.start_time.hour for e in self.events]
        counts = pd.Series(start_hours).value_counts().reindex(range(24), fill_value=0)
        counts.index.name = "hour"
        return counts
    
    def get_starting_month_distribution(self, normalize: bool = True) -> pd.Series:
        """
        Calculates the monthly distribution of event start times.

        Parameters
        ----------
        normalize : bool, default True
            If True, returns the proportion/percentage (0.0 to 1.0) of events 
            starting in each month. If False, returns raw counts.

        Returns
        -------
        pd.Series
            Series indexed by month number (1–12) containing the starting proportions 
            or counts.
        """
        if not self.events:
            return pd.Series(0.0 if normalize else 0, index=pd.RangeIndex(1, 13, name="month"))

        start_months = [e.start_time.month for e in self.events]
        counts = pd.Series(start_months).value_counts(normalize=normalize).reindex(range(1, 13), fill_value=0.0 if normalize else 0)
        counts.index.name = "month"
        
        return counts

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

    def detect_non_events(
    self,
    data: Dict[str, Dict[str, xr.Dataset]],
    variable: str,
    additional_variables: Optional[List[str]] = None,
    ) -> "EventCollection":
        """
        Detects contiguous periods where the trigger variable does NOT exceed the threshold.

        Parameters
        ----------
        data : Dict[str, Dict[str, xr.Dataset]]
            Nested dictionary of sensor datasets structured as {sensor: {site: dataset}}.
        variable : str
            Trigger variable name evaluated against the threshold.
        additional_variables : Optional[List[str]]
            Secondary variables to include in extracted xarray Dataset slices.

        Returns
        -------
        EventCollection
            Collection of non-event periods encapsulated as Event objects.
        """
        non_events = []
        target_vars = [variable] + (additional_variables or [])
        global_event_id = 0

        sites = set(site for sensor_dict in data.values() for site in sensor_dict)

        for site_id in sites:
            merged_ds = extract_site_dataset(
                sensor_datasets=data,
                site=site_id,
                variables=target_vars,
                variable_to_sensor=self.variable_to_sensor,
            )

            if merged_ds == xr.Dataset() or variable not in merged_ds:
                continue

            da_trigger = merged_ds[variable]
            
            # Invert the trigger condition to identify quiet/non-event steps
            is_non_event = da_trigger.values <= self.threshold

            # Find continuous non-event windows meeting min_timesteps
            ne_starts, ne_ends = self._find_event_indices(is_non_event)

            for start_idx, end_idx in zip(ne_starts, ne_ends):
                time_vals = da_trigger.time.values
                start_time = pd.to_datetime(time_vals[start_idx])
                end_time = pd.to_datetime(time_vals[end_idx])

                non_event_ds = merged_ds.sel(time=slice(start_time, end_time))

                non_event = Event(
                    site=site_id,
                    event_id=global_event_id,
                    ds=non_event_ds,
                    trigger_variable=variable,
                )
                non_events.append(non_event)
                global_event_id += 1

        return EventCollection(non_events)
    
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
    show_mean: bool = False,
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
    #return fig, ax

def plot_events_vs_nonevents_chronological(
    events: "EventCollection",
    non_events: "EventCollection",
    variables: List[str],
    site: Optional[str] = None,
    event_color: str = "crimson",
    non_event_color: str = "tab:blue",
    event_alpha: float = 0.25,
    var_to_units: Optional[Dict[str, str]] = None,
    var_to_longname: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots continuous chronological time series for specified variables with event regions shaded.
    """
    if site is not None:
        events = events.filter_by_site(site)
        non_events = non_events.filter_by_site(site)

    if len(events) == 0 and len(non_events) == 0:
        raise ValueError(f"No events or non-events found for site '{site}'." if site else "Collections are empty.")

    # 1. Combine and sort all periods chronologically by start_time
    all_periods = list(events.events) + list(non_events.events)
    all_periods.sort(key=lambda p: p.start_time)

    n_vars = len(variables)
    if figsize is None:
        figsize = (12, 3 * n_vars)

    fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=figsize, sharex=True)
    if n_vars == 1:
        axes = np.array([axes])

    for var_idx, var in enumerate(variables):
        ax = axes[var_idx]
        long_name = (var_to_longname or {}).get(var, var)
        unit = (var_to_units or {}).get(var, None)

        # Track global time min/max to prevent autoscale resets
        global_min_time = None
        global_max_time = None

        # 2. Plot time series slices chronologically
        for period in all_periods:
            if var not in period.ds:
                continue

            da = period.ds[var]
            
            # Convert explicitly to python datetime objects for reliable matplotlib parsing
            raw_times = pd.to_datetime(da[period.time_dim].values)
            plot_times = raw_times.to_pydatetime()
            values = da.values

            if len(plot_times) > 0:
                if global_min_time is None or plot_times[0] < global_min_time:
                    global_min_time = plot_times[0]
                if global_max_time is None or plot_times[-1] > global_max_time:
                    global_max_time = plot_times[-1]

            is_event = period in events.events
            color = event_color if is_event else non_event_color
            linewidth = 1.2 if is_event else 0.8

            format_time_plot(
                plot_data=(plot_times, values),
                ax=ax,
                color=color,
                linewidth=linewidth,
                show_legend=False,
                grid=False,
            )

        # 3. Highlight event windows
        for e in events.events:
            if var in e.ds:
                ax.axvspan(
                    e.start_time.to_pydatetime(),
                    e.end_time.to_pydatetime(),
                    color=event_color,
                    alpha=event_alpha,
                    linewidth=0,
                )

        # 4. Dummy elements for legend
        if var_idx == 0:
            ax.plot([], [], color=non_event_color, linewidth=1.0, label="Non-Event Baseline")
            ax.plot([], [], color=event_color, linewidth=1.5, label="Detected Event")
            ax.axvspan(
                global_min_time or all_periods[0].start_time.to_pydatetime(),
                global_min_time or all_periods[0].start_time.to_pydatetime(),
                color=event_color,
                alpha=event_alpha,
                label="Event Window",
            )

        # 5. Apply final formatting pass without resetting data bounds
        ylabel_str = f"{long_name} [{unit}]" if unit else long_name
        
        # Explicitly lock time limits if valid bounds exist
        if global_min_time and global_max_time:
            ax.set_xlim(global_min_time, global_max_time)

        format_time_plot(
            plot_data=(np.array([global_min_time]), np.array([np.nan])),  # Preserves X limits
            ax=ax,
            #title=f"{long_name}" if var_idx == 0 else None,
            xlabel="Time" if var_idx == n_vars - 1 else "",
            ylabel=ylabel_str,
            grid=True,
            show_legend=(var_idx == 0),
        )

    plt.tight_layout()
    #return fig, axes

def plot_collection_bivariate_scatter(
    collection: "EventCollection",
    var1: str,
    var2: str,
    site: Optional[str] = None,
    complementary_collection: Optional["EventCollection"] = None,
    var_to_units: Optional[Dict[str, str]] = None,
    var_to_longname: Optional[Dict[str, str]] = None,
    min_val: Optional[List[Optional[float]]] = None,
    max_val: Optional[List[Optional[float]]] = None,
    show_corr: bool = True,
    show_fit: bool = True,
    show_oneone: bool = False,
    figsize: Tuple[int, int] = (7, 6),
    **scatter_kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots bivariate scatter of var1 vs var2 extracted from an EventCollection 
    (or compares events vs. non-events if a complementary collection is provided).

    Parameters
    ----------
    collection : EventCollection
        Primary collection (e.g. detected events).
    var1 : str
        Variable for the X-axis.
    var2 : str
        Variable for the Y-axis.
    site : Optional[str]
        If provided, filters collection(s) to this site prior to extraction.
    complementary_collection : Optional[EventCollection]
        Optional second collection (e.g., non-events) to overlay on the scatter plot.
    var_to_units : Optional[Dict[str, str]]
        Mapping of variable names to units for axis labeling.
    var_to_longname : Optional[Dict[str, str]]
        Mapping of variable names to descriptive display names.
    min_val : Optional[List[Optional[float]]]
        Sequence of 2 elements: [x_min, y_min].
    max_val : Optional[List[Optional[float]]]
        Sequence of 2 elements: [x_max, y_max].
    show_corr : bool
        Whether to calculate and display correlation statistics.
    show_fit : bool
        Whether to compute and plot linear regression fit line.
    show_oneone : bool
        Whether to draw 1:1 reference line.
    figsize : Tuple[int, int]
        Dimensions (width, height) of the figure.
    **scatter_kwargs : Any
        Additional keyword arguments forwarded directly to format_scatter_plot.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
    """
    if min_val is not None and len(min_val) != 2:
        raise ValueError("min_val must be a sequence of exactly 2 elements: [var1_min, var2_min]")
    if max_val is not None and len(max_val) != 2:
        raise ValueError("max_val must be a sequence of exactly 2 elements: [var1_max, var2_max]")

    def _extract_pair_values(col: "EventCollection", target_site: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts aligned var1 and var2 numeric values across all events in a collection."""
        if target_site is not None:
            col = col.filter_by_site(target_site)

        x_list, y_list = [], []
        for period in col.events:
            if var1 in period.ds and var2 in period.ds:
                # Align on time coordinate within the xarray dataset slice
                da1, da2 = xr.align(period.ds[var1], period.ds[var2], join="inner")
                x_vals = da1.values.flatten()
                y_vals = da2.values.flatten()

                # Clean NaNs
                valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
                x_list.append(x_vals[valid])
                y_list.append(y_vals[valid])

        if not x_list:
            return np.array([]), np.array([])

        return np.concatenate(x_list), np.concatenate(y_list)

    # 1. Extract Primary Collection Data
    x_primary, y_primary = _extract_pair_values(collection, site)
    if len(x_primary) == 0:
        raise ValueError(f"No concurrent data found for '{var1}' and '{var2}' in the provided collection.")

    # 2. Metadata Labels
    def _format_label(v: str) -> str:
        name = (var_to_longname or {}).get(v, v)
        unit = (var_to_units or {}).get(v, None)
        return f"{name} [{unit}]" if unit else name

    xlabel = _format_label(var1)
    ylabel = _format_label(var2)
    site_str = f" Site: {site}" if site else ""
    title = f"Bivariate Scatter: {var1} vs {var2}{site_str}"

    xlim = (min_val[0] if min_val else None, max_val[0] if max_val else None)
    ylim = (min_val[1] if min_val else None, max_val[1] if max_val else None)

    fig, ax = plt.subplots(figsize=figsize)

    # 3. Render
    if complementary_collection is not None:
        # Extract complementary data (e.g., non-events)
        x_comp, y_comp = _extract_pair_values(complementary_collection, site)

        # Plot non-events / baseline first in background
        if len(x_comp) > 0:
            format_scatter_plot(
                x=x_comp,
                y=y_comp,
                ax=ax,
                color="gray",
                alpha=0.3,
                label="Non-Events",
                show_corr=False,
                show_fit=False,
                show_oneone=False,
                grid=False,
            )

        # Overlay primary collection (events)
        format_scatter_plot(
            x=x_primary,
            y=y_primary,
            ax=ax,
            color=scatter_kwargs.pop("color", "crimson"),
            label="Events",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            show_corr=show_corr,
            show_fit=show_fit,
            show_oneone=show_oneone,
            grid=True,
            **scatter_kwargs,
        )
    else:
        # Single collection scatter render
        format_scatter_plot(
            x=x_primary,
            y=y_primary,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            show_corr=show_corr,
            show_fit=show_fit,
            show_oneone=show_oneone,
            grid=True,
            **scatter_kwargs,
        )

    plt.tight_layout()
    
def print_integrated_flux_stats(
        collection: EventCollection,
        variable: str = "FluxMean2",
        site: Optional[str] = None,
    ) :
        """
        Prints a formatted summary of cumulative flux statistics for an EventCollection.
        """
        if site is not None:
            collection = collection.filter_by_site(site)

        stats = collection.get_integrated_flux_stats(variable=variable)
        if not stats:
            print(f"No valid integrated flux data available for variable '{variable}'.")
            return {}

        def _to_exp(val: float) -> str:
            if val == 0 or np.isnan(val):
                return "0"
            m, e = f"{val:.2e}".split("e")
            return f"{m} \u00d7 10^{int(e)}"

        mean_val = stats["mean_g_m2"]
        std_val = stats["std_g_m2"]
        sum_val = stats["sum_g_m2"]
        median_val = stats["median_g_m2"]
        min_val = stats["min_g_m2"]
        max_val = stats["max_g_m2"]

        site_str = f" [Site: {site}]" if site else ""
        header = f"  Cumulative Flux Summary ({variable}){site_str}  "
        divider = "=" * len(header)

        print(divider)
        print(header)
        print(divider)
        print(f" Events Analysed : {stats['count']}")
        print(f" Total Sum       : {_to_exp(sum_val)} g m\u207b\u00b2")
        print(f" Mean \u00b1 Std       : {_to_exp(mean_val)} \u00b1 {_to_exp(std_val)} g m\u207b\u00b2")
        print(f" Median          : {_to_exp(median_val)} g m\u207b\u00b2")
        print(f" Range           : [{_to_exp(min_val)}, {_to_exp(max_val)}] g m\u207b\u00b2")
        print(divider)

def print_duration_stats(
        collection: EventCollection,
        site: Optional[str] = None,
    ) :
        """
        Prints a formatted summary of event duration statistics for an EventCollection.
        """
        if site is not None:
            collection = collection.filter_by_site(site)

        stats = collection.get_duration_stats()
        if not stats:
            print("No event duration data available.")
            return {}

        site_str = f" [Site: {site}]" if site else ""
        header = f"  Event Duration Summary{site_str}  "
        divider = "=" * len(header)

        print(divider)
        print(header)
        print(divider)
        print(f" Events Analysed : {len(collection)}")
        print(f" Mean \u00b1 Std       : {stats['mean_hours']:.2f} \u00b1 {stats['std_hours']:.2f} hours")
        print(f" Median          : {stats['median_hours']:.2f} hours")
        print(f" Range           : [{stats['min_hours']:.2f}, {stats['max_hours']:.2f}] hours")
        print(divider)


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

def plot_diurnal_start_distribution(
    collection: "EventCollection",
    site: Optional[str] = None,
    color: str = "navy",
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a 24-hour bar distribution showing when events start throughout the day.
    """
    if site is not None:
        collection = collection.filter_by_site(site)

    if len(collection) == 0:
        raise ValueError("Collection is empty.")

    hourly_counts = collection.get_diurnal_start_distribution()

    fig, ax = plt.subplots(figsize=figsize)
    hours = hourly_counts.index.values
    counts = hourly_counts.values

    ax.bar(hours, counts, color=color, alpha=alpha, edgecolor="black", width=0.8)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Event Onset Count")

    site_str = f" - Site: {site}" if site else ""
    ax.set_title(f"Diurnal Event Onset Distribution (N={len(collection)}){site_str}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

def plot_starting_month_distribution(
    collection: "EventCollection",
    site: Optional[str] = None,
    color: str = "navy",
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a 12 months bar distribution showing when events start throughout the year.
    """
    if site is not None:
        collection = collection.filter_by_site(site)

    if len(collection) == 0:
        raise ValueError("Collection is empty.")

    monthly_counts = collection.get_starting_month_distribution()

    fig, ax = plt.subplots(figsize=figsize)
    months = monthly_counts.index.values
    counts = monthly_counts.values

    ax.bar(months, counts, color=color, alpha=alpha, edgecolor="black", width=0.8)

    ax.set_xticks(range(1, 13, 1))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel("Month")
    ax.set_ylabel("Event Frequency")

    site_str = f" - Site: {site}" if site else ""
    ax.set_title(f"Event Starting Month Distribution (N={len(collection)}){site_str}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
#