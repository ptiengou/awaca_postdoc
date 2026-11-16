from imports import *


def create_antarctic_axes(
    *,
    extent: tuple[float, float, float, float] = (-180.0, 180.0, -90.0, -60.0),
    figsize: tuple[float, float] = (9, 9),
) -> tuple[Figure, GeoAxes]:
    """Create a south-polar map with land, coastlines, and graticules."""
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.SouthPolarStereo()},
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.coastlines(resolution="110m", color="black", linewidth=0.8, zorder=2)
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        color="gray",
        linestyle=":",
        linewidth=0.5,
        alpha=0.6,
    )
    return fig, ax


def plot_antarctic_map(
    dataset: xr.Dataset,
    variable: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    time_index: int = 0,
    level_index: int = 0,
    time_dim: str = "time_counter",
    level_dim: str = "presnivs",
    cmap: str = "viridis",
    extent: tuple[float, float, float, float] = (-180.0, 180.0, -90.0, -60.0),
    ax: GeoAxes | None = None,
) -> tuple[Figure, GeoAxes, QuadMesh]:
    """Plot a time-mean simulation field on a south-polar map.

    If the field has a time dimension, it is averaged over that dimension;
    this also works when it contains only one timestep. The vertical level is
    selected with ``level_index``, and the first value is used for other
    extra dimensions.
    """
    if variable not in dataset.data_vars:
        raise KeyError(f"{variable!r} is not a data variable in the dataset")
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be less than vmax")

    field = dataset[variable]
    has_time_dimension = time_dim in field.dims
    if has_time_dimension:
        field = field.mean(dim=time_dim, skipna=True, keep_attrs=True)

    if "lat" not in field.coords or "lon" not in field.coords:
        raise ValueError(f"{variable!r} must have 'lat' and 'lon' coordinates")
    if field["lat"].ndim != 1 or field["lon"].ndim != 1:
        raise ValueError("This map helper requires one-dimensional lat/lon coordinates")

    indexers = {
        dim: level_index if dim == level_dim else 0
        for dim in field.dims
        if dim not in {"lat", "lon"}
    }
    if indexers:
        field = field.isel(indexers)
    if set(field.dims) != {"lat", "lon"}:
        raise ValueError(
            f"{variable!r} must reduce to lat/lon dimensions; got {field.dims}"
        )
    field = field.transpose("lat", "lon")

    if ax is None:
        fig, ax = create_antarctic_axes(extent=extent)
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        field["lon"],
        field["lat"],
        field,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    title = field.attrs.get("long_name", variable)
    if has_time_dimension:
        title = f"{title} - temporal mean"
    ax.set_title(title)

    units = field.attrs.get("units")
    colorbar_label = f"{variable} ({units})" if units else variable
    fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.05, label=colorbar_label)
    return fig, ax, mesh


##############
# TIME PLOTS #
##############

def _get_time_field(
    dataset: xr.Dataset,
    variable: str,
    *,
    time_dim: str,
    level_dim: str,
    level_index: int,
) -> xr.DataArray:
    if variable not in dataset.data_vars:
        raise KeyError(f"{variable!r} is not a data variable in the dataset")

    field = dataset[variable]
    required_dims = {time_dim, "lat", "lon"}
    if not required_dims.issubset(field.dims):
        raise ValueError(
            f"{variable!r} must have dimensions {required_dims}; got {field.dims}"
        )

    indexers = {
        dim: level_index if dim == level_dim else 0
        for dim in field.dims
        if dim not in required_dims
    }
    if indexers:
        field = field.isel(indexers)
    if set(field.dims) != required_dims:
        raise ValueError(
            f"{variable!r} must reduce to {time_dim}, lat, and lon dimensions; "
            f"got {field.dims}"
        )
    return field.transpose(time_dim, "lat", "lon")


def _domain_mean(
    field: xr.DataArray,
    *,
    mask: xr.DataArray | None,
) -> xr.DataArray:
    weights = np.cos(np.deg2rad(field["lat"])).clip(min=0)
    if mask is not None:
        if not isinstance(mask, xr.DataArray):
            raise TypeError("mask must be an xarray.DataArray with boolean values")
        if not mask.dims or not set(mask.dims).issubset({"lat", "lon"}):
            raise ValueError("mask dimensions must be a non-empty subset of lat/lon")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError("mask must contain boolean values")
        field, mask = xr.align(field, mask, join="exact", copy=False)
        mask = mask.fillna(False)
        field = field.where(mask)
        weights = weights.where(mask, 0)

    spatial_dims = ("lat", "lon")
    weighted_sum = (field * weights).sum(dim=spatial_dims, skipna=True)
    valid_weights = weights.broadcast_like(field).where(field.notnull(), 0)
    weight_sum = valid_weights.sum(dim=spatial_dims, skipna=True)
    mean = weighted_sum / weight_sum
    mean.attrs = field.attrs.copy()
    return mean


def _nearest_gridcell(
    field: xr.DataArray,
    *,
    lon: float,
    lat: float,
) -> tuple[xr.DataArray, float, float]:
    if field["lat"].ndim != 1 or field["lon"].ndim != 1:
        raise ValueError("Grid-cell selection requires one-dimensional lat/lon")
    lat_index = int(abs(field["lat"] - lat).argmin().item())
    lon_difference = abs((field["lon"] - lon + 180) % 360 - 180)
    lon_index = int(lon_difference.argmin().item())
    selected = field.isel(lat=lat_index, lon=lon_index)
    return (
        selected,
        float(field["lon"].isel(lon=lon_index).item()),
        float(field["lat"].isel(lat=lat_index).item()),
    )


def _plot_time_series(
    series: xr.DataArray,
    *,
    variable: str,
    title: str,
    time_dim: str,
    ax: Axes | None,
    plot_kwargs: dict,
    ymin=None,
    ymax=None,
) -> tuple[Figure, Axes, Line2D]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    (line,) = ax.plot(
        series[time_dim].values,
        series.values,
        label=variable,
        **plot_kwargs,
    )
    units = series.attrs.get("units")
    ax.set_ylabel(f"{variable} ({units})" if units else variable)
    ax.set_xlabel("Time")
    ax.set_title(title)
    # Conditionally set y-axis limits if they are not None
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax, line


def _plot_diurnal_cycle(
    series: xr.DataArray,
    *,
    variable: str,
    title: str,
    ax: Axes | None,
    plot_kwargs: dict,
    ymin=None,
    ymax=None,
) -> tuple[Figure, Axes, Line2D]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    (line,) = ax.plot(series["hour"].values, series.values, label=variable, **plot_kwargs)
    units = series.attrs.get("units")
    ax.set_ylabel(f"{variable} ({units})" if units else variable)
    ax.set_xlabel("Hour of day")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_title(title)
    # Conditionally set y-axis limits if they are not None
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax, line


def plot_domain_mean_timeseries(
    dataset: xr.Dataset,
    variable: str,
    *,
    mask: xr.DataArray | None = None,
    time_dim: str = "time_counter",
    level_dim: str = "presnivs",
    level_index: int = 0,
    ax: Axes | None = None,
    ymin=None,
    ymax=None,
    **plot_kwargs,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the latitude-weighted domain mean, optionally restricted by a mask."""
    field = _get_time_field(
        dataset,
        variable,
        time_dim=time_dim,
        level_dim=level_dim,
        level_index=level_index,
    )
    series = _domain_mean(field, mask=mask)
    area = "masked domain" if mask is not None else "full domain"
    return _plot_time_series(
        series,
        variable=variable,
        title=f"{variable} - {area} mean",
        time_dim=time_dim,
        ax=ax,
        ymin=ymin,
        ymax=ymax,
        plot_kwargs=plot_kwargs,
    )


def plot_gridcell_timeseries(
    dataset: xr.Dataset,
    variable: str,
    *,
    lon: float,
    lat: float,
    time_dim: str = "time_counter",
    level_dim: str = "presnivs",
    level_index: int = 0,
    ax: Axes | None = None,
    title=None,
    ymin=None,
    ymax=None,
    **plot_kwargs,
) -> tuple[Figure, Axes, Line2D]:
  """Plot the time series at the grid cell nearest the requested lon/lat."""
  field = _get_time_field(
      dataset,
      variable,
      time_dim=time_dim,
      level_dim=level_dim,
      level_index=level_index,
  )
  series, grid_lon, grid_lat = _nearest_gridcell(field, lon=lon, lat=lat)

  if title is None:
    title = f"{variable} at nearest grid cell ({grid_lon:.2f}°, {grid_lat:.2f}°)"

  return _plot_time_series(
      series,
      variable=variable,
      title=title,
      time_dim=time_dim,
      ax=ax,
      ymin=ymin,
      ymax=ymax,
      plot_kwargs=plot_kwargs,
  )


def plot_domain_mean_diurnal_cycle(
    dataset: xr.Dataset,
    variable: str,
    *,
    mask: xr.DataArray | None = None,
    time_dim: str = "time_counter",
    level_dim: str = "presnivs",
    level_index: int = 0,
    ax: Axes | None = None,
    ymin=None,
    ymax=None,
    **plot_kwargs,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the mean diurnal cycle of the latitude-weighted domain mean."""
    field = _get_time_field(
        dataset,
        variable,
        time_dim=time_dim,
        level_dim=level_dim,
        level_index=level_index,
    )
    series = _domain_mean(field, mask=mask)
    cycle = series.groupby(f"{time_dim}.hour").mean(
        dim=time_dim,
        skipna=True,
        keep_attrs=True,
    ).reindex(hour=np.arange(24))
    area = "masked domain" if mask is not None else "full domain"
    return _plot_diurnal_cycle(
        cycle,
        variable=variable,
        title=f"{variable} - mean diurnal cycle ({area})",
        ax=ax,
        ymin=ymin,
        ymax=ymax,
        plot_kwargs=plot_kwargs,
    )


def plot_gridcell_mean_diurnal_cycle(
    dataset: xr.Dataset,
    variable: str,
    *,
    lon: float,
    lat: float,
    time_dim: str = "time_counter",
    level_dim: str = "presnivs",
    level_index: int = 0,
    ax: Axes | None = None,
    title=None,
    ymin=None,
    ymax=None,
    **plot_kwargs,
) -> tuple[Figure, Axes, Line2D]:
    """Plot the mean diurnal cycle at the nearest grid cell to lon/lat."""
    field = _get_time_field(
        dataset,
        variable,
        time_dim=time_dim,
        level_dim=level_dim,
        level_index=level_index,
    )
    series, grid_lon, grid_lat = _nearest_gridcell(field, lon=lon, lat=lat)
    cycle = series.groupby(f"{time_dim}.hour").mean(
        dim=time_dim,
        skipna=True,
        keep_attrs=True,
    ).reindex(hour=np.arange(24))
    if title is None:
        title=(
            f"{variable} - mean diurnal cycle at "
            f"({grid_lon:.2f}°, {grid_lat:.2f}°)"
        )
    return _plot_diurnal_cycle(
        cycle,
        variable=variable,
        title=title,
        ax=ax,
        ymin=ymin,
        ymax=ymax,
        plot_kwargs=plot_kwargs
    )
