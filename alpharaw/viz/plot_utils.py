import typing

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from alphatims.bruker import TimsTOF


def plot_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    marker_size: int,
    hovertext: str,
    hovertemplate: str,
    name: str = "",
) -> go.Scatter:
    """
    plotly scatter based on x and y points.

    Parameters
    ----------
    x_values : np.ndarray
        x values for :class:`go.Scatter`
    y_values : np.ndarray
        y values for :class:`go.Scatter`
    color : str
        color name
    marker_size : int
        marker size of the point (x,y)
    hovertext : str
        `hovertext` parameter of :class:`go.Scatter`
    hovertemplate : str
        `hovertemplate` parameter of :class:`go.Scatter`
    name : str, optional
        `name` parameter of :class:`go.Scatter`, by default ''

    Returns
    -------
    go.Scatter
        go.Scatter of plotly
    """
    return go.Scatter(
        x=x_values,
        y=y_values,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        mode="markers",
        marker=dict(color=color, size=marker_size),
        name=name,
        showlegend=False,
    )


alphatims_labels = {
    "rt": "rt_values",
    "im": "mobility_values",
}


def plot_line_tims(
    tims_data: TimsTOF,
    tims_raw_indices: np.ndarray,
    tims_view_indices: np.array,
    name: str,
    legend_group: str,
    marker_color: str,
    view_dim: typing.Literal["rt", "im"] = "rt",  # or 'im'
    intensity_scale: float = 1.0,
    rt_unit: str = "minute",
) -> go.Figure:
    """
    Plot an XIC line on alphatims `TimsTOF` data

    Parameters
    ----------
    tims_data : TimsTOF
        The alphatims `TimsTOF` object
    tims_raw_indices : np.ndarray
        Raw indices on `TimsTOF` object
    tims_view_indices : np.array
        View indices on `TimsTOF` object
    name : str
        Display name
    legend_group : str
        Lines will be grouped by `legend_group`
    marker_color : str
        Color of the scatter (x, y)
    view_dim : "rt", "im", optional
        View dimension, "rt" or "im", by default "rt"
    rt_unit : str, optional
        RT unit, by default "minute"

    Returns
    -------
    go.Figure
        Line plot
    """
    x_dimension = alphatims_labels[view_dim]

    intensities = tims_data.bin_intensities(tims_raw_indices, [x_dimension])
    if view_dim == "rt":
        x_ticks = tims_data.rt_values[tims_view_indices]
        intensities = intensities[tims_view_indices]
        if rt_unit == "minute":
            x_ticks /= 60.0
    else:
        x_ticks = tims_data.mobility_values[tims_view_indices]
        intensities = intensities[tims_view_indices]

    return plot_line(
        x_ticks,
        intensities * intensity_scale,
        name=name,
        marker_color=marker_color,
        legend_group=legend_group,
        x_text=view_dim.upper(),
    )


def plot_line_tims_fast(
    tims_data: TimsTOF,
    tims_raw_indices: np.ndarray,
    tims_view_indices: np.array,
    name: str,
    legend_group: str,
    marker_color: str,
    view_dim: typing.Literal["rt", "im"] = "rt",
    intensity_scale: float = 1.0,
    rt_unit: str = "minute",
    add_peak_area=True,
) -> go.Figure:
    """
    Plot an XIC line on alphatims `TimsTOF` data

    Parameters
    ----------
    tims_data : TimsTOF
        The alphatims `TimsTOF` object
    tims_raw_indices : np.ndarray
        Raw indices on `TimsTOF` object
    tims_view_indices : np.array
        View indices on `TimsTOF` object
    name : str
        Display name
    legend_group : str
        Lines will be grouped by `legend_group`
    marker_color : str
        Color of the scatter (x, y)
    view_dim : "rt", "im", optional
        View dimension, "rt" or "im", by default "rt"
    intensity_scale : float, optional
        Intensity scale of mirror plot, by default 1.0
    rt_unit : str, optional
        RT unit, by default "minute"
    add_peak_area : bool, optional
        If add peak area in the hover text, by default True

    Returns
    -------
    go.Figure
        _description_
    """
    x_dimension = alphatims_labels[view_dim]

    intensities = tims_data.bin_intensities(tims_raw_indices, [x_dimension])
    if view_dim == "rt":
        x_ticks = tims_data.rt_values[tims_view_indices]
        intensities = intensities[tims_view_indices]
        if rt_unit == "minute":
            x_ticks /= 60.0
    else:
        x_ticks = tims_data.mobility_values[tims_view_indices]
        intensities = intensities[tims_view_indices]

    if add_peak_area:
        peak_area = abs(np.trapz(y=intensities, x=x_ticks))

    return plot_line(
        x_ticks,
        intensities * intensity_scale,
        name=name,
        marker_color=marker_color,
        legend_group=legend_group,
        x_text=view_dim.upper(),
        other_info=f"Peak area: {peak_area:.2e}" if add_peak_area else "",
    )


def plot_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    name: str,
    marker_color: str,
    legend_group: str,
    x_text: str = "RT",
    other_info: str = "",
    hovertemplate: str = "%{text} <br>Intensity: %{y}",
) -> go.Scatter:
    """
    Plot a line for given x and y points, this is usually for XIC plots.

    Parameters
    ----------
    x_values : np.ndarray
        x values for :class:`go.Scatter`
    y_values : np.ndarray
        y values for :class:`go.Scatter`
    name : str
        Legend name for the plotted line
    marker_color : str
        Marker color of the point (x,y)
    legend_group : str
        Different lines can be grouped by the same `legend_group`
    x_text : str, optional
        Hover text of the x axis, by default "RT"
    other_info: str, optional
        Other hover information, by default ""
    hovertemplate : str, optional
        `hovertemplate` parameter of :class:`go.Scatter`,
        by default '%{text} <br><b>Intensity:</b> %{y}'

    Returns
    -------
    go.Scatter
        The plotted line
    """
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines",
        name=name,
        marker={"color": marker_color},
        legendgroup=legend_group,
        text=[
            f"{x_text}: {_x:.3f}" + (f"<br>{other_info}" if other_info else "")
            for _x in x_values
        ],
        hovertemplate=hovertemplate,
    )


def plot_line_df(
    tims_sliced_df: pd.DataFrame,
    view_indices_df: pd.DataFrame,
    label: str,
    legend_group: str,
    marker_color: str,
    view_dim: str = "rt",  # or 'im'
) -> go.Figure:
    """
    TODO deprecated
    """
    if view_dim == "rt":
        tims_sliced_df = tims_sliced_df.groupby("frame_indices", as_index=False).agg(
            {
                "rt_values": "mean",
                "intensity_values": "sum",
            }
        )
        tims_sliced_df["rt_values"] /= 60
        tims_sliced_df.sort_values("rt_values", inplace=True)
        tims_sliced_df = view_indices_df.merge(
            tims_sliced_df, on=["frame_indices", "rt_values"], how="left"
        )
        tims_sliced_df.loc[
            tims_sliced_df.intensity_values.isna(), "intensity_values"
        ] = 0
        x_ticks = tims_sliced_df.rt_values.values
    else:
        tims_sliced_df = tims_sliced_df.groupby("scan_indices", as_index=False).agg(
            {
                "mobility_values": "mean",
                "intensity_values": "sum",
            }
        )
        tims_sliced_df.sort_values("mobility_values", inplace=True)
        tims_sliced_df = view_indices_df.merge(
            tims_sliced_df, on=["scan_indices", "mobility_values"], how="left"
        )
        tims_sliced_df.loc[
            tims_sliced_df.intensity_values.isna(), "intensity_values"
        ] = 0
        x_ticks = tims_sliced_df.mobility_values.values

    trace = go.Scatter(
        x=x_ticks,
        y=tims_sliced_df.intensity_values.values,
        mode="lines",
        text=[f"RT: {_x*60:.3f}s" for _x in x_ticks]
        if view_dim == "rt"
        else [f"IM: {_x:.3f}" for _x in x_ticks],
        hovertemplate="%{text}<br>Intensity: %{y}",
        name=label,
        marker={"color": marker_color},
        legendgroup=legend_group,
    )
    return trace
