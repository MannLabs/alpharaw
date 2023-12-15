import pandas as pd
import numpy as np

import typing

import plotly.graph_objects as go

from alphatims.bruker import TimsTOF

def plot_scatter(
    x_values, y_values,
    color, marker_size,
    hovertext, 
    hovertemplate,
    name='',
):
    return go.Scatter(
        x=x_values,
        y=y_values,
        hovertext=hovertext,
        hovertemplate=hovertemplate,
        mode='markers',
        marker=dict(
            color=color,
            size=marker_size
        ),
        name=name,
        showlegend=False
    )

alphatims_labels = {
    'rt': "rt_values",
    'im': "mobility_values",
}

def plot_line_fast(
    tims_data:TimsTOF,
    tims_raw_indices: np.ndarray,
    tims_view_indices: np.array,
    name: str,
    legend_group:str,
    marker_color: str,
    view_dim:typing.Literal['rt','im']='rt', # or 'im'
    intensity_scale:float=1.0,
    rt_unit:str = "minute"
):
    """Plot an XIC as a lineplot.

    Parameters
    ----------
    tims_data : alphatims.bruker.TimsTOF
        An alphatims.bruker.TimsTOF data object.
    selected_indices : np.ndarray
        The raw indices of tims_data that are selected for this plot.
    label : str
        The label for the line plot.
    remove_zeros : bool
        If True, zeros are removed. Default: False.
    trim : bool
        If True, zeros on the left and right are trimmed. Default: True.

    Returns
    -------
    go.Figure
        the XIC line plot.
    """

    x_dimension = alphatims_labels[view_dim]

    intensities = tims_data.bin_intensities(tims_raw_indices, [x_dimension])
    if view_dim == 'rt': 
        x_ticks = tims_data.rt_values[tims_view_indices]
        intensities = intensities[tims_view_indices]
        if rt_unit == "minute":
            x_ticks /= 60.0
    else: 
        x_ticks = tims_data.mobility_values[tims_view_indices]
        intensities = intensities[tims_view_indices]

    return plot_line(
        x_ticks, intensities*intensity_scale,
        name=name, 
        marker_color=marker_color,
        legend_group=legend_group,
        x_text=view_dim.upper(),
    )

def plot_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    name: str,
    marker_color: str,
    legend_group:str,
    x_text: str = "RT (Sec)",
    hovertemplate: str='%{text} <br><b>Intensity:</b> %{y}',
):
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name=name,
        marker={"color":marker_color},
        legendgroup=legend_group,
        text=[f"{x_text}: {_x:.3f}" for _x in x_values],
        hovertemplate=hovertemplate,
    )

def plot_line_df(
    tims_sliced_df:pd.DataFrame,
    view_indices_df:pd.DataFrame,
    label: str,
    legend_group:str,
    marker_color: str,
    view_dim: str='rt' # or 'im'
):
    """Plot an XIC as a lineplot.

    Parameters
    ----------
    tims_sliced_df : pd.DataFrame
        TimsTOF[...] df
    label : str
        The label for the line plot.
    trim : bool
        If True, zeros on the left and right are trimmed. Default: True.

    Returns
    -------
    go.Figure
        the XIC line plot.
    """

    if view_dim == 'rt':
        tims_sliced_df = tims_sliced_df.groupby(
            'frame_indices', as_index=False
        ).agg(
            {
                'rt_values':'mean',
                'intensity_values':'sum',
            }
        )
        tims_sliced_df['rt_values'] /= 60
        tims_sliced_df.sort_values('rt_values', inplace=True)
        tims_sliced_df = view_indices_df.merge(
            tims_sliced_df, on=['frame_indices','rt_values'], how='left'
        )
        tims_sliced_df.loc[tims_sliced_df.intensity_values.isna(),'intensity_values'] = 0
        x_ticks = tims_sliced_df.rt_values.values
    else: 
        tims_sliced_df = tims_sliced_df.groupby(
            'scan_indices', as_index=False
        ).agg(
            {
                'mobility_values':'mean',
                'intensity_values':'sum',
            }
        )
        tims_sliced_df.sort_values('mobility_values', inplace=True)
        tims_sliced_df = view_indices_df.merge(
            tims_sliced_df, on=['scan_indices','mobility_values'], how='left'
        )
        tims_sliced_df.loc[tims_sliced_df.intensity_values.isna(),'intensity_values'] = 0
        x_ticks = tims_sliced_df.mobility_values.values

    trace = go.Scatter(
        x=x_ticks,
        y=tims_sliced_df.intensity_values.values,
        mode='lines',
        text=[
            f'RT: {_x*60:.3f}s' for _x in x_ticks] if view_dim == 'rt' 
            else [f'IM: {_x:.3f}' for _x in x_ticks],
        hovertemplate='%{text}<br>Intensity: %{y}',
        name=label,
        marker={"color":marker_color},
        legendgroup=legend_group,
    )
    return trace