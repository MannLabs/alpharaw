import typing

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from alphatims.bruker import (
    TimsTOF, 
)

from .plot_utils import (
    plot_line_fast
)

class XIC_Plot():
    # hovermode = "x" | "y" | "closest" | False | "x unified" | "y unified"
    hovermode = 'closest'
    plot_height = 550
    colorscale_qualitative="Alphabet"
    colorscale_sequential="Viridis"
    theme_template='plotly_white'
    ms1_ppm = 20.0
    ms2_ppm = 20.0
    rt_sec_win = 30.0
    plot_rt_unit:str = "minute"
    im_win = 0.05
    fig:go.Figure = None

    "list of XIC_Trace objects"
    traces:list = []

    def plot(self, 
        tims_data:TimsTOF,
        query_df:pd.DataFrame,
        view_dim:typing.Literal["rt","im"]="rt",
        title:str="",
    ):
        rt_sec = query_df['rt_sec'].values[0]
        if "im" not in query_df.columns:
            im = 0.0
        else:
            im = query_df["im"].values[0]
        if "precursor_mz" not in query_df.columns:
            precursor_mz = 0.0
        else:
            precursor_mz = query_df.precursor_mz.values[0]
        query_masses = query_df.mz.values
        if "intensity" in query_df.columns:
            query_intensities = query_df.intensity.values
        else:
            query_intensities = None
        ion_names = query_df.ion_name.values

        if "color" not in query_df.columns:
            marker_colors = None
        else:
            marker_colors = query_df.color.values

        return self.plot_query_masses(
            tims_data=tims_data,
            query_masses=query_masses,
            query_ion_names=ion_names,
            query_rt_sec=rt_sec,
            query_im=im,
            precursor_mz=precursor_mz,
            marker_colors=marker_colors,
            view_dim=view_dim,
            query_intensities=query_intensities,
            title=title,
        )

    def _init_plot(self, rows=1, view_dim='rt'):
        self.fig = make_subplots(
            rows=rows, 
            cols=1,
            shared_xaxes=True,
            x_title=f'RT ({self.plot_rt_unit})' if view_dim == 'rt' else 'Mobility',
            y_title='intensity',
            vertical_spacing=0.2/rows,
        )
        self.traces = [
            XIC_Trace(fig=self.fig, row=i+1) 
            for i in range(rows)
        ]

    def plot_query_masses(self,
        tims_data:TimsTOF,
        query_masses:np.ndarray,
        query_ion_names:typing.List[str],
        query_rt_sec:float, 
        query_im:float,
        precursor_mz:float,
        marker_colors:list = None,
        view_dim:typing.Literal["rt","im"]="rt",
        query_intensities:np.ndarray = None,
        title="",
    ):
        self._init_plot(rows=1, view_dim=view_dim)
        mass_tols = query_masses*1e-6*(
            self.ms1_ppm if precursor_mz == 0 else self.ms2_ppm
        )
        if marker_colors is None:
            marker_colors = self._get_color_set(len(query_masses))
        self.traces[0].add_traces(
            tims_data=tims_data,
            query_masses=query_masses,
            mass_tols=mass_tols,
            ion_names=query_ion_names,
            marker_colors=marker_colors,
            query_rt_sec=query_rt_sec,
            query_im=query_im,
            precursor_mz=precursor_mz,
            precursor_mz_tol=precursor_mz*1e-6*self.ms1_ppm,
            view_dim=view_dim,
            rt_sec_win=self.rt_sec_win,
            im_win=self.im_win,
            query_intensities=query_intensities,
        )
        self.fig.update_layout(
            template=self.theme_template,
            title=dict(
                text=title,
                yanchor='bottom'
            ),
            # width=width,
            height=self.plot_height,
            hovermode=self.hovermode,
            showlegend=True,
        )
        return self.fig

    def _get_color_set(self, n_query):
        if n_query <= len(
            getattr(px.colors.qualitative, self.colorscale_qualitative)
        ):
            color_set = getattr(
                px.colors.qualitative, self.colorscale_qualitative
            )
        else:
            color_set = px.colors.sample_colorscale(
                self.colorscale_sequential, 
                samplepoints=n_query
            )
        return color_set


class XIC_Trace():
    label_format = '{ion_name} {mz:.3f}'
    legend_group = '{ion_name}' # {ion_name}, {mz} or None
    fig:go.Figure
    row:int = 1
    col:int = 1
    plot_rt_unit:str = "minute"

    def __init__(self, 
        fig:go.Figure, 
        row:int=1, col:int=1,
        plot_rt_unit:str = "minute",
    ):
        self.fig = fig
        self.row = row
        self.col = col
        self.plot_rt_unit = plot_rt_unit

    def add_traces(self, 
        tims_data:TimsTOF,
        query_masses:np.ndarray,
        mass_tols:np.ndarray,
        ion_names:typing.List[str],
        marker_colors:typing.List,
        query_rt_sec:float, 
        query_im:float,
        precursor_mz:float = 0.0,
        precursor_mz_tol:float = 0.02,
        view_dim:typing.Literal["rt","im"]="rt",
        rt_sec_win = 30.0,
        im_win = 0.05,
        query_intensities:np.ndarray = None,
    )->go.Figure:
        """Add traces for the query_masses.

        Args:
            tims_data (TimsTOF): AlphaTims TimsTOF object.
            query_masses (np.ndarray): Query masses.
            ion_names (typing.List[str]): Ion names for query_masses.
            marker_colors (typing.List): Colors for each query mass.
            query_rt_sec (float): Query RT in seconds.
            query_im (float): Query ion mobility.
            precursor_mz (float, optional): Precursor mz, 0 means it is MS1. Defaults to 0.0.
            query_intensities (np.ndarray, optional): Query intensities. Defaults to None.

        Returns:
            go.Figure: self.fig.
        """
        (
            rt_slice, im_slice, prec_mz_slice, view_indices
        ) = get_plotting_slices(
            tims_data=tims_data, 
            rt_sec=query_rt_sec, 
            rt_sec_win=rt_sec_win,
            im=query_im, im_win=im_win,
            precursor_mz=precursor_mz, 
            precursor_mz_tol=precursor_mz_tol,
            view_dim=view_dim,
        )

        if query_intensities is None:
            query_intensities = np.zeros_like(query_masses)
        else:
            query_intensities /= query_intensities.max()

        for (
            ion_name, query_mass, query_inten,
            marker_color, mass_tol
        ) in zip(
            ion_names, query_masses, query_intensities,
            marker_colors, mass_tols
        ):
            self._add_one_trace(
                tims_data=tims_data,
                query_mass=query_mass, 
                mass_tol=mass_tol, 
                rt_slice=rt_slice, im_slice=im_slice,
                prec_mz_slice=prec_mz_slice, 
                view_indices=view_indices,
                view_dim=view_dim,
                label=self.label_format.format(ion_name=ion_name, mz=query_mass),
                legend_group=self.legend_group.format(ion_name=ion_name),
                marker_color=marker_color,
            )
            if query_inten > 0:
                self._add_one_trace(
                    tims_data=tims_data,
                    query_mass=query_mass, 
                    mass_tol=mass_tol, 
                    rt_slice=rt_slice, im_slice=im_slice,
                    prec_mz_slice=prec_mz_slice, 
                    view_indices=view_indices,
                    view_dim=view_dim,
                    label=self.label_format.format(ion_name=ion_name, mz=query_mass),
                    legend_group=self.legend_group.format(ion_name=ion_name),
                    marker_color=marker_color,
                    intensity_scale=-query_inten,
                )

    def _add_one_trace(self,
        tims_data:TimsTOF,
        query_mass:float,
        mass_tol:float,
        rt_slice:slice, im_slice:slice, 
        prec_mz_slice:slice,
        view_indices:np.ndarray,
        view_dim:str,
        label:str, 
        legend_group:str, 
        marker_color:str,
        intensity_scale:float=1.0,
    ):
        frag_indices = tims_data[
            rt_slice,
            im_slice,
            prec_mz_slice,
            slice(
                query_mass - mass_tol, 
                query_mass + mass_tol,
            ),
            'raw'
        ]
        if len(frag_indices) == 0: return self.fig
        self.fig.add_trace(
            plot_line_fast(
                tims_data, 
                frag_indices,
                view_indices,
                name=label,
                legend_group=legend_group,
                marker_color=marker_color, 
                view_dim=view_dim,
                intensity_scale=intensity_scale
            ),
            row=self.row, col=self.col,
        )

def get_plotting_slices(
    tims_data: TimsTOF,
    rt_sec:float, rt_sec_win:float=30.0,
    im:float = 0., im_win:float = 0.05,
    precursor_mz:float = 0.0,
    precursor_mz_tol:float = 0.02,
    view_dim:str = "rt",
):
    """
    Get plotting slices for target queries in TimsTOF data.
    Args:
        tims_data (TimsTOF): AlphaTims TimsTOF object.
        rt_sec (float): Query RT in seconds.
        rt_sec_win (float, optional): Query RT window in seconds. Defaults to 30.0.
        im (float, optional): Query ion mobility, 0 means no mobility dimension. Defaults to 0..
        im_win (float, optional): Ion mobility window. Defaults to 0.05.
        precursor_mz (float, optional): Precursor mz, 0 means it is MS1. Defaults to 0.0.
        ppm (float, optional): PPM tolerance for `precursor_mz`. Defaults to 20.0.
        view_dim (str, optional): View dimension, "rt" or "im". Defaults to "rt"
    """
    rt_slice = slice(
        rt_sec - rt_sec_win/2, 
        rt_sec + rt_sec_win/2
    )
    im_slice = slice(
        im - im_win/2, 
        im + im_win/2
    )

    if precursor_mz == 0:
        prec_mz_slice = 0
        raw_indices = tims_data[
            rt_slice,
            im_slice,
            0,
            :,
            'raw'
        ]
    else:
        prec_mz_slice = slice(
            precursor_mz - precursor_mz_tol,
            precursor_mz + precursor_mz_tol,
        )
        raw_indices = tims_data[
            rt_slice,
            im_slice,
            prec_mz_slice,
            :,
            'raw'
        ]

    if view_dim == 'rt':
        view_indices = np.sort(np.array(list(set(
            tims_data.convert_from_indices(
            raw_indices,
            return_frame_indices=True
        )['frame_indices'])), dtype=np.int64))
    else:
        view_indices = np.sort(np.array(list(set(
            tims_data.convert_from_indices(
            raw_indices,
            return_scan_indices=True
        )['scan_indices'])), dtype=np.int64))

    return rt_slice, im_slice, prec_mz_slice, view_indices