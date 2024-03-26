import typing

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .plot_utils import (
    plot_line
)

from alphabase.constants.atom import MASS_ISOTOPE
from alpharaw.match.spec_finder import (
    find_spec_idxes,
    find_multinotch_spec_idxes,
)
from alpharaw.match.match_utils import match_batch_spec

class XIC_Plot():
    # hovermode = "x" | "y" | "closest" | False | "x unified" | "y unified"
    hovermode = 'closest'
    plot_height = 550
    colorscale_qualitative="Alphabet"
    colorscale_sequential="Viridis"
    theme_template='plotly_white'
    ms1_ppm = 20.0
    ms2_ppm = 20.0
    rt_sec_win = 60.0
    plot_rt_unit:str = "minute"
    fig:go.Figure = None
    isotope_cum_abundance=0.9,

    # list of XIC_Trace objects
    traces:list = []

    """
    Only apply for 3D-MS, i.e. without mobility
    """

    def plot(self, 
        spectrum_df:pd.DataFrame,
        peak_df:pd.DataFrame,
        query_df:pd.DataFrame,
        title:str="",
        add_peak_area=False,
        create_new_fig=True,
        plot_rows = 1,
        ith_plot_row = 0,
    ):
        if "rt_sec" in query_df.columns:
            rt_sec = query_df.rt_sec.values[0]
        else:
            rt_sec = query_df.rt.values[0]*60
        
        (
            precursor_left_mz, precursor_right_mz
        ) = self._get_precursor_mz_range(query_df)
        
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

        if add_peak_area:
            self.get_peak_area(
                spectrum_df, peak_df, query_df
            )

        return self.plot_query_masses(
            spectrum_df=spectrum_df,
            peak_df=peak_df,
            query_masses=query_masses,
            query_ion_names=ion_names,
            query_rt_sec=rt_sec,
            precursor_left_mz=precursor_left_mz,
            precursor_right_mz=precursor_right_mz,
            marker_colors=marker_colors,
            query_intensities=query_intensities,
            title=title,
            create_new_fig = create_new_fig,
            plot_rows = plot_rows,
            ith_plot_row = ith_plot_row,
        )
    
    def _get_precursor_mz_range(
        self, query_df,
    ):
        if "precursor_mz" not in query_df.columns:
            precursor_left_mz = -1.0
            precursor_right_mz = -1.0
        else:
            precursor_left_mz = query_df.precursor_mz.values[0]*(1-self.ms1_ppm*1e-6)
            precursor_right_mz = query_df.precursor_mz.values[0]*(1+self.ms1_ppm*1e-6)
        
            for iso in range(10):
                if f"precursor_i_{iso}" not in query_df.columns:
                    break
            if iso > 0:
                mono_idx = query_df.precursor_mono_idx.values[0]
                charge = query_df.precursor_charge.values[0]
                precursor_mz = query_df.precursor_mz.values[0]
                precursor_left_mz = precursor_mz-mono_idx*MASS_ISOTOPE/charge
                isotope_cumsum = np.cumsum([
                    query_df[f"precursor_i_{i}"].values[0] for i in range(iso)
                ])
                for i in range(iso-1,-1,-1):
                    if isotope_cumsum[i] >= self.isotope_cum_abundance:
                        precursor_right_mz = precursor_mz+(i-mono_idx)*MASS_ISOTOPE/charge
                        break
        return precursor_left_mz, precursor_right_mz
    
    def get_peak_area(self, 
        spectrum_df:pd.DataFrame,
        peak_df:pd.DataFrame,
        query_df:pd.DataFrame,
        query_start_rt_sec:float=None,
        query_stop_rt_sec:float=None,
        precursor_left_mz:float=None,
        precursor_right_mz:float=None,
    ):
        if query_start_rt_sec is None:
            query_start_rt_sec = query_df.rt_sec.values[0] - self.rt_sec_win/2
            query_stop_rt_sec = query_df.rt_sec.values[0] + self.rt_sec_win/2
        if precursor_left_mz is None:
            (
                precursor_left_mz, precursor_right_mz
            ) = self._get_precursor_mz_range(query_df)

        mass_tols = query_df.mz.values*1e-6*(
            self.ms1_ppm if precursor_left_mz<=0 else self.ms2_ppm
        )

        query_df["peak_area"] = get_peak_area(
            spectrum_df=spectrum_df,
            peak_df=peak_df,
            query_masses=query_df.mz.values,
            mass_tols=mass_tols,
            query_start_rt_sec=query_start_rt_sec,
            query_stop_rt_sec=query_stop_rt_sec,
            precursor_left_mz=precursor_left_mz,
            precursor_right_mz=precursor_right_mz,
        )
        return query_df

    def plot_query_masses(self,
        spectrum_df:pd.DataFrame,
        peak_df:pd.DataFrame,
        query_masses:np.ndarray,
        query_ion_names:typing.List[str],
        query_rt_sec:float, 
        precursor_left_mz:float,
        precursor_right_mz:float,
        marker_colors:list = None,
        query_intensities:np.ndarray = None,
        title="",
        create_new_fig = True,
        plot_rows = 1,
        ith_plot_row = 0,
    ):
        if create_new_fig:
            self._init_plot(rows=plot_rows)
        mass_tols = query_masses*1e-6*(
            self.ms1_ppm if precursor_left_mz<0 else self.ms2_ppm
        )
        if marker_colors is None:
            marker_colors = self._get_color_set(len(query_masses))
        self.traces[ith_plot_row].add_traces(
            spectrum_df=spectrum_df,
            peak_df=peak_df,
            query_masses=query_masses,
            mass_tols=mass_tols,
            ion_names=query_ion_names,
            marker_colors=marker_colors,
            query_start_rt_sec=query_rt_sec-self.rt_sec_win/2,
            query_stop_rt_sec=query_rt_sec+self.rt_sec_win/2,
            precursor_left_mz=precursor_left_mz,
            precursor_right_mz=precursor_right_mz,
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
    
    def _init_plot(self, rows=1):
        self.fig = make_subplots(
            rows=rows, 
            cols=1,
            shared_xaxes=True,
            x_title=f'RT ({self.plot_rt_unit})',
            y_title='intensity',
            vertical_spacing=0.2/rows,
        )
        self.traces:typing.List[XIC_Trace] = [
            XIC_Trace(
                fig=self.fig, row=i+1, 
                plot_rt_unit=self.plot_rt_unit,
            ) 
            for i in range(rows)
        ]

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
        spectrum_df:pd.DataFrame,
        peak_df:pd.DataFrame,
        query_masses:np.ndarray,
        mass_tols:np.ndarray,
        ion_names:typing.List[str],
        marker_colors:typing.List,
        query_start_rt_sec:float, 
        query_stop_rt_sec:float, 
        precursor_left_mz:float = -1.0,
        precursor_right_mz:float = -1.0,
        query_intensities:np.ndarray = None,
    )->go.Figure:

        spec_idxes = get_spec_idxes_from_df(
            spectrum_df,
            query_start_rt_sec=query_start_rt_sec,
            query_stop_rt_sec=query_stop_rt_sec,
            precursor_left_mz=precursor_left_mz,
            precursor_right_mz=precursor_right_mz,
        )

        rt_values = spectrum_df.rt.values[spec_idxes]
        if self.plot_rt_unit == "second":
            rt_values = rt_values * 60
        matched_mzs, matched_intens = match_batch_spec(
            spec_idxes, 
            peak_mzs=peak_df.mz.values,
            peak_intens=peak_df.intensity.values,
            peak_start_idxes=spectrum_df.peak_start_idx.values,
            peak_stop_idxes=spectrum_df.peak_stop_idx.values,
            query_mzs=query_masses,
            query_mz_tols=mass_tols,
        )

        if query_intensities is None:
            query_intensities = np.zeros_like(query_masses)
        else:
            query_intensities /= query_intensities.max()

        for i_query, (
            ion_name, query_mass, 
            query_inten, marker_color
        ) in enumerate(zip(
            ion_names, query_masses, 
            query_intensities, marker_colors
        )):
            if (matched_intens[:,i_query]==0).all(): continue
            self._add_one_trace(
                rt_values, matched_intens[:,i_query],
                label=self.label_format.format(ion_name=ion_name, mz=query_mass),
                legend_group=self.legend_group.format(ion_name=ion_name),
                marker_color=marker_color,
            )
            if query_inten > 0:
                self._add_one_trace(
                    rt_values, -query_inten*matched_intens[:,i_query],
                    label=self.label_format.format(ion_name=ion_name, mz=query_mass),
                    legend_group=self.legend_group.format(ion_name=ion_name),
                    marker_color=marker_color,
                )
    def _add_one_trace(self,
        rt_values:np.ndarray,
        matched_intens:np.ndarray,
        label:str, 
        legend_group:str, 
        marker_color:str,
    ):
        self.fig.add_trace(
            plot_line(
                rt_values, 
                matched_intens,
                name=label,
                legend_group=legend_group,
                marker_color=marker_color, 
                x_text="RT",
            ),
            row=self.row, col=self.col,
        )

def get_spec_idxes_from_df(
    spectrum_df:pd.DataFrame,
    query_start_rt_sec,
    query_stop_rt_sec,
    precursor_left_mz,
    precursor_right_mz,
):
    if "multinotch" in spectrum_df.columns:
        return find_multinotch_spec_idxes(
            spectrum_df.rt.values,
            spectrum_df.multinotch.values,
            spectrum_df.ms_level.values,
            query_start_rt=query_start_rt_sec/60,
            query_stop_rt=query_stop_rt_sec/60,
            query_left_mz=precursor_left_mz,
            query_right_mz=precursor_right_mz
        )
    else:
        return find_spec_idxes(
            spectrum_df.rt.values,
            spectrum_df.isolation_lower_mz.values,
            spectrum_df.isolation_upper_mz.values,
            query_start_rt=query_start_rt_sec/60,
            query_stop_rt=query_stop_rt_sec/60,
            query_left_mz=precursor_left_mz,
            query_right_mz=precursor_right_mz
        )
    
def get_peak_area(
    spectrum_df:pd.DataFrame,
    peak_df:pd.DataFrame,
    query_masses:np.ndarray,
    mass_tols:np.ndarray,
    query_start_rt_sec:float, 
    query_stop_rt_sec:float,
    precursor_left_mz:float = -1.0,
    precursor_right_mz:float = -1.0,
)->np.ndarray:
    spec_idxes = get_spec_idxes_from_df(
        spectrum_df,
        query_start_rt_sec=query_start_rt_sec,
        query_stop_rt_sec=query_stop_rt_sec,
        precursor_left_mz=precursor_left_mz,
        precursor_right_mz=precursor_right_mz,
    )

    rt_values = spectrum_df.rt.values[spec_idxes]

    matched_mzs, matched_intens = match_batch_spec(
        spec_idxes, 
        peak_mzs=peak_df.mz.values,
        peak_intens=peak_df.intensity.values,
        peak_start_idxes=spectrum_df.peak_start_idx.values,
        peak_stop_idxes=spectrum_df.peak_stop_idx.values,
        query_mzs=query_masses,
        query_mz_tols=mass_tols,
    )

    return np.trapz(
        y=matched_intens, x=rt_values*60, axis=0
    )