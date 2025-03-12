import typing
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .df_utils import (
    make_psm_plot_df,
    make_query_df_for_peptide,
)
from .plot_utils import plot_scatter

color_map: dict = defaultdict(lambda: "brown")
"""
The colors of different peak annotations. By default:
```
{
    '-': 'lightgrey', # umnatched peaks
    'a': 'darkskyblue',
    'b': 'blue',
    'c': 'skyblue',
    'x': 'darkred',
    'y': 'red',
    'z': 'deeppink',
    'M': 'purple', # precursor
    'Y': 'orange',
    'B': 'darkgreen',
}
```
For other annotations, the default color is "brown".
"""
color_map.update(
    {
        "-": "lightgrey",  # umnatched peaks
        "a": "darkskyblue",
        "b": "blue",
        "c": "skyblue",
        "x": "darkred",
        "y": "red",
        "z": "deeppink",
        "M": "purple",  # precursor
        "Y": "orange",
        "B": "darkgreen",
    }
)


def plot_multi_psms(
    spec_masses_list: typing.List[np.ndarray],
    spec_intens_list: typing.List[np.ndarray],
    sequence: str,
    mods: str,
    mod_sites: str,
    charge: int,
    title: str = "",
    ppm: float = 20.0,
    charged_frag_types: list = ["b_z1", "b_z2", "y_z1", "y_z2"],
    include_fragments: bool = True,
    include_precursor_isotopes: bool = False,
    max_isotope: int = 6,
    min_frag_mz: float = 100.0,
    plot_unmatched_peaks=True,
    match_mode: typing.Literal["closest", "highest"] = "closest",
    plot_template="plotly_white",
    plot_height=600,
    query_left_margin: float = 100000.0,
    query_right_margin: float = 100000.0,
):
    """
    Annotate multiple spectra for a peptide in a single plotly go.Figure.

    Parameters
    ----------
    spec_masses_list : typing.List[np.ndarray]
        The list of peak masses of multiple spectra to be annotated
    spec_intens_list : typing.List[np.ndarray]
        The list of peak intensities of multiple spectra to be annotated
    sequence : str
        Peptide sequence
    mods : str
        Modifications in alphabase format
    mod_sites : str
        Modification sites in alphabase format
    charge : int
        Charge state of the precursor
    title : str, optional
        Plotting title, by default ""
    ppm : float, optional
        Matching mass tolerance in ppm, by default 20.0
    charged_frag_types : list, optional
        Fragment charge states, by default ["b_z1","b_z2","y_z1","y_z2"]
    include_fragments : bool, optional
        If annotate fragments, by default True
    include_precursor_isotopes : bool, optional
        If annotate precursor isotopes, by default False
    max_isotope : int, optional
        Maximal number of isotopes, by default 6
    min_frag_mz : float, optional
        Minimal fragment m/z value to annotate, by default 100.0
    plot_unmatched_peaks : bool, optional
        If plot unmatched peaks (ion_name='-'), by default True
    match_mode : "closest", "highest", optional
        Match the closest or highest peak within the matching tolerance, by default "closest"
    plot_template : str, optional
        Plot template, by default 'plotly_white'
    plot_height : int, optional
        Plot height, by default 600
    query_left_margin : float, optional
        Slice margin to the left mz, by default 100000.0
    query_right_margin : float, optional
        Slice martin to the right mz, by default 100000.0

    Returns
    -------
    go.Figure
        The annotation plots of multiple spectra
    """
    plot_df = make_query_df_for_peptide(
        sequence,
        mods,
        mod_sites,
        charge,
        charged_frag_types=charged_frag_types,
        include_fragments=include_fragments,
        include_precursor_isotopes=include_precursor_isotopes,
        max_isotope=max_isotope,
        min_frag_mz=min_frag_mz,
    )

    slice_masses_list = []
    slice_intens_list = []
    for spec_masses, spec_intens in zip(spec_masses_list, spec_intens_list):
        _slice = (spec_masses >= plot_df.mz.min() - query_left_margin) & (
            spec_masses <= plot_df.mz.max() + query_right_margin
        )
        slice_masses_list.append(spec_masses[_slice])
        slice_intens_list.append(spec_intens[_slice])

    return plot_multi_spectra(
        slice_masses_list,
        slice_intens_list,
        query_masses=plot_df.mz.values,
        query_ion_names=plot_df.ion_name.values,
        query_mass_tols=plot_df.mz.values * ppm * 1e-6,
        title=title,
        plot_unmatched_peaks=plot_unmatched_peaks,
        match_mode=match_mode,
        plot_template=plot_template,
        plot_height=plot_height,
    )


def plot_multi_spectra(
    spec_masses_list: typing.List[np.ndarray],
    spec_intens_list: typing.List[np.ndarray],
    query_masses: np.ndarray,
    query_ion_names: typing.List[str],
    query_mass_tols: np.ndarray,
    title: str = "",
    plot_unmatched_peaks=True,
    match_mode: typing.Literal["closest", "highest"] = "closest",
    plot_template="plotly_white",
    plot_height=600,
):
    """
    Annotate multiple spectra for given queries in a single plotly go.Figure.

    Parameters
    ----------
    spec_masses_list : typing.List[np.ndarray]
        The list of peak masses of multiple spectra to be annotated
    spec_intens_list : typing.List[np.ndarray]
        The list of peak intensities of multiple spectra to be annotated
    query_masses : np.ndarray
        The query m/z values
    query_ion_names : typing.List[str]
        The query ion names
    query_mass_tols : np.ndarray
        The query mass tolerance in Da
    title : str, optional
        The plot title, by default ""
    plot_unmatched_peaks : bool, optional
        If plot unmatched peaks (ion_name='-'), by default True
    match_mode : "closest", "highest", optional
        Match the closest or highest peak within the matching tolerance, by default "closest"
    plot_template : str, optional
        Plot template, by default 'plotly_white'
    plot_height : int, optional
        Plot height, by default 600

    Returns
    -------
    go.Figure
        The annotation plots of multiple spectra
    """
    plot_dfs = []
    for spec_masses, spec_intens in zip(spec_masses_list, spec_intens_list):
        plot_dfs.append(
            make_psm_plot_df(
                spec_masses=spec_masses,
                spec_intensities=spec_intens,
                query_masses=query_masses,
                query_ion_names=query_ion_names,
                query_mass_tols=query_mass_tols,
                query_frag_idxes=np.zeros_like(query_masses, dtype=np.int64),
                modified_sequence="",
                match_mode=match_mode,
            )
        )
    fig = make_subplots(
        rows=len(plot_dfs),
        cols=1,
        shared_xaxes=True,
    )
    layout_vlines = []
    for i in range(len(plot_dfs)):
        _plot = PeakPlot(fig, i + 1)
        _plot.plot(plot_dfs[i], plot_unmatched_peaks=plot_unmatched_peaks)
        layout_vlines.extend(_plot.layout_vlines)

    fig.update_layout(shapes=layout_vlines)

    fig.update_layout(
        template=plot_template,
        title=dict(text=title, yanchor="bottom"),
        hovermode="x",
        height=plot_height,
    )
    fig.update_xaxes(matches="x")
    fig.update_yaxes(
        title="intensity",
    )
    return fig


class PSM_Plot:
    """
    The main class for spectrum annotation of a PSM. It contains three plots:
    1. Ladder plot (`FragCoveragePlot`) for fragment coverage of the peptide.
    2. Peak annotation plot (`PeakPlot`) for the spectrum.
    3. Matching mass error plot (`MassErrPlot`) for the matched peaks.
    """

    vertical_spacing = 0.05
    template = "plotly_white"
    plot_height = 600

    def __init__(
        self,
        peak_plot_rows: int = 4,
        mass_err_plot_rows: int = 1,
        frag_coverage_plot_rows: int = 1,
        frag_coverage: bool = True,
    ):
        """
        Parameters
        ----------
        peak_plot_rows : int, optional
            The height (ratio) of peak plot, by default 4
        mass_err_plot_rows : int, optional
            The height (ratio) of mass error plot, by default 1
        frag_coverage_plot_rows : int, optional
            The height (ratio) of fragment coverage plot, by default 1
        frag_coverage : bool, optional
            If plot fragment coverage, by default True
        """
        specs = []
        if frag_coverage:
            specs.append(
                [{"rowspan": frag_coverage_plot_rows, "colspan": 3}, None, None]
            )
            specs.extend([[None, None, None]] * (frag_coverage_plot_rows - 1))
        specs.append([{"rowspan": peak_plot_rows, "colspan": 3}, None, None])
        specs.extend([[None, None, None]] * (peak_plot_rows - 1))
        specs.append([{"rowspan": mass_err_plot_rows, "colspan": 3}, None, None])
        specs.extend([[None, None, None]] * (mass_err_plot_rows - 1))
        if not frag_coverage:
            specs.append(
                [{"rowspan": frag_coverage_plot_rows, "colspan": 3}, None, None]
            )
            specs.extend([[None, None, None]] * (frag_coverage_plot_rows - 1))

        if frag_coverage:
            (
                frag_cov_row,
                peak_row,
                mass_err_row,
            ) = np.cumsum(
                [
                    1,
                    frag_coverage_plot_rows,
                    peak_plot_rows,
                ]
            )
        else:
            (
                peak_row,
                mass_err_row,
                frag_cov_row,
            ) = np.cumsum([1, peak_plot_rows, mass_err_plot_rows])

        self.specs = specs
        self.peak_row = peak_row
        self.mass_err_row = mass_err_row
        self.frag_cov_row = frag_cov_row
        self.rows = peak_plot_rows + mass_err_plot_rows + frag_coverage_plot_rows

    def plot(
        self,
        plot_df: pd.DataFrame,
        sequence: str,
        title: str,
        plot_unmatched_peaks: bool = False,
    ) -> go.Figure:
        """
        Main entry of `PSM_Plot` for peak annotation.

        Parameters
        ----------
        plot_df : pd.DataFrame
            The plot_df can be generated by
            :func:`alpharaw.viz.df_utils.make_psm_plot_df_for_peptide`,
            :func:`alpharaw.viz.df_utils.make_psm_plot_for_frag_dfs`, or
            :func:`alpharaw.viz.df_utils.make_psm_plot_df`.
        sequence : str
            Peptide sequence, for fragment coverage plot
        title : str
            Plot title
        plot_unmatched_peaks : bool, optional
            If plot unmatched peaks with ion_name `-`, by default False

        Returns
        -------
        go.Figure
            Peak annotation plot in plotly go.Figure
        """
        if "pcc" in plot_df.columns and len(plot_df) > 0:
            title = f"{title} (R={plot_df.pcc.values[0]:.3f})"
        self._init_plot(title)

        self.peak_plot.plot(plot_df, plot_unmatched_peaks=plot_unmatched_peaks)
        self.mass_err_plot.plot(
            plot_df,
        )
        self.frag_cov_plot.plot(plot_df, sequence)

        self.fig.update_layout(shapes=self.peak_plot.layout_vlines)

        return self.fig

    def _init_plot(self, title):
        self.fig = make_subplots(
            rows=self.rows,
            cols=3,
            shared_xaxes=True,
            specs=self.specs,
            vertical_spacing=self.vertical_spacing,
            column_widths=[0.25, 0.5, 0.25],
        )

        self.peak_plot = PeakPlot(
            self.fig,
            self.peak_row,
        )
        self.mass_err_plot = MassErrPlot(
            self.fig,
            self.mass_err_row,
        )
        self.frag_cov_plot = FragCoveragePlot(self.fig, self.frag_cov_row)

        self.fig.update_layout(
            template=self.template,
            title=dict(text=title, yanchor="bottom"),
            hovermode="x",
            height=self.plot_height,
        )
        self.fig.update_xaxes(matches="x")
        self.fig.update_yaxes(
            title="intensity",
        )


class MassErrPlot:
    def __init__(self, fig_subplots, row):
        self.fig = fig_subplots
        self.row = row
        self.col = 1
        self.hovertemplate = "%{hovertext}<br>" "<b>m/z:</b> %{x}<br><b>ppm:</b> %{y}"

    def plot(self, plot_df: pd.DataFrame):
        if "color" not in plot_df.columns:
            plot_df["color"] = [
                color_map[ion_type] for ion_type in plot_df.ion_name.str[0].values
            ]

        for color, df in plot_df.query("intensity>0 and ion_name!='-'").groupby(
            "color"
        ):
            self._plot_one_type(df, color)

        self.fig.update_yaxes(title_text="ppm", row=self.row, col=self.col)
        return self.fig

    def _plot_one_type(self, df, color):
        self.fig.add_trace(
            plot_scatter(
                df.mz.values,
                df.ppm_err.round(4).values,
                color=color,
                marker_size=5,
                hovertext=df.ion_name.values,
                hovertemplate=self.hovertemplate,
                name=color,
            ),
            row=self.row,
            col=self.col,
        )


class PeakPlot:
    fig: go.Figure
    row: int = 1
    col: int = 1
    hovertemplate = (
        "<b>%{hovertext}</b><br>" "<b>m/z:</b> %{x}<br>" "<b>intensity:</b> %{y}"
    )
    peak_line_width = 1.5

    def __init__(self, fig: go.Figure, row: int, col: int = 1):
        self.fig = fig
        self.row = row
        self.col = col

    def plot(
        self,
        plot_df: pd.DataFrame,
        plot_unmatched_peaks: bool = True,
    ) -> go.Figure:
        if "color" not in plot_df.columns:
            plot_df["color"] = [
                color_map[ion_type] for ion_type in plot_df.ion_name.str[0].values
            ]
        plot_df.loc[plot_df.ion_name == "-", "color"] = color_map["-"]
        if plot_unmatched_peaks:
            _df = plot_df[plot_df.ion_name == "-"]
            self.fig.add_trace(
                plot_scatter(
                    _df.mz,
                    _df.intensity,
                    color=color_map["-"],
                    marker_size=1,
                    hovertext=None,
                    hovertemplate=None,
                    name="",
                ),
                row=self.row,
                col=self.col,
            )
        else:
            plot_df = plot_df.query('ion_name != "-"')

        matched_df = plot_df.query('ion_name != "-"')
        for color, df in matched_df.groupby("color"):
            self._plot_one_ion_type_scatter(df, color)

        self._get_peak_vlines(
            plot_df,
        )

        self._plot_ion_name(matched_df)

        return self.fig

    def _plot_one_ion_type_scatter(self, df, color):
        _df = df.query("intensity>0")
        self.fig.add_trace(
            plot_scatter(
                _df.mz,
                _df.intensity,
                color=color,
                marker_size=1,
                hovertext=_df.ion_name,
                hovertemplate=self.hovertemplate,
                name="",
            ),
            row=self.row,
            col=self.col,
        )
        _df = df.query("intensity<0")
        self.fig.add_trace(
            plot_scatter(
                _df.mz,
                _df.intensity,
                color=color,
                marker_size=1,
                hovertext=_df.ion_name,
                hovertemplate=self.hovertemplate,
                name="",
            ),
            row=self.row,
            col=self.col,
        )

    def _plot_ion_name(self, plot_df):
        df = plot_df.query("intensity>0")
        max_inten = df.intensity.max()
        yshift = max_inten * 0.02
        for mz, inten, ion in df[["mz", "intensity", "ion_name"]].values:
            self.fig.add_annotation(
                x=mz,
                y=inten + yshift,
                text=ion,
                textangle=-90,
                font_size=10,
                row=self.row,
                col=self.col,
            )

        pred_df = plot_df.query("intensity<0")
        pred_df = pred_df[~pred_df.ion_name.isin(set(df.ion_name))]
        for mz, inten, ion in pred_df[["mz", "intensity", "ion_name"]].values:
            self.fig.add_annotation(
                x=mz,
                y=inten - yshift,
                text=ion,
                textangle=-90,
                font_size=10,
                ay=inten - yshift - max_inten * (0.28 + len(ion) / 60),
                ayref=f"y{self.row}",
                yref=f"y{self.row}",
                row=self.row,
                col=self.col,
            )

    def _get_peak_vlines(
        self,
        plot_df,
    ):
        self.layout_vlines = [
            dict(
                type="line",
                xref=f"x{self.row}",
                yref=f"y{self.row}",
                x0=plot_df.loc[i, "mz"],
                y0=0,
                x1=plot_df.loc[i, "mz"],
                y1=plot_df.loc[i, "intensity"],
                line=dict(
                    color=plot_df.loc[i, "color"],
                    width=self.peak_line_width,
                ),
            )
            for i in plot_df.index
        ]
        # for i in plot_df.index:
        #     self.fig.add_shape(type='line',
        #         x0=plot_df.loc[i, 'mz'],
        #         x1=plot_df.loc[i, 'mz'],
        #         y0=0,
        #         y1=plot_df.loc[i, 'intensity'],
        #         line_color = color_map[plot_df.loc[i, 'ion_name'][0]],
        #         line_width = self.peak_line_width,
        #         row=self.row,
        #         col=self.col,
        #     )


class FragCoveragePlot:
    def __init__(self, fig_subplots, row):
        self.fig = fig_subplots
        self.row = row
        self.col = 1
        self.font_size_sequence = 14
        self.font_size_coverage = 8
        self.mod_aa_color = "firebrick"

    def plot(
        self,
        plot_df,
        sequence,
    ):
        if "color" not in plot_df.columns:
            plot_df["color"] = [
                color_map[ion_type] for ion_type in plot_df.ion_name.str[0].values
            ]
        if len(sequence) > 0:
            d = (plot_df.mz.max() - plot_df.mz.min()) * 2 / len(sequence)
            aa_x_position_name = np.linspace(
                plot_df.mz.min() + d, plot_df.mz.max() - d, len(sequence) + 1
            )

            colors = [None] * len(sequence)
            if "mod_sites" in plot_df.columns:
                mod_sites = plot_df.mod_sites.values[0]
                if len(mod_sites) > 0:
                    for site in mod_sites.split(";"):
                        site = int(site)
                        if site == 0 or site == -1:
                            colors[site] = self.mod_aa_color
                        else:
                            colors[site - 1] = self.mod_aa_color

            self._plot_sequence(sequence, aa_x_position_name, colors)
            for ion_type in np.unique(plot_df.ion_name.str[0]):
                self._plot_coverage_one_frag_type(
                    plot_df,
                    sequence,
                    aa_x_position_name,
                    ion_type,
                )
            # self._plot_coverage_one_frag_type(
            #     plot_df, sequence, aa_x_position_name,
            #     'b', color_map['b'],
            # )
            # self._plot_coverage_one_frag_type(
            #     plot_df, sequence, aa_x_position_name,
            #     'y', color_map['y'],
            # )
            self.fig.update_yaxes(
                visible=False,
                range=(-1.1, 1.1),
                row=self.row,
                col=self.col,
            )
            self.fig.update_xaxes(
                visible=False,
                row=self.row,
                col=self.col,
            )
        return self.fig

    def _plot_sequence(
        self,
        sequence,
        aa_x_position_name,
        colors,
    ):
        for i, aa in enumerate(sequence):
            self.fig.add_annotation(
                dict(
                    text=aa,
                    x=aa_x_position_name[i],
                    y=0,
                    showarrow=False,
                    font=dict(
                        size=self.font_size_sequence,
                        color=colors[i],
                    ),
                    yshift=1,
                    align="center",
                ),
                row=self.row,
                col=self.col,
            )

    def _plot_coverage_one_frag_type(
        self,
        plot_df,
        sequence,
        aa_x_position_name,
        ion_type,
    ):
        nAA = len(sequence)
        plot_df = plot_df[plot_df.ion_name.str.startswith(ion_type)].query(
            "intensity>0"
        )

        covs = np.zeros(
            max(plot_df.fragment_site.max() + 2 if len(plot_df) > 0 else 0, nAA),
            dtype=np.int64,
        )
        cov_colors = [""] * len(covs)
        if ion_type in "abc":
            for frag_idx, color in zip(
                plot_df.fragment_site.values, plot_df.color.values
            ):
                covs[frag_idx] = 1
                cov_colors[frag_idx] = color
        elif ion_type in "xyz":
            for frag_idx, color in zip(
                plot_df.fragment_site.values, plot_df.color.values
            ):
                covs[frag_idx + 1] = 1
                cov_colors[frag_idx + 1] = color

        def get_position_name(ion_type, i):
            if ion_type in "abc":
                return dict(
                    x=[
                        aa_x_position_name[i],
                        aa_x_position_name[i]
                        + (aa_x_position_name[i + 1] - aa_x_position_name[i]) / 2,
                        aa_x_position_name[i]
                        + (aa_x_position_name[i + 1] - aa_x_position_name[i]) / 2,
                    ],
                    y=[0.6, 0.6, 0.4],
                    tx=aa_x_position_name[i]
                    + (aa_x_position_name[i + 1] - aa_x_position_name[i]) / 4,
                    ty=1.0,
                    s=i + 1,
                )
            elif ion_type in "xyz":
                return dict(
                    x=[
                        aa_x_position_name[i],
                        aa_x_position_name[i]
                        - (aa_x_position_name[i + 1] - aa_x_position_name[i]) / 2,
                        aa_x_position_name[i]
                        - (aa_x_position_name[i + 1] - aa_x_position_name[i]) / 2,
                    ],
                    y=[-0.6, -0.6, -0.4],
                    tx=aa_x_position_name[i]
                    - (aa_x_position_name[i + 1] - aa_x_position_name[i]) / 4,
                    ty=-1.0,
                    s=nAA - i,
                )

        for i, (cov, color) in enumerate(zip(covs[:nAA], cov_colors[:nAA])):
            if cov:
                pos = get_position_name(ion_type, i)
                self.fig.add_trace(
                    go.Scatter(
                        x=pos["x"],
                        y=pos["y"],
                        mode="lines",
                        showlegend=False,
                        marker_color=color,
                        line_width=1,
                        hoverinfo="skip",
                    ),
                    row=self.row,
                    col=self.col,
                )
                self.fig.add_annotation(
                    dict(
                        text=f"{ion_type}{pos['s']}",
                        x=pos["tx"],
                        y=pos["ty"],
                        showarrow=False,
                        font_size=self.font_size_coverage,
                    ),
                    row=self.row,
                    col=self.col,
                )
