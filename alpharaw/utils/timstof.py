import typing

import pandas as pd

from alpharaw.bruker.timstof import TimsTOF
from alpharaw.ms_data_base import MSData_Base
from alpharaw.utils.df_processing import remove_unused_peaks
from alpharaw.wrappers.alphatims_wrapper import AlphaTimsWrapper

# IMPORTANT NOTE: all references to "alphatims" in this module are legacy names from the times when the TimsTOF class was still part of AlphaTims.


# TODO: rename, this name is confusing now that AlphaTims functionality has been moved to here.
def convert_to_alphatims(
    spec_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    dda: bool = False,
) -> typing.Tuple[MSData_Base, TimsTOF]:
    """
    Convert any spectrum dataframe or sliced spectrum dataframe
    and its peak dataframe into a TimsTOF object (AlphaTimsWrapper).

    Args:
        spec_df (pd.DataFrame):
            spectrum dataframe or sliced spectrum dataframe in AlphaRaw's format.
        peak_df (pd.DataFrame):
            peak dataframe in AlphaRaw's format by removing unused peaks in spec_df.
        dda (bool):
            if dda data.

    Returns:
        MSData_Base: AlphaRaw object
        TimsTOF: TimsTOF object (AlphaTimsWrapper).
    """
    spec_df, peak_df = remove_unused_peaks(spec_df, peak_df)
    ms_data = MSData_Base()
    ms_data.spectrum_df = spec_df
    ms_data.peak_df = peak_df

    return ms_data, AlphaTimsWrapper(ms_data, dda=dda)
