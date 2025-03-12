from typing import Tuple

import pandas as pd
from alphabase.peptide.fragment import (
    compress_fragment_indices,
    remove_unused_fragments,
)


def remove_unused_peaks(
    spectrum_df: pd.DataFrame,
    peak_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Removes unused peaks of removed spectra,
    reannotates the peak_start_idx and peak_stop_idx

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        Spectrum dataframe which contains peak_start_idx and peak_stop_idx columns

    peak_df : pd.DataFrame
        The peak dataframe which should be compressed by removing unused peaks.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The reindexed spectrum DataFrame and the sliced peak DataFrame
    """

    spectrum_df, (peak_df,) = remove_unused_fragments(
        spectrum_df,
        (peak_df,),
        frag_start_col="peak_start_idx",
        frag_stop_col="peak_stop_idx",
    )
    return spectrum_df, peak_df

    spectrum_df = spectrum_df.sort_values(["peak_start_idx"], ascending=True)
    frag_idx = spectrum_df[["peak_start_idx", "peak_stop_idx"]].values

    new_frag_idx, fragment_pointer = compress_fragment_indices(frag_idx)

    spectrum_df[["peak_start_idx", "peak_stop_idx"]] = new_frag_idx
    spectrum_df = spectrum_df.sort_index()

    return (spectrum_df, peak_df.iloc[fragment_pointer].copy().reset_index(drop=True))
