import pandas as pd

from alphabase.peptide.fragment import compress_fragment_indices

from typing import Tuple

def remove_unused_peaks(
    spectrum_df: pd.DataFrame, 
    peak_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Removes unused peaks of removed precursors, 
    reannotates the peak_start_idx and peak_stop_idx
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe which contains peak_start_idx and peak_stop_idx columns
    
    peak_df : pd.DataFrame
        The peak dataframe which should be compressed by removing unused peaks. 
    
    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The reindexed spectrum DataFrame and the sliced peak DataFrame
    """

    spectrum_df = spectrum_df.sort_values(['peak_start_idx'], ascending=True)
    frag_idx = spectrum_df[['peak_start_idx','peak_stop_idx']].values

    new_frag_idx, fragment_pointer = compress_fragment_indices(frag_idx)

    spectrum_df[['peak_start_idx','peak_stop_idx']] = new_frag_idx
    spectrum_df = spectrum_df.sort_index()


    return (
        spectrum_df, 
        peak_df.iloc[fragment_pointer].copy().reset_index(drop=True)
    )