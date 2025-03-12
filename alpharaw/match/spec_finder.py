from typing import List

import numba
import numpy as np
import pandas as pd


def find_spec_idxes_by_rt(
    spectrum_df: pd.DataFrame,
    query_start_rt: float,
    query_stop_rt: float,
    query_left_mz: float,
    query_right_mz: float,
) -> np.ndarray:
    """
    Find MS2 spectrum indices (int32) from the `spectrum_df`
    by given RT window and precursor m/z window.

    Parameters
    ----------
    spectrum_df : pd.DataFrame
        Spectrum dataframe to find spectrum indices.
    query_start_rt : float
        Left RT of the query RT window.
    query_stop_rt : float
        Right RT of the query RT window.
    query_left_mz : float
        Left m/z of the query m/z window.
    query_right_mz : float
        Right m/z of the query m/z window.

    Returns
    -------
    ndarray[int32]
        Result spectrum indices. `int32` is used here as there will be
        no more than 2 billions of spectra in a raw file.
    """
    if "multinotch" in spectrum_df.columns:
        # if multinotch, there are multiple isolation windows of MS2 spectra.
        return find_multinotch_spec_idxes(
            spec_rts=spectrum_df.rt.values,
            spec_multinotch_wins=spectrum_df.multinotch.values,
            spec_ms_levels=spectrum_df.ms_levels.values,
            query_start_rt=query_start_rt,
            query_stop_rt=query_stop_rt,
            query_left_mz=query_left_mz,
            query_right_mz=query_right_mz,
        )
    else:
        # normal isolation windows (one window to one MS2 spectrum)
        return find_spec_idxes(
            spec_rts=spectrum_df.rt.values,
            spec_isolation_lower_mzs=spectrum_df.isolation_lower_mz.values,
            spec_isolation_upper_mzs=spectrum_df.isolation_upper_mz.values,
            query_start_rt=query_start_rt,
            query_stop_rt=query_stop_rt,
            query_left_mz=query_left_mz,
            query_right_mz=query_right_mz,
        )


def find_multinotch_spec_idxes(
    spec_rts: np.ndarray,
    spec_multinotch_wins: List[List],
    spec_ms_levels: np.ndarray,
    query_start_rt: float,
    query_stop_rt: float,
    query_left_mz: float,
    query_right_mz: float,
) -> np.ndarray:
    """
    Find MS2 spectrum indices (int32) from the "multinotch" `spectrum_df`
    by given RT window and precursor m/z window.
    "multinotch" means there are multiple isolation windows of MS2 spectra.

    Parameters
    ----------
    spec_rts : np.ndarray
        RT values of the spectra.
    spec_multinotch_wins : List[List]
        List (num of spectra) of list (multiple isolation windows).
    spec_ms_levels : np.ndarray
        MS levels of the spectra.
    query_start_rt : float
        Left RT of the query RT window.
    query_stop_rt : float
        Right RT of the query RT window.
    query_left_mz : float
        Left m/z of the query m/z window.
    query_right_mz : float
        Right m/z of the query m/z window.

    Returns
    -------
    np.ndarray[int32]
        Result spectrum indices.
    """
    start_idx = np.searchsorted(spec_rts, query_start_rt)
    stop_idx = np.searchsorted(spec_rts, query_stop_rt) + 1
    spec_idxes = []
    for ispec in range(start_idx, stop_idx):
        for win_left, win_right in spec_multinotch_wins[ispec]:
            if spec_ms_levels[ispec] == 1:
                if query_left_mz <= 0:
                    spec_idxes.append(ispec)
            elif max(query_left_mz, win_left) <= min(query_right_mz, win_right):
                spec_idxes.append(ispec)
    return np.array(spec_idxes, dtype=np.int32)


@numba.njit
def find_dia_spec_idxes_same_window(
    spec_rt_values: np.ndarray,
    query_rt_values: np.ndarray,
    max_spec_per_query: int,
) -> np.ndarray:
    """
    For given array of query RT values, find spectrum indices
    from the subset of spectra within the same normal DIA m/z window.
    This function is numba accelerated.

    Parameters
    ----------
    spec_rt_values : np.ndarray
        RT values of given DIA spectra.
    query_rt_values : np.ndarray
        Query RT values.
    max_spec_per_query : int
        Return maximal spectrum indices (scan windows) for the given query.

    Returns
    -------
    ndarray[int32]
        Result spectrum indices with shape (query num, max_spec_per_query).
    """
    rt_idxes = np.searchsorted(spec_rt_values, query_rt_values)

    spec_idxes = np.full((len(rt_idxes), max_spec_per_query), -1, dtype=np.int32)
    n = max_spec_per_query // 2

    for iquery in range(len(rt_idxes)):
        if rt_idxes[iquery] < n:
            spec_idxes[iquery, :] = np.arange(0, max_spec_per_query)
        else:
            spec_idxes[iquery, :] = np.arange(
                rt_idxes[iquery] - n, rt_idxes[iquery] - n + max_spec_per_query
            )
    return spec_idxes


@numba.njit
def find_spec_idxes(
    spec_rts: np.ndarray,
    spec_isolation_lower_mzs: np.ndarray,
    spec_isolation_upper_mzs: np.ndarray,
    query_start_rt: float,
    query_stop_rt: float,
    query_left_mz: float,
    query_right_mz: float,
) -> np.ndarray:
    """
    Find MS2 spectrum indices (int32) from all the spectra
    by given RT window and precursor m/z window.
    This function is numba accelerated.

    Parameters
    ----------
    spec_rts : np.ndarray
        RT values of the spectra.
    spec_isolation_lower_mzs : np.ndarray
        Left m/z values of the isolation windows.
    spec_isolation_upper_mzs : np.ndarray
        Right m/z values of the isolation windows.
    query_start_rt : float
        Left RT of the query RT window.
    query_stop_rt : float
        Right RT of the query RT window.
    query_left_mz : float
        Left m/z of the query m/z window.
    query_right_mz : float
        Right m/z of the query m/z window.

    Returns
    -------
    np.ndarray[int32]
        Result spectrum indices.
    """
    rt_start_idx = np.searchsorted(spec_rts, query_start_rt)
    rt_stop_idx = np.searchsorted(spec_rts, query_stop_rt) + 1

    spec_idxes = []

    for ispec in range(rt_start_idx, rt_stop_idx):
        if max(query_left_mz, spec_isolation_lower_mzs[ispec]) <= min(
            query_right_mz, spec_isolation_upper_mzs[ispec]
        ):
            spec_idxes.append(ispec)
    return np.array(spec_idxes, dtype=np.int32)


@numba.njit
def find_batch_spec_idxes(
    spec_rts: np.ndarray,
    spec_isolation_lower_mzs: np.ndarray,
    spec_isolation_upper_mzs: np.ndarray,
    query_start_rts: np.ndarray,
    query_stop_rts: np.ndarray,
    query_left_mzs: np.ndarray,
    query_right_mzs: np.ndarray,
    max_spec_per_query: int,
) -> np.ndarray:
    """
    Find MS2 spectrum indices (int32) from all the spectra
    by the given batch of RT windows and precursor m/z windows.
    This function is numba accelerated.

    Parameters
    ----------
    spec_rts : np.ndarray
        RT values of the spectra.
    spec_isolation_lower_mzs : np.ndarray
        Left m/z values of the isolation windows.
    spec_isolation_upper_mzs : np.ndarray
        Right m/z values of the isolation windows.
    query_start_rts : np.ndarray
        Left RT values of the query RT windows.
    query_stop_rts : np.ndarray
        Right RT values of the query RT windows.
    query_left_mzs : np.ndarray
        Left m/z values of the query m/z windows.
    query_right_mzs : np.ndarray
        Right m/z values of the query m/z windows.
    max_spec_per_query : int
        Return maximal spectrum indices (scan windows) for the given query.

    Returns
    -------
    ndarray[int32]
        Result spectrum indices with shape (query num, max_spec_per_query).
    """
    rt_start_idxes = np.searchsorted(spec_rts, query_start_rts)
    rt_stop_idxes = np.searchsorted(spec_rts, query_stop_rts) + 1

    spec_idxes = np.full((len(query_left_mzs), max_spec_per_query), -1, dtype=np.int32)
    for iquery in range(len(rt_start_idxes)):
        idx_list = []
        for ispec in range(rt_start_idxes[iquery], rt_stop_idxes[iquery]):
            if max(query_left_mzs[iquery], spec_isolation_lower_mzs[ispec]) <= min(
                query_right_mzs[iquery], spec_isolation_upper_mzs[ispec]
            ):
                idx_list.append(ispec)
        if len(idx_list) > max_spec_per_query:
            spec_idxes[iquery, :] = idx_list[
                len(idx_list) / 2 - max_spec_per_query // 2 : len(idx_list) / 2
                + max_spec_per_query // 2
                + 1
            ]
        else:
            spec_idxes[iquery, : len(idx_list)] = idx_list
    return spec_idxes
