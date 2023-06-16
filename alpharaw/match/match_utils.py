import numpy as np
import numba
import pandas as pd
import tqdm
import os

from typing import Tuple

@numba.njit
def match_closest_peaks(
    spec_mzs:np.ndarray,
    spec_intens:np.ndarray,
    query_mzs:np.ndarray,
    query_mz_tols:np.ndarray,
)->np.ndarray:
    """Matching query mz values against sorted MS2/spec masses, 
    only closest (minimal abs mass error) peaks are returned.

    Parameters
    ----------
    spec_mzs : np.ndarray
        sorted 1-D mz array of the spectrum

    spec_intens : np.ndarray
        1-D intensity array of the spectrum. Not used here

    query_mzs : np.ndarray
        query n-D mz array

    query_mz_tols : np.ndarray
        query n-D mz tolerance array, same shape as query_mzs

    Returns
    -------
    np.ndarray
        Matched indices of spec_mzs, -1 means no peaks were matched.
        Same shape as query_mzs 
    """
    mzs = query_mzs.reshape(-1)
    query_mz_tols = query_mz_tols.reshape(-1)
    query_left_mzs = mzs-query_mz_tols
    query_right_mzs = mzs+query_mz_tols
    idxes = np.searchsorted(spec_mzs, query_left_mzs)
    ret_indices = np.empty_like(mzs, dtype=np.int32)
    for i,idx in enumerate(idxes):
        min_merr = 1000000
        closest_idx = -1
        for _idx in range(idx, len(spec_mzs)):
            if spec_mzs[_idx]>query_right_mzs[i]:
                break
            elif spec_mzs[_idx]<query_left_mzs[i]:
                continue
            elif min_merr > abs(spec_mzs[_idx]-mzs[i]):
                min_merr = abs(spec_mzs[_idx]-mzs[i])
                closest_idx = _idx
        ret_indices[i] = closest_idx
    return ret_indices.reshape(query_mzs.shape)


@numba.njit
def match_highest_peaks(
    spec_mzs:np.ndarray,
    spec_intens:np.ndarray,
    query_mzs:np.ndarray,
    query_mz_tols:np.ndarray,
)->np.ndarray:
    """Matching query mz values against sorted MS2/spec masses, 
    only highest peaks are returned.

    Parameters
    ----------
    spec_mzs : np.ndarray
        sorted 1-D mz array of the spectrum

    spec_intens : np.ndarray
        1-D intensity array of the spectrum. Not used here

    query_mzs : np.ndarray
        query n-D mz array

    query_mz_tols : np.ndarray
        query n-D mz tolerance array, same shape as query_mzs

    Returns
    -------
    np.ndarray
        Matched indices of spec_mzs, -1 means no peaks were matched.
        Same shape as query_mzs 
    """
    mzs = query_mzs.reshape(-1)
    query_mz_tols = query_mz_tols.reshape(-1)
    query_left_mzs = mzs-query_mz_tols
    query_right_mzs = mzs+query_mz_tols
    idxes = np.searchsorted(spec_mzs, query_left_mzs)
    ret_indices = np.empty_like(mzs, dtype=np.int32)
    for i,idx in enumerate(idxes):
        highest = 0
        highest_idx = -1
        for _idx in range(idx, len(spec_mzs)):
            if spec_mzs[_idx]>query_right_mzs[i]:
                break
            elif spec_mzs[_idx]<query_left_mzs[i]:
                continue
            elif highest < spec_intens[_idx]:
                highest = spec_intens[_idx]
                highest_idx = _idx
        ret_indices[i] = highest_idx
    return ret_indices.reshape(query_mzs.shape)

@numba.njit
def match_profile_peaks(
    spec_mzs:np.ndarray,
    spec_intens:np.ndarray,
    query_mzs:np.ndarray,
    query_mz_tols:np.ndarray,
)->Tuple[np.ndarray, np.ndarray]:
    """
    Matching query mz values against sorted MS2/spec profile masses,
    both left- and right-most m/z values are returned.

    Parameters
    ----------
    spec_mzs : np.ndarray
        sorted 1-D mz array of the spectrum

    spec_intens : np.ndarray
        1-D intensity array of the spectrum. Not used here

    query_mzs : np.ndarray
        query n-D mz array

    query_mz_tols : np.ndarray
        query n-D mz tolerance array, same shape as query_mzs

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        np.ndarray: matched first (left-most) indices, the shape is the same as query_mzs.
        -1 means no peaks are matched for the query mz.

        np.ndarray: matched last (right-most) indices, the shape is the same as query_mzs.
        -1 means no peaks are matched for the query mz.
    """
    mzs = query_mzs.reshape(-1)
    query_mz_tols = query_mz_tols.reshape(-1)
    query_left_mzs = mzs-query_mz_tols
    query_right_mzs = mzs+query_mz_tols
    idxes = np.searchsorted(spec_mzs, query_left_mzs)
    first_indices = np.full_like(
        mzs, -1, dtype=np.int32
    )
    last_indices = np.full_like(
        mzs, -1, dtype=np.int32
    )
    for i,idx in enumerate(idxes):
        for first_idx in range(idx, len(spec_mzs)):
            if spec_mzs[first_idx]<query_left_mzs[i]:
                continue
            elif spec_mzs[first_idx]>query_right_mzs[i]:
                break
            else:
                first_indices[i] = first_idx
                if first_idx == len(spec_mzs)-1:
                    last_indices[i] = first_idx
                else:
                    for last_idx in range(first_idx+1, len(spec_mzs)):
                        if spec_mzs[last_idx]>query_right_mzs[i]:
                            break
                    last_indices[i] = last_idx-1
                    break
    return (
        first_indices.reshape(query_mzs.shape), 
        last_indices.reshape(query_mzs.shape),
    )

