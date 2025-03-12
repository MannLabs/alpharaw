from typing import Tuple

import numba
import numpy as np


@numba.njit
def match_batch_spec(
    spec_idxes: np.ndarray,
    peak_mzs: np.ndarray,
    peak_intens: np.ndarray,
    peak_start_idxes: np.ndarray,
    peak_stop_idxes: np.ndarray,
    query_mzs: np.ndarray,
    query_mz_tols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matched mzs and intensities for query m/z values against the given batch spectra.

    Parameters
    ----------
    spec_idxes : np.ndarray
        The batch spectra, given as spectrum indexes.
    peak_mzs : np.ndarray
        The peak m/z values in the whole raw data.
    peak_intens : np.ndarray
        The peak intensities in the whole raw data.
    peak_start_idxes : np.ndarray
        The batch spectra, given as the start indexes in peak m/z and intensities.
    peak_stop_idxes : np.ndarray
        The batch spectra, given as the stop indexes in peak m/z and intensities.
    query_mzs : np.ndarray
        The query m/z values, these can be from fragments of a precursor.
    query_mz_tols : np.ndarray
        The query tolerance values of query_mzs.

    Returns
    -------
    Tuple[ndarray, ndarray]
        ndarray with shape (spectrum num, query num): matched m/z values. 0.0 if not matched.
        ndarray with shape (spectrum num, query num): matched intensity values. 0.0 if not matched.
    """
    matched_mzs = np.zeros((len(spec_idxes), len(query_mzs)), dtype=peak_mzs.dtype)
    matched_intens = np.zeros(
        (len(spec_idxes), len(query_mzs)), dtype=peak_intens.dtype
    )

    for i_spec, spec_idx in enumerate(spec_idxes):
        cur_peak_mzs = peak_mzs[peak_start_idxes[spec_idx] : peak_stop_idxes[spec_idx]]
        cur_peak_intens = peak_intens[
            peak_start_idxes[spec_idx] : peak_stop_idxes[spec_idx]
        ]

        idxes = np.searchsorted(cur_peak_mzs, query_mzs)

        for i, idx in enumerate(idxes):
            if idx == 0:
                if abs(query_mzs[i] - cur_peak_mzs[idx]) <= query_mz_tols[i]:
                    matched_mzs[i_spec, i] = cur_peak_mzs[idx]
                    matched_intens[i_spec, i] = cur_peak_intens[idx]
            elif idx == len(cur_peak_mzs):
                if abs(query_mzs[i] - cur_peak_mzs[idx - 1]) <= query_mz_tols[i]:
                    matched_mzs[i_spec, i] = cur_peak_mzs[idx - 1]
                    matched_intens[i_spec, i] = cur_peak_intens[idx - 1]
            else:
                left_dist = abs(query_mzs[i] - cur_peak_mzs[idx - 1])
                right_dist = abs(query_mzs[i] - cur_peak_mzs[idx])
                if right_dist <= query_mz_tols[i] and left_dist <= query_mz_tols[i]:
                    matched_mzs[i_spec, i] = (
                        cur_peak_mzs[idx] * cur_peak_intens[idx]
                        + cur_peak_mzs[idx - 1] * cur_peak_intens[idx - 1]
                    ) / (cur_peak_intens[idx] + cur_peak_intens[idx - 1])
                    matched_intens[i_spec, i] = (
                        cur_peak_intens[idx] + cur_peak_intens[idx - 1]
                    )
                elif left_dist <= query_mz_tols[i]:
                    matched_mzs[i_spec, i] = cur_peak_mzs[idx - 1]
                    matched_intens[i_spec, i] = cur_peak_intens[idx - 1]
                elif right_dist <= query_mz_tols[i]:
                    matched_mzs[i_spec, i] = cur_peak_mzs[idx]
                    matched_intens[i_spec, i] = cur_peak_intens[idx]
    return matched_mzs, matched_intens


@numba.njit
def match_closest_peaks(
    spec_mzs: np.ndarray,
    spec_intens: np.ndarray,
    query_mzs: np.ndarray,
    query_mz_tols: np.ndarray,
) -> np.ndarray:
    """Matching query mz values against sorted MS2/spec m/z values,
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
    query_left_mzs = mzs - query_mz_tols
    query_right_mzs = mzs + query_mz_tols
    idxes = np.searchsorted(spec_mzs, query_left_mzs)
    ret_indices = np.empty_like(mzs, dtype=np.int32)
    for i, idx in enumerate(idxes):
        min_merr = 1000000
        closest_idx = -1
        for _idx in range(idx, len(spec_mzs)):
            if spec_mzs[_idx] > query_right_mzs[i]:
                break
            elif spec_mzs[_idx] < query_left_mzs[i]:
                continue
            elif min_merr > abs(spec_mzs[_idx] - mzs[i]):
                min_merr = abs(spec_mzs[_idx] - mzs[i])
                closest_idx = _idx
        ret_indices[i] = closest_idx
    return ret_indices.reshape(query_mzs.shape)


@numba.njit
def match_highest_peaks(
    spec_mzs: np.ndarray,
    spec_intens: np.ndarray,
    query_mzs: np.ndarray,
    query_mz_tols: np.ndarray,
) -> np.ndarray:
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
    query_left_mzs = mzs - query_mz_tols
    query_right_mzs = mzs + query_mz_tols
    idxes = np.searchsorted(spec_mzs, query_left_mzs)
    ret_indices = np.empty_like(mzs, dtype=np.int32)
    for i, idx in enumerate(idxes):
        highest = 0
        highest_idx = -1
        for _idx in range(idx, len(spec_mzs)):
            if spec_mzs[_idx] > query_right_mzs[i]:
                break
            elif spec_mzs[_idx] < query_left_mzs[i]:
                continue
            elif highest < spec_intens[_idx]:
                highest = spec_intens[_idx]
                highest_idx = _idx
        ret_indices[i] = highest_idx
    return ret_indices.reshape(query_mzs.shape)


@numba.njit
def match_profile_peaks(
    spec_mzs: np.ndarray,
    spec_intens: np.ndarray,
    query_mzs: np.ndarray,
    query_mz_tols: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
    query_left_mzs = mzs - query_mz_tols
    query_right_mzs = mzs + query_mz_tols
    idxes = np.searchsorted(spec_mzs, query_left_mzs)
    match_start_indices = np.full_like(mzs, -1, dtype=np.int32)
    match_stop_indices = np.full_like(mzs, -1, dtype=np.int32)
    for i, idx in enumerate(idxes):
        for first_idx in range(idx, len(spec_mzs)):
            if spec_mzs[first_idx] < query_left_mzs[i]:
                continue
            elif spec_mzs[first_idx] > query_right_mzs[i]:
                break
            else:
                match_start_indices[i] = first_idx
                for last_idx in np.arange(first_idx + 1, len(spec_mzs)):
                    if spec_mzs[last_idx] > query_right_mzs[i]:
                        match_stop_indices[i] = last_idx
                        break
                if match_stop_indices[i] == -1:
                    match_stop_indices[i] = len(spec_mzs)
                break
    return (
        match_start_indices.reshape(query_mzs.shape),
        match_stop_indices.reshape(query_mzs.shape),
    )
