# Modified from AlphaPept

import numpy as np
from numba import njit
from numba.typed import List

@njit
def get_neighbors_start_stop(
    peak_mzs:np.ndarray,
    start_list, stop_list,
    start_idx:int,
    mz_tol:float,
    mz_diffs:np.ndarray,
):
    for i in range(start_idx, len(mz_diffs)):
        if mz_diffs[i] > mz_tol:
            start_list.append(start_idx)
            stop_list.append(i+1)
            return i + 1
    start_list.append(start_idx)
    stop_list.append(len(mz_diffs))
    return len(mz_diffs)+1

@njit
def get_peaks(
    mz_array:np.ndarray,
    int_array: np.ndarray,
    mz_tol:float,
) -> list:
    start_list = List()
    end_list = List()
    gradient = np.diff(int_array)
    mz_diffs = np.diff(mz_array)
    start = 0
    while start < len(mz_array):
        start = get_a_centroid_group(
            start_list, end_list, start, mz_tol, gradient, mz_diffs
        )
    return start_list, end_list

@njit
def get_a_centroid(
    start, end,
    mz_array: np.ndarray,
    int_array: np.ndarray,
) -> tuple:
    inten = np.sum(int_array[start:end+1])

    mz_cent = 0

    for i in range(start, end+1):
        mz_cent += int_array[i]*mz_array[i]/100.0
    mz_cent = mz_cent*100.0/inten

    return mz_cent, inten

@njit
def centroid_peaks(
    mz_array: np.ndarray,
    int_array: np.ndarray,
    mz_tol:float = 0.006,
) -> tuple:
    """Estimate centroids and intensities from profile data.
    Args:
        mz_array (np.ndarray): An array with mz values.
        int_array (np.ndarray): An array with intensity values.
    Returns:
        tuple: A tuple of the form (mz_array_centroided, int_array_centroided)
    """
    starts, ends = get_peaks(mz_array, int_array, mz_tol)

    mz_array_centroided = np.zeros(len(starts))
    int_array_centroided = np.zeros(len(starts))
    mz_starts = np.zeros(len(starts))
    mz_ends = np.zeros(len(starts))


    for i,(start, end) in enumerate(zip(starts, ends)):
        mz_, int_ = get_a_centroid(start,end, mz_array, int_array)
        mz_array_centroided[i] = mz_
        int_array_centroided[i] = int_
        mz_starts[i] = mz_array[start]
        mz_ends[i] = mz_array[end]

    return (
        mz_array_centroided, 
        int_array_centroided,
        mz_starts, mz_ends
    )