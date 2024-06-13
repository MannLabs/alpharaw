from typing import Tuple

import numpy as np
from numba import njit


@njit
def naive_centroid(
    peak_mzs: np.ndarray,
    peak_intens: np.ndarray,
    centroiding_ppm: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A naive centroiding algorithm.

    Parameters
    ----------
    peak_mzs : np.ndarray
        peak m/z values to centroid.
    peak_intens : np.ndarray
        peak intensities to centroid.
    centroiding_ppm : float, optional
        The centroiding ppm, by default 20.0

    Returns
    -------
    Tuple
        ndarray: peak m/z array
        ndarray: peak intensity array
    """
    mz_list = []
    inten_list = []
    start, stop = 0, 1
    centroiding_peak_tols = 2 * peak_mzs * centroiding_ppm * 1e-6
    while start < len(peak_mzs):
        stop = _find_sister_peaks(peak_mzs, centroiding_peak_tols, start)
        mz_list.append(
            np.average(peak_mzs[start:stop], weights=peak_intens[start:stop])
        )
        inten_list.append(np.sum(peak_intens[start:stop]))
        start = stop
    return (
        np.array(mz_list, dtype=peak_mzs.dtype),
        np.array(inten_list, dtype=peak_intens.dtype),
    )


@njit
def _find_sister_peaks(
    peak_mzs: np.ndarray, centroiding_peak_tols: np.ndarray, start: int
):
    """
    Find sister peak stop idx for the given start idx.
    Internal function.
    """
    stop = start + 1
    for i in range(start + 1, len(peak_mzs)):
        if peak_mzs[i] - peak_mzs[start] <= centroiding_peak_tols[start]:
            stop = i + 1
        else:
            stop = i
            break
    return stop
