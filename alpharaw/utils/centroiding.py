# Modified from AlphaPept

import numpy as np
from numba import njit
from numba.typed import List

@njit
def naive_centroid(
    peak_mzs, peak_intens, centroiding_ppm=20.0,
):
    mz_list = []
    inten_list = []
    start,stop = 0,1
    centroiding_peak_tols = 2*peak_mzs*centroiding_ppm*1e-6
    while start < len(peak_mzs):
        stop = _find_sister_peaks(
            peak_mzs, centroiding_peak_tols, start
        )
        mz_list.append(
            np.average(peak_mzs[start:stop], 
            weights=peak_intens[start:stop]
        ))
        inten_list.append(np.sum(peak_intens[start:stop]))
        start = stop
    return (
        np.array(mz_list, dtype=peak_mzs.dtype),
        np.array(inten_list, dtype=peak_intens.dtype)
    )

@njit
def _find_sister_peaks(
    peak_mzs, centroiding_peak_tols, start
):
    stop = start + 1
    for i in range(start+1, len(peak_mzs)):
        if peak_mzs[i]-peak_mzs[start] <= centroiding_peak_tols[start]:
            stop = i+1
        else:
            stop = i
            break
    return stop