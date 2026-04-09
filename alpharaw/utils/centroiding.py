from typing import Tuple

import numpy as np
from numba import njit


# =============================================================================
# Local Maxima Centroiding Algorithm
# =============================================================================


@njit
def _find_local_maxima(intensities: np.ndarray) -> np.ndarray:
    """
    Find indices of local maxima in an intensity array.

    A local maximum is a point strictly greater than both neighbors.
    For edge points, only one neighbor is checked.

    Parameters
    ----------
    intensities : np.ndarray
        Intensity values (1D array).

    Returns
    -------
    np.ndarray
        Array of indices where local maxima occur.
    """
    n = len(intensities)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)

    # Pre-allocate (upper bound = n)
    maxima = np.empty(n, dtype=np.int64)
    count = 0

    # First point
    if n >= 2 and intensities[0] > intensities[1]:
        maxima[count] = 0
        count += 1

    # Interior points
    for i in range(1, n - 1):
        if intensities[i] > intensities[i - 1] and intensities[i] >= intensities[i + 1]:
            maxima[count] = i
            count += 1

    # Last point
    if n >= 2 and intensities[n - 1] > intensities[n - 2]:
        maxima[count] = n - 1
        count += 1

    return maxima[:count]


@njit
def _find_valley_boundaries(
    intensities: np.ndarray,
    maxima_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find left and right valley boundaries for each local maximum.

    For each peak (local maximum), the boundary extends to the nearest
    local minimum (valley) between it and the adjacent peak, or to the
    edge of the array.

    Parameters
    ----------
    intensities : np.ndarray
        Intensity values.
    maxima_indices : np.ndarray
        Indices of local maxima (sorted).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (left_boundaries, right_boundaries) as index arrays.
        left_boundaries[i] is the first index of peak i's region.
        right_boundaries[i] is one past the last index (exclusive).
    """
    n_peaks = len(maxima_indices)
    n = len(intensities)

    left_bounds = np.empty(n_peaks, dtype=np.int64)
    right_bounds = np.empty(n_peaks, dtype=np.int64)

    for i in range(n_peaks):
        # Left boundary: go left from the maximum to find the valley
        if i == 0:
            # First peak: extend to the start of the array
            left_bounds[i] = 0
        else:
            # Find minimum between previous maximum and this one
            prev_max = maxima_indices[i - 1]
            curr_max = maxima_indices[i]
            min_idx = prev_max
            min_val = intensities[prev_max]
            for j in range(prev_max + 1, curr_max):
                if intensities[j] < min_val:
                    min_val = intensities[j]
                    min_idx = j
            left_bounds[i] = min_idx

        # Right boundary: go right from the maximum to find the valley
        if i == n_peaks - 1:
            # Last peak: extend to the end of the array
            right_bounds[i] = n
        else:
            # Find minimum between this maximum and the next one
            curr_max = maxima_indices[i]
            next_max = maxima_indices[i + 1]
            min_idx = curr_max
            min_val = intensities[curr_max]
            for j in range(curr_max + 1, next_max):
                if intensities[j] < min_val:
                    min_val = intensities[j]
                    min_idx = j
            right_bounds[i] = min_idx + 1  # exclusive

    return left_bounds, right_bounds


@njit
def _estimate_noise_mad(intensities: np.ndarray) -> np.float64:
    """
    Estimate noise level using the Median Absolute Deviation (MAD).

    Uses the lower half of intensities (below median) to avoid
    signal peaks biasing the noise estimate.

    The MAD is scaled by 1.4826 to be consistent with the standard
    deviation for Gaussian noise.

    Parameters
    ----------
    intensities : np.ndarray
        Intensity values.

    Returns
    -------
    np.float64
        Estimated noise standard deviation.
    """
    if len(intensities) == 0:
        return np.float64(0.0)

    median_val = np.median(intensities)
    abs_deviations = np.abs(intensities - median_val)
    mad = np.median(abs_deviations)

    return np.float64(1.4826 * mad)


@njit
def _split_segments_by_gaps(
    peak_mzs: np.ndarray,
    gap_ppm: float = 50.0,
) -> np.ndarray:
    """
    Find indices where large m/z gaps occur, indicating separate peak groups.

    A gap is detected when the spacing between consecutive points exceeds
    gap_ppm (relative to the current m/z). This handles the common case
    in profile data where peaks are separated by regions with no data points.

    Parameters
    ----------
    peak_mzs : np.ndarray
        Sorted m/z values.
    gap_ppm : float
        Minimum gap in ppm to consider as a peak boundary.

    Returns
    -------
    np.ndarray
        Array of gap indices (the index AFTER the gap). These are the
        start indices of new segments.
    """
    n = len(peak_mzs)
    if n <= 1:
        return np.empty(0, dtype=np.int64)

    gaps = np.empty(n, dtype=np.int64)
    count = 0

    for i in range(1, n):
        spacing = peak_mzs[i] - peak_mzs[i - 1]
        threshold = peak_mzs[i - 1] * gap_ppm * 1e-6
        if spacing > threshold:
            gaps[count] = i
            count += 1

    return gaps[:count]


@njit
def centroid_local_maxima(
    peak_mzs: np.ndarray,
    peak_intensities: np.ndarray,
    snr_threshold: float = 1.0,
    gap_ppm: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centroid profile data using local maxima detection with valley boundaries.

    Algorithm:
    1. Split the spectrum into segments at large m/z gaps.
    2. Within each segment, find local maxima and valley boundaries.
    3. Optionally filter peaks by signal-to-noise ratio (MAD-based noise).
    4. Compute intensity-weighted average m/z within each peak's boundaries.

    This approach is adaptive to peak width (no PPM parameter needed for
    peak grouping) and avoids the fragmentation problem of the naive
    PPM-window algorithm.

    Parameters
    ----------
    peak_mzs : np.ndarray
        Profile m/z values (sorted, 1D array).
    peak_intensities : np.ndarray
        Profile intensity values (1D array, same length as peak_mzs).
    snr_threshold : float, optional
        Minimum signal-to-noise ratio for a peak to be kept.
        Set to 0 to disable filtering. Default is 1.0.
    gap_ppm : float, optional
        Minimum m/z gap (in ppm) to split into separate peak groups.
        Points separated by more than this are always in different peaks.
        Default is 50.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (centroid_mzs, centroid_intensities) arrays.
    """
    n = len(peak_mzs)
    if n == 0:
        return (
            np.empty(0, dtype=peak_mzs.dtype),
            np.empty(0, dtype=peak_intensities.dtype),
        )

    if n == 1:
        return (
            peak_mzs.copy(),
            peak_intensities.copy(),
        )

    # Step 1: Find m/z gaps that split the data into segments
    gap_indices = _split_segments_by_gaps(peak_mzs, gap_ppm)

    # Build segment boundaries: [0, gap1, gap2, ..., n]
    n_segments = len(gap_indices) + 1
    seg_starts = np.empty(n_segments, dtype=np.int64)
    seg_ends = np.empty(n_segments, dtype=np.int64)

    seg_starts[0] = 0
    for i in range(len(gap_indices)):
        seg_ends[i] = gap_indices[i]
        seg_starts[i + 1] = gap_indices[i]
    seg_ends[n_segments - 1] = n

    # Step 2: SNR noise estimate (global)
    # Skip SNR filtering for very small spectra where MAD is unreliable
    if snr_threshold > 0 and n >= 10:
        noise_estimate = _estimate_noise_mad(peak_intensities)
    else:
        noise_estimate = np.float64(0.0)

    # Pre-allocate output (upper bound: one centroid per segment)
    max_peaks = n  # Upper bound
    mz_out = np.empty(max_peaks, dtype=peak_mzs.dtype)
    int_out = np.empty(max_peaks, dtype=peak_intensities.dtype)
    out_count = 0

    # Step 3: Process each segment independently
    for seg_idx in range(n_segments):
        s_start = seg_starts[seg_idx]
        s_end = seg_ends[seg_idx]
        seg_len = s_end - s_start

        if seg_len == 0:
            continue

        seg_mz = peak_mzs[s_start:s_end]
        seg_int = peak_intensities[s_start:s_end]

        # Find local maxima within this segment
        maxima = _find_local_maxima(seg_int)

        if len(maxima) == 0:
            # No local maxima: treat entire segment as one peak
            total_int = np.float64(0.0)
            weighted_mz = np.float64(0.0)
            for j in range(seg_len):
                total_int += seg_int[j]
                weighted_mz += seg_mz[j] * seg_int[j]

            if total_int > 0:
                # SNR filter
                max_int = seg_int[0]
                for j in range(1, seg_len):
                    if seg_int[j] > max_int:
                        max_int = seg_int[j]

                if snr_threshold > 0 and noise_estimate > 0:
                    if max_int < snr_threshold * noise_estimate:
                        continue

                mz_out[out_count] = weighted_mz / total_int
                int_out[out_count] = total_int
                out_count += 1
            continue

        # Find valley boundaries between local maxima
        left_bounds, right_bounds = _find_valley_boundaries(seg_int, maxima)

        # Compute centroids for each peak
        for i in range(len(maxima)):
            apex_intensity = seg_int[maxima[i]]

            # SNR filter
            if snr_threshold > 0 and noise_estimate > 0:
                if apex_intensity < snr_threshold * noise_estimate:
                    continue

            left = left_bounds[i]
            right = right_bounds[i]

            total_int = np.float64(0.0)
            weighted_mz = np.float64(0.0)

            for j in range(left, right):
                total_int += seg_int[j]
                weighted_mz += seg_mz[j] * seg_int[j]

            if total_int > 0:
                mz_out[out_count] = weighted_mz / total_int
                int_out[out_count] = total_int
                out_count += 1

    return mz_out[:out_count], int_out[:out_count]


# =============================================================================
# Naive Centroiding Algorithm (original)
# =============================================================================


@njit
def naive_centroid(
    peak_mzs: np.ndarray,
    peak_intensities: np.ndarray,
    centroiding_ppm: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A naive centroiding algorithm.

    Parameters
    ----------
    peak_mzs : np.ndarray
        peak m/z values to centroid.
    peak_intensities : np.ndarray
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
            np.average(peak_mzs[start:stop], weights=peak_intensities[start:stop])
        )
        inten_list.append(np.sum(peak_intensities[start:stop]))
        start = stop
    return (
        np.array(mz_list, dtype=peak_mzs.dtype),
        np.array(inten_list, dtype=peak_intensities.dtype),
    )


@njit
def _find_sister_peaks(
    peak_mzs: np.ndarray, centroiding_peak_tols: np.ndarray, start: int
):
    """
    Find sister peak stop idx for the given start idx.
    Sister peaks refers to peaks from the same ion in profile mode.
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
