"""JIT-compiled filtering functions for multi-dimensional TimsTOF data slicing."""

import numpy as np

from alpharaw.utils.pjit import njit, pjit


@njit
def valid_quad_mz_values(
    low_mz_value: float,
    high_mz_value: float,
    quad_slices: np.ndarray,
) -> bool:
    """Check if the low and high quad mz values are included in the slices.

    NOTE: Just a part of the quad range needs to overlap with a part
    of a single slice.

    Parameters
    ----------
    low_mz_value : float
        The lower mz value of the current quad selection.
    high_mz_value : float
        The upper mz value of the current quad selection.
    quad_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).

    Returns
    -------
    : bool
        True if some part of the quad overlaps with some part of some slice.
        False if there is no overlap in the range.
    """
    slice_index = np.searchsorted(quad_slices[:, 0].ravel(), high_mz_value, "right")
    if slice_index == 0:
        return False
    if low_mz_value <= quad_slices[slice_index - 1, 1]:
        return True
    return False


@njit
def valid_precursor_index(precursor_index: int, precursor_slices: np.ndarray) -> bool:
    """Check if a precursor index is included in the slices.

    Parameters
    ----------
    precursor_index : int
        The precursor index to validate.
    precursor_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).

    Returns
    -------
    : bool
        True if the precursor index is present in any of the slices.
        False otherwise.
    """
    slice_index = np.searchsorted(
        precursor_slices[:, 0].ravel(), precursor_index, side="right"
    )
    if slice_index == 0:
        return False
    return precursor_index in range(
        precursor_slices[slice_index - 1, 0],
        precursor_slices[slice_index - 1, 1],
        precursor_slices[slice_index - 1, 2],
    )


@njit
def filter_indices(
    frame_slices: np.ndarray,
    scan_slices: np.ndarray,
    precursor_slices: np.ndarray,
    tof_slices: np.ndarray,
    quad_slices: np.ndarray,
    intensity_slices: np.ndarray,
    frame_max_index: int,
    scan_max_index: int,
    push_indptr: np.ndarray,
    precursor_indices: np.ndarray,
    quad_mz_values: np.ndarray,
    quad_indptr: np.ndarray,
    tof_indices: np.ndarray,
    intensities: np.ndarray,
):
    """Filter raw indices by slices from all dimensions.

    Parameters
    ----------
    frame_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(frame_slices[:, :2].ravel()) >= 0) = True).
    scan_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(scan_slices[:, :2].ravel()) >= 0) = True).
    precursor_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).
    tof_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
    quad_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).
    intensity_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(intensity_slices.ravel()) >= 0) = True).
    frame_max_index : int
        The maximum frame index of a TimsTOF object.
    scan_max_index : int
        The maximum scan index of a TimsTOF object.
    push_indptr : np.int64[:]
        The self.push_indptr array of a TimsTOF object.
    precursor_indices : np.int64[:]
        The self.precursor_indices array of a TimsTOF object.
    quad_mz_values : np.float64[:, 2]
        The self.quad_mz_values array of a TimsTOF object.
    quad_indptr : np.int64[:]
        The self.quad_indptr array of a TimsTOF object.
    tof_indices : np.uint32[:]
        The self.tof_indices array of a TimsTOF object.
    intensities : np.uint16[:]
        The self.intensity_values array of a TimsTOF object.

    Returns
    -------
    : np.int64[:]
        The raw indices that satisfy all the slices.
    """
    result = []
    quad_index = -1
    new_quad_index = -1
    quad_end = -1
    is_valid_quad_index = True
    starts = push_indptr[:-1].reshape(frame_max_index, scan_max_index)
    ends = push_indptr[1:].reshape(frame_max_index, scan_max_index)
    for frame_start, frame_stop, frame_step in frame_slices:
        for frame_start_slice, frame_end_slice in zip(
            starts[slice(frame_start, frame_stop, frame_step)],
            ends[slice(frame_start, frame_stop, frame_step)],
        ):
            for scan_start, scan_stop, scan_step in scan_slices:
                for sparse_start, sparse_end in zip(
                    frame_start_slice[slice(scan_start, scan_stop, scan_step)],
                    frame_end_slice[slice(scan_start, scan_stop, scan_step)],
                ):
                    if sparse_start == sparse_end:
                        continue
                    while quad_end < sparse_end:
                        new_quad_index += 1
                        quad_end = quad_indptr[new_quad_index + 1]
                    if quad_index != new_quad_index:
                        quad_index = new_quad_index
                        if not valid_quad_mz_values(
                            quad_mz_values[quad_index, 0],
                            quad_mz_values[quad_index, 1],
                            quad_slices,
                        ) or not valid_precursor_index(
                            precursor_indices[quad_index],
                            precursor_slices,
                        ):
                            is_valid_quad_index = False
                        else:
                            is_valid_quad_index = True
                    if not is_valid_quad_index:
                        continue
                    idx = sparse_start
                    for tof_start, tof_stop, tof_step in tof_slices:
                        idx += np.searchsorted(tof_indices[idx:sparse_end], tof_start)
                        tof_value = tof_indices[idx]
                        while (tof_value < tof_stop) and (idx < sparse_end):
                            if tof_value in range(tof_start, tof_stop, tof_step):
                                intensity = intensities[idx]
                                for low_intensity, high_intensity in intensity_slices:
                                    if low_intensity <= intensity:
                                        if intensity <= high_intensity:
                                            result.append(idx)
                                            break
                            idx += 1
                            tof_value = tof_indices[idx]
    return np.array(result)


# Overhead of using more than 1 threads is actually slower
@pjit(thread_count=1, include_progress_callback=False)
def add_intensity_to_bin(
    query_index: int,
    intensities: np.ndarray,
    parsed_indices: np.ndarray,
    intensity_bins: np.ndarray,
) -> None:
    """Add the intensity of a query to the appropriate bin.

    IMPORTANT NOTE: This function is decorated with `pjit`.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

    Parameters
    ----------
    query_index : int
        The query whose intensity needs to be binned
        The first argument is thus expected to be provided as an iterable
        containing ints instead of a single int.
    intensities : np.float64[:]
        An array with intensities that need to be binned.
    parsed_indices : np.int64[:], np.int64[:, :]
        Description of parameter `parsed_indices`.
    intensity_bins : np.float64[:]
        A buffer with intensity bins to which the current query will be added.
    """
    intensity = intensities[query_index]
    if len(parsed_indices) == 1:
        intensity_bins[parsed_indices[0][query_index]] += intensity
    elif len(parsed_indices) == 2:
        intensity_bins[
            parsed_indices[0][query_index], parsed_indices[1][query_index]
        ] += intensity
