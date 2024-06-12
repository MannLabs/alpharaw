import numpy as np
import pandas as pd
from alphatims.utils import threadpool
from numba import njit

from alpharaw.feature.centroids import connect_centroids


@threadpool(include_progress_callback=False)
@njit
def path_finder(
    x: np.ndarray,
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
):
    """Extracts path information and writes to path matrix.

    Args:
        x (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        from_idx (np.ndarray): Array containing from indices.
        to_idx (np.ndarray): Array containing to indices.
        forward (np.ndarray): Array to report forward connection.
        backward (np.ndarray): Array to report backward connection.
    """

    fr = from_idx[x]
    to = to_idx[x]

    forward[fr] = to
    backward[to] = fr


@threadpool(include_progress_callback=False)
@njit
def find_path_start(
    x: np.ndarray, forward: np.ndarray, backward: np.ndarray, path_starts: np.ndarray
):
    """Function to find the start of a path.

    Args:
        x (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        forward (np.ndarray):  Array to report forward connection.
        backward (np.ndarray):  Array to report backward connection.
        path_starts (np.ndarray): Array to report path starts.
    """
    if forward[x] > -1 and backward[x] == -1:
        path_starts[x] = 0


@threadpool(include_progress_callback=False)
@njit
def find_path_length(
    x: np.ndarray, path_starts: np.ndarray, forward: np.ndarray, path_cnt: np.ndarray
):
    """Function to extract the length of a path.

    Args:
        x (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        path_starts (np.ndarray): Array that stores the starts of the paths.
        forward (np.ndarray): Array that stores forward information.
        path_cnt (np.ndarray): Reporting array to count the paths.
    """
    ctr = 1
    idx = path_starts[x]
    while forward[idx] > -1:
        ctr += 1
        idx = forward[idx]
    path_cnt[x] = ctr


@threadpool(include_progress_callback=False)
@njit
def fill_path_matrix(
    x: np.ndarray,
    path_start: np.ndarray,
    forwards: np.ndarray,
    out_hill_data: np.ndarray,
    out_hill_ptr: np.ndarray,
):
    """Function to fill the path matrix.

    Args:
        x (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        path_starts (np.ndarray): Array that stores the starts of the paths.
        forwards (np.ndarray): Forward array.
        out_hill_data (np.ndarray): Array containing the indices to hills.
        out_hill_ptr (np.ndarray): Array containing the bounds to out_hill_data.
    """
    path_position = 0
    idx = path_start[x]
    while idx > -1:
        out_hill_data[out_hill_ptr[x] + path_position] = idx
        idx = forwards[idx]
        path_position += 1


def get_hills(
    centroids: np.ndarray,
    from_idx: np.ndarray,
    to_idx: np.ndarray,
    hill_length_min: int = 3,
) -> (np.ndarray, np.ndarray, int):
    """Function to get hills from centroid connections.

    Args:
        centroids (np.ndarray): 1D Array containing the masses of the centroids.
        from_idx (np.ndarray): From index.
        to_idx (np.ndarray): To index.
        hill_length_min (int): Minimum hill length:

    Returns:
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        path_node_cnt (int): Number of elements in this path.
    """

    forward = np.full(centroids.shape[0], -1)
    backward = np.full(centroids.shape[0], -1)
    path_starts = np.full(centroids.shape[0], -1)

    path_finder(range(len(from_idx)), from_idx, to_idx, forward, backward)
    find_path_start(range(len(forward)), forward, backward, path_starts)

    # path_starts will now container the first index of all connected centroids
    path_starts = np.where(path_starts == 0)[0]

    path_node_cnt = np.full(path_starts.shape[0], -1)
    find_path_length(range(len(path_starts)), path_starts, forward, path_node_cnt)

    relavant_path_node = np.where(path_node_cnt >= hill_length_min)[0]
    path_starts = np.take(path_starts, relavant_path_node)
    path_node_cnt = np.take(path_node_cnt, relavant_path_node)
    del relavant_path_node

    # Generate the hill matix indice ptr data
    hill_ptrs = np.empty((path_starts.shape[0] + 1), dtype=np.int32)

    hill_ptrs[0] = 0
    hill_ptrs[1:] = path_node_cnt.cumsum()
    hill_data = np.empty((int(hill_ptrs[-1])), np.int32)

    fill_path_matrix(
        range(len(path_starts)), path_starts, forward, hill_data, hill_ptrs
    )

    del from_idx, to_idx, path_starts, forward, backward
    return hill_ptrs, hill_data, path_node_cnt


def extract_hills(
    query_data: dict, max_gap: int, centroid_tol: float
) -> (np.ndarray, np.ndarray, int, float, float):
    """[summary]

    Args:
        query_data (dict): Data structure containing the query data.
        max_gap (int): Maximum gap when connecting centroids.
        centroid_tol (float): Centroid tolerance.

    Returns:
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        path_node_cnt (int): Number of elements in this path.
        score_median (float): Median score.
        score_std (float): Std deviation of the score.
    """

    indices = np.array(query_data["indices_ms1"])
    mass_data = np.array(query_data["mass_list_ms1"])

    rowwise_peaks = indices[1:] - indices[:-1]
    row_borders = indices[1:]

    from_idx, to_idx, score_median, score_std = connect_centroids(
        rowwise_peaks, row_borders, mass_data, max_gap, centroid_tol
    )

    hill_ptrs, hill_data, path_node_cnt = get_hills(mass_data, from_idx, to_idx)

    del mass_data
    del indices

    return hill_ptrs, hill_data, path_node_cnt, score_median, score_std


@njit
def remove_duplicate_hills(hill_ptrs, hill_data, path_node_cnt):
    """
    Removes hills that share datapoints. Starts from the largest hills.

    """
    taken_points = np.zeros(hill_data.max() + 1)

    c = 0
    current_idx = 0

    hill_ptrs_new = np.zeros_like(hill_ptrs)
    hill_data_new = np.zeros_like(hill_data)

    for p in np.argsort(path_node_cnt)[::-1]:
        s, e = hill_ptrs[p], hill_ptrs[p + 1]

        point_idx = hill_data[s:e]

        hill_pts = taken_points[point_idx]

        if hill_pts.sum() == 0:
            hill_data_new[current_idx : current_idx + len(hill_pts)] = point_idx
            current_idx += len(hill_pts)
            hill_ptrs_new[c + 1] = current_idx
            c += 1

        taken_points[point_idx] += 1

    hill_data_new = hill_data_new[:current_idx]
    hill_ptrs_new = hill_ptrs_new[:c]

    return hill_ptrs_new, hill_data_new


@njit
def fast_minima(y: np.ndarray) -> np.ndarray:
    """Function to calculate the local minimas of an array.

    Args:
        y (np.ndarray): Input array.

    Returns:
        np.ndarray: Array containing minima positions.
    """
    minima = np.zeros(len(y))

    start = 0
    end = len(y)

    for i in range(start + 2, end - 2):
        if (
            ((y[i - 1] > y[i]) & (y[i + 1] > y[i]))
            or ((y[i - 1] > y[i]) & (y[i + 1] == y[i]) & (y[i + 2] > y[i]))
            or ((y[i - 2] > y[i]) & (y[i - 1] == y[i]) & (y[i + 1] > y[i]))
            or (
                (y[i - 2] > y[i])
                & (y[i - 1] == y[i])
                & (y[i + 1] == y[i])
                & (y[i + 2] > y[i])
            )
        ):
            minima[i] = 1

    minima = minima.nonzero()[0]

    return minima


# %% ../nbs/04_feature_finding.ipynb 15
@threadpool(include_progress_callback=False)
@njit
def split(
    k: np.ndarray,
    hill_ptrs: np.ndarray,
    int_data: np.ndarray,
    hill_data: np.ndarray,
    splits: np.ndarray,
    hill_split_level: float,
    window: int,
):
    """Function to split hills.

    Args:
        k (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        hill_data (np.ndarray): Array containing the indices to hills.
        splits (np.ndarray): Array containing splits.
        hill_split_level (float): Split level for hills.
        window (int): Smoothing window.
    """

    start = hill_ptrs[k]
    end = hill_ptrs[k + 1]

    int_idx = hill_data[start:end]  # index to hill data

    int_trace = int_data[int_idx]

    for i in range(len(int_idx)):
        min_index = max(0, i - window)
        max_index = min(len(int_idx), i + window + 1)
        int_trace[i] = np.median(int_trace[min_index:max_index])

    for i in range(len(int_idx)):
        min_index = max(0, i - window)
        max_index = min(len(int_idx), i + window + 1)
        int_trace[i] = np.mean(int_trace[min_index:max_index])

    # minima = (np.diff(np.sign(np.diff(int_trace))) > 0).nonzero()[0] + 1 #This works also but is slower

    minima = fast_minima(int_trace)

    sorted_minima = np.argsort(int_trace[minima])

    minima = minima[sorted_minima]

    for min_ in minima:
        minval = int_trace[min_]

        left_max = max(int_trace[:min_])
        right_max = max(int_trace[min_:])

        min_max = min(left_max, right_max)

        if (minval == 0) or ((min_max / minval) > hill_split_level):
            splits[k] = start + min_
            break  # Split only once per iteration


def split_hills(
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    hill_split_level: float,
    window: int,
) -> np.ndarray:
    """Wrapper function to split hills

    Args:
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        hill_split_level (float): Split level for hills.
        window (int): Smoothing window.

    Returns:
        np.ndarray: Array containing the bounds to the hill_data with splits.
    """

    splits = np.zeros(len(int_data), dtype=np.int32)
    to_check = np.arange(len(hill_ptrs) - 1)

    while len(to_check) > 0:
        split(
            to_check, hill_ptrs, int_data, hill_data, splits, hill_split_level, window
        )
        splitpoints = splits.nonzero()[0]

        to_check = np.zeros(len(hill_ptrs))
        to_check[splitpoints] = 1

        to_check = np.insert(
            to_check, splitpoints + 1, np.ones(len(splitpoints))
        ).nonzero()[0]  # array, index, what
        hill_ptrs = np.insert(
            hill_ptrs, splitpoints + 1, splits[splitpoints]
        )  # array, index, what

        splits = np.zeros(len(hill_ptrs), dtype=np.int32)  # was np np.int32

    return hill_ptrs


# %% ../nbs/04_feature_finding.ipynb 17
@threadpool(include_progress_callback=False)
@njit
def check_large_hills(
    idx: np.ndarray,
    large_peaks: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    to_remove: np.ndarray,
    large_peak: int = 40,
    hill_peak_factor: float = 2,
    window: int = 1,
):
    """Function to check large hills and flag them for removal.

    Args:
        idx (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        large_peaks (np.ndarray): Array containing large peaks.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        to_remove (np.ndarray): Array with indexes to remove.
        large_peak (int, optional): Length criterion when a peak is large. Defaults to 40.
        hill_peak_factor (float, optional): Hill maximum criterion. Defaults to 2.
        window (int, optional): Smoothing window.. Defaults to 1.
    """
    k = large_peaks[idx]

    start = hill_ptrs[k]
    end = hill_ptrs[k + 1]

    int_idx = hill_data[start:end]  # index to hill data

    int_smooth_ = int_data[int_idx]

    for i in range(len(int_idx)):
        min_index = max(0, i - window)
        max_index = min(len(int_idx), i + window + 1)
        int_smooth_[i] = np.median(int_smooth_[min_index:max_index])

    for i in range(len(int_idx)):
        min_index = max(0, i - window)
        max_index = min(len(int_idx), i + window + 1)
        int_smooth_[i] = np.mean(int_smooth_[min_index:max_index])

    int_ = int_data[int_idx]

    max_ = np.max(int_)

    if (max_ / int_smooth_[0] > hill_peak_factor) & (
        max_ / int_smooth_[-1] > hill_peak_factor
    ):
        to_remove[idx] = 0


def filter_hills(
    hill_data: np.ndarray,
    hill_ptrs: np.ndarray,
    int_data: np.ndarray,
    hill_check_large: int = 40,
    window: int = 1,
) -> (np.ndarray, np.ndarray):
    """Filters large hills.

    Args:
        hill_data (np.ndarray): Array containing the indices to hills.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        hill_check_large (int, optional): Length criterion when a hill is considered large.. Defaults to 40.
        window (int, optional): Smoothing window. Defaults to 1.

    Returns:
        np.ndarray: Filtered hill data.
        np.ndarray: Filtered hill points.
    """

    large_peaks = np.where(np.diff(hill_ptrs) >= hill_check_large)[0]

    to_remove = np.ones(len(large_peaks), dtype=np.int32)
    check_large_hills(
        range(len(large_peaks)),
        large_peaks,
        hill_ptrs,
        hill_data,
        int_data,
        to_remove,
        window,
    )

    idx_ = np.ones(len(hill_data), dtype=np.int32)
    keep = np.ones(len(hill_ptrs) - 1, dtype=np.int32)

    to_remove = to_remove.nonzero()[0]

    for _ in to_remove:
        idx_[hill_ptrs[_] : hill_ptrs[_ + 1]] = 0
        keep[_] = 0

    hill_lens = np.diff(hill_ptrs)
    keep_ = hill_lens[keep.nonzero()[0]]

    hill_data_ = hill_data[idx_.nonzero()[0]]
    hill_ptrs_ = np.empty((len(keep_) + 1), dtype=np.int32)
    hill_ptrs_[0] = 0
    hill_ptrs_[1:] = keep_.cumsum()

    return hill_data_, hill_ptrs_


# %% ../nbs/04_feature_finding.ipynb 20
@threadpool
@njit
def hill_stats(
    idx: np.ndarray,
    hill_range: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    mass_data: np.ndarray,
    rt_: np.ndarray,
    rt_idx: np.ndarray,
    stats: np.ndarray,
    hill_nboot_max: int,
    hill_nboot: int,
):
    """Function to calculate hill stats.

    Args:
        idx (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        hill_range (np.ndarray): Hill range.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        mass_data (np.ndarray): Array containing mass data.
        rt_ (np.ndarray): Array with retention time information for each scan.
        rt_idx (np.ndarray): Lookup array to match centroid idx to rt.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        hill_nboot_max (int): Maximum number of bootstrap comparisons.
        hill_nboot (int): Number of bootstrap comparisons
    """
    np.random.seed(42)

    start = hill_ptrs[idx]
    end = hill_ptrs[idx + 1]

    idx_ = hill_data[start:end]

    int_ = int_data[idx_]
    mz_ = mass_data[idx_]

    ms1_int_apex = np.max(int_)
    ms1_int_area = np.abs(np.trapz(int_, rt_[rt_idx[idx_]]))  # Area

    rt_min = rt_[rt_idx[idx_]].min()
    rt_max = rt_[rt_idx[idx_]].max()

    bootsize = hill_nboot_max if len(idx_) > hill_nboot_max else len(idx_)

    averages = np.zeros(hill_nboot)
    average = 0

    for i in range(hill_nboot):
        boot = np.random.choice(len(int_), bootsize, replace=True)
        boot_mz = np.sum(mz_[boot] * int_[boot]) / np.sum(int_[boot])
        averages[i] = boot_mz
        average += boot_mz

    average_mz = average / hill_nboot

    delta = 0
    for i in range(hill_nboot):
        delta += (average_mz - averages[i]) ** 2  # maybe easier?
    delta_m = np.sqrt(delta / (hill_nboot - 1))

    stats[idx, 0] = average_mz
    stats[idx, 1] = delta_m
    stats[idx, 2] = ms1_int_area
    stats[idx, 3] = ms1_int_apex
    stats[idx, 4] = rt_min
    stats[idx, 5] = rt_max


def remove_duplicates(
    stats: np.ndarray, hill_data: np.ndarray, hill_ptrs: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Remove duplicate hills.

    Args:
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        hill_data (np.ndarray): Array containing the indices to hills.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.

    Returns:
        np.ndarray: Filtered hill data.
        np.ndarray: Filtered hill points.
        np.ndarray: Filtered hill stats.
    """

    dups = pd.DataFrame(stats).duplicated()  # all duplicated hills

    idx_ = np.ones(len(hill_data), dtype=np.int32)  # keep all
    keep = np.ones(len(hill_ptrs) - 1, dtype=np.int32)

    for _ in np.arange(len(stats))[dups]:  # duplicates will be assigned zeros
        idx_[hill_ptrs[_] : hill_ptrs[_ + 1]] = 0
        keep[_] = 0

    hill_lens = np.diff(hill_ptrs)
    keep_ = hill_lens[keep.nonzero()[0]]

    hill_data_ = hill_data[idx_.nonzero()[0]]
    hill_ptrs_ = np.empty((len(keep_) + 1), dtype=np.int32)
    hill_ptrs_[0] = 0
    hill_ptrs_[1:] = keep_.cumsum()

    return hill_data_, hill_ptrs_, stats[~dups]


def get_hill_data(
    query_data: dict,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    hill_nboot_max: int = 300,
    hill_nboot: int = 150,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Wrapper function to get the hill data.

    Args:
        query_data (dict): Data structure containing the query data.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        hill_nboot_max (int): Maximum number of bootstrap comparisons.
        hill_nboot (int): Number of bootstrap comparisons

    Returns:
        np.ndarray: Hill stats.
        np.ndarray: Sortindex.
        np.ndarray: Upper index.
        np.ndarray: Scan index.
        np.ndarray: Hill data.
        np.ndarray: Hill points.
    """
    indices_ = np.array(query_data["indices_ms1"])
    rt_ = np.array(query_data["rt_list_ms1"])
    mass_data = np.array(query_data["mass_list_ms1"])
    scan_idx = np.searchsorted(indices_, np.arange(len(mass_data)), side="right") - 1
    int_data = np.array(query_data["int_list_ms1"])

    stats = np.zeros((len(hill_ptrs) - 1, 6))  # mz, delta, rt_min, rt_max, sum_max
    hill_stats(
        range(len(hill_ptrs) - 1),
        np.arange(len(hill_ptrs) - 1),
        hill_ptrs,
        hill_data,
        int_data,
        mass_data,
        rt_,
        scan_idx,
        stats,
        hill_nboot_max,
        hill_nboot,
    )

    # sort the stats
    sortindex = np.argsort(stats[:, 4])  # Sorted by rt_min
    stats = stats[sortindex, :]
    idxs_upper = stats[:, 4].searchsorted(stats[:, 5], side="right")
    sortindex_ = np.arange(len(sortindex))[sortindex]

    return stats, sortindex_, idxs_upper, scan_idx, hill_data, hill_ptrs
