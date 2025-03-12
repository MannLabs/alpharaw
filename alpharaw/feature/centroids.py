import numpy as np
from alphatims.utils import threadpool
from numba import njit


@threadpool
@njit
def connect_centroids_unidirection(
    x: np.ndarray,
    row_borders: np.ndarray,
    connections: np.ndarray,
    scores: np.ndarray,
    centroids: np.ndarray,
    max_gap: int,
    centroid_tol: float,
):
    """Connect centroids.

    Args:
        x (np.ndarray): Index to datapoint. Note that this using the threadpool, so one passes an ndarray.
        row_borders (np.ndarray): Row borders of the centroids array.
        connections (np.ndarray): Connections matrix to store the connections
        scores (np.ndarray):  Score matrix to store the connections
        centroids (np.ndarray): 1D Array containing the masses of the centroids data.
        max_gap (int): Maximum gap when connecting centroids.
        centroid_tol (float): Centroid tolerance.
    """
    for gap in range(max_gap + 1):
        y = x + gap + 1
        if y >= row_borders.shape[0]:
            return

        start_index_f = 0
        if x > 0:
            start_index_f = row_borders[x - 1]

        centroids_1 = centroids[start_index_f : row_borders[x]]
        start_index_b = row_borders[y - 1]
        centroids_2 = centroids[start_index_b : row_borders[y]]

        i = 0
        j = 0

        while (i < len(centroids_1)) & (j < len(centroids_2)):
            mz1, mz2 = centroids_1[i], centroids_2[j]
            diff = mz1 - mz2
            mz_sum = mz1 + mz2
            delta = 2 * 1e6 * abs(diff) / mz_sum

            if delta < centroid_tol and scores[x, i, gap] > delta:
                scores[x, i, gap] = delta
                connections[x, i, gap] = (connections.shape[1] * y) + j

            if diff > 0:
                j += 1
            else:
                i += 1


def find_centroid_connections(
    rowwise_peaks: np.ndarray,
    row_borders: np.ndarray,
    centroids: np.ndarray,
    max_gap: int,
    centroid_tol: float,
):
    """Wrapper function to call connect_centroids_unidirection

    Args:
        rowwise_peaks (np.ndarray): Length of centroids with respect to the row borders.
        row_borders (np.ndarray): Row borders of the centroids array.
        centroids (np.ndarray): Array containing the centroids data.
        max_gap (int): Maximum gap when connecting centroids.
        centroid_tol (float): Centroid tolerance.
    """

    max_centroids = int(np.max(rowwise_peaks))
    spectra_cnt = len(row_borders) - 1

    connections = np.full((spectra_cnt, max_centroids, max_gap + 1), -1, dtype=np.int32)
    score = np.full((spectra_cnt, max_centroids, max_gap + 1), np.inf)

    connect_centroids_unidirection(
        range(len(row_borders)),
        row_borders,
        connections,
        score,
        centroids,
        max_gap,
        centroid_tol,
    )

    score = score[np.where(score < np.inf)]

    score_median = np.median(score)
    score_std = np.std(score)

    del score, max_centroids, spectra_cnt

    c_shape = connections.shape
    from_r, from_c, from_g = np.where(connections >= 0)
    to_r = connections[from_r, from_c, from_g] // c_shape[1]
    to_c = connections[from_r, from_c, from_g] - to_r * c_shape[1]

    del connections, from_g

    return from_r, from_c, to_r, to_c, score_median, score_std


@threadpool(include_progress_callback=False)
@njit
def convert_connections_to_array(
    x: np.ndarray,
    from_r: np.ndarray,
    from_c: np.ndarray,
    to_r: np.ndarray,
    to_c: np.ndarray,
    row_borders: np.ndarray,
    out_from_idx: np.ndarray,
    out_to_idx: np.ndarray,
):
    """Convert integer indices of a matrix to coordinates.

    Args:
        x (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        from_r (np.ndarray): From array with row coordinates.
        from_c (np.ndarray): From array with column coordinates.
        to_r (np.ndarray): To array with row coordinates.
        to_c (np.ndarray): To array with column coordinates.
        row_borders (np.ndarray): Row borders (for indexing).
        out_from_idx (np.ndarray): Reporting array: 1D index from.
        out_to_idx (np.ndarray): Reporting array: 1D index to.
    """
    row = from_r[x]
    col = from_c[x]
    start_index_f = 0
    if row > 0:
        start_index_f = row_borders[row - 1]
    out_from_idx[x] = start_index_f + col

    row = to_r[x]
    col = to_c[x]
    start_index_f = 0
    if row > 0:
        start_index_f = row_borders[row - 1]
    out_to_idx[x] = start_index_f + col


@threadpool(include_progress_callback=False)
@njit
def eliminate_overarching_vertex(
    x: np.ndarray, from_idx: np.ndarray, to_idx: np.ndarray
):
    """Eliminate overacrhing vertex.

    Args:
        x (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        from_idx (np.ndarray): From index.
        to_idx (np.ndarray): To index.
    """
    if x == 0:
        return

    if from_idx[x - 1] == from_idx[x]:
        to_idx[x] = -1


def connect_centroids(
    rowwise_peaks: np.ndarray,
    row_borders: np.ndarray,
    centroids: np.ndarray,
    max_gap: int,
    centroid_tol: float,
) -> (np.ndarray, np.ndarray, float, float):
    """Function to connect centroids.

    Args:
        rowwise_peaks (np.ndarray): Indexes for centroids.
        row_borders (np.ndarray): Row borders (for indexing).
        centroids (np.ndarray): Centroid data.
        max_gap: Maximum gap.
        centroid_tol: Centroid tol for matching centroids.
    Returns:
        np.ndarray: From index.
        np.ndarray: To index.
        float: Median score.
        float: Std deviation of the score.
    """

    from_r, from_c, to_r, to_c, score_median, score_std = find_centroid_connections(
        rowwise_peaks, row_borders, centroids, max_gap, centroid_tol
    )

    from_idx = np.zeros(len(from_r), np.int32)
    to_idx = np.zeros(len(from_r), np.int32)

    convert_connections_to_array(
        range(len(from_r)), from_r, from_c, to_r, to_c, row_borders, from_idx, to_idx
    )

    eliminate_overarching_vertex(range(len(from_idx)), from_idx, to_idx)

    relavent_idx = np.where(to_idx >= 0)
    from_idx = np.take(from_idx, relavent_idx)[0]
    to_idx = np.take(to_idx, relavent_idx)[0]

    del from_r, from_c, to_r, to_c, relavent_idx
    return from_idx, to_idx, score_median, score_std
