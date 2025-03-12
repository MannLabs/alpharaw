from typing import Callable, Union

import numpy as np
import pandas as pd
from alphatims.utils import threadpool
from numba import njit
from numba.typed import Dict, List

# TODO: Move hardcoded constants
from alpharaw.feature.chem import (
    DELTA_M,
    DELTA_S,
    M_PROTON,
    mass_to_dist,
)


def find_connected_components(edges, min_size=2):
    num_nodes = np.max(edges) + 1
    parent = np.arange(num_nodes)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX

    for edge in edges:
        union(edge[0], edge[1])

    for i in range(num_nodes):
        find(i)

    components = {}
    for node in range(num_nodes):
        root = find(node)
        if root in components:
            components[root].append(node)
        else:
            components[root] = [node]

    pre_isotope_patterns = sorted(
        [
            sorted(component)
            for component in components.values()
            if len(component) >= min_size
        ],
        key=len,
        reverse=True,
    )

    return pre_isotope_patterns


@njit
def check_isotope_pattern(
    mass1: float,
    mass2: float,
    delta_mass1: float,
    delta_mass2: float,
    charge: int,
    iso_mass_range: int = 5,
) -> bool:
    """Check if two masses could belong to the same isotope pattern.

    Args:
        mass1 (float): Mass of the first pattern.
        mass2 (float): Mass of the second pattern.
        delta_mass1 (float): Delta mass of the first pattern.
        delta_mass2 (float): Delta mass of the second pattern.
        charge (int): Charge.
        iso_mass_range (int, optional): Mass range. Defaults to 5.

    Returns:
        bool: Flag to see if pattern belongs to the same pattern.
    """
    delta_mass1 = delta_mass1 * iso_mass_range
    delta_mass2 = delta_mass2 * iso_mass_range

    delta_mass = np.abs(mass1 - mass2)

    left_side = np.abs(delta_mass - DELTA_M / charge)
    right_side = np.sqrt((DELTA_S / charge) ** 2 + delta_mass1**2 + delta_mass2**2)

    return left_side <= right_side


@njit
def correlate(
    scans_: np.ndarray, scans_2: np.ndarray, int_: np.ndarray, int_2: np.ndarray
) -> float:
    """Correlate two scans.

    Args:
        scans_ (np.ndarray): Masses of the first scan.
        scans_2 (np.ndarray): Masses of the second scan.
        int_ (np.ndarray): Intensity of the first scan.
        int_2 (np.ndarray): Intensity of the second scan.

    Returns:
        float: Correlation.
    """

    min_one, max_one = scans_[0], scans_[-1]
    min_two, max_two = scans_2[0], scans_2[-1]

    if (
        min_one + 3 > max_two or min_two + 3 > max_one
    ):  # at least an overlap of 3 elements
        corr = 0
    else:
        min_s = min(min_one, min_two)
        max_s = max(max_one, max_two)

        int_one_scaled = np.zeros(int(max_s - min_s + 1))
        int_two_scaled = np.zeros(int(max_s - min_s + 1))

        int_one_scaled[scans_ - min_s] = int_
        int_two_scaled[scans_2 - min_s] = int_2

        corr = np.sum(int_one_scaled * int_two_scaled) / np.sqrt(
            np.sum(int_one_scaled**2) * np.sum(int_two_scaled**2)
        )

    return corr


# %% ../nbs/04_feature_finding.ipynb 29
@njit
def extract_edge(
    stats: np.ndarray,
    idxs_upper: np.ndarray,
    runner: int,
    max_index: int,
    maximum_offset: float,
    iso_charge_min: int = 1,
    iso_charge_max: int = 6,
    iso_mass_range: int = 5,
) -> list:
    """Extract edges.

    Args:
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        idxs_upper (np.ndarray): Upper index for comparing.
        runner (int): Index.
        max_index (int): Unused.
        maximum_offset (float): Maximum offset when comparing edges.
        iso_charge_min (int, optional): Minimum isotope charge. Defaults to 1.
        iso_charge_max (int, optional): Maximum isotope charge. Defaults to 6.
        iso_mass_range (float, optional): Mass search range. Defaults to 5.

    Returns:
        list: List of edges.
    """
    edges = []

    mass1 = stats[runner, 0]
    delta_mass1 = stats[runner, 1]

    for j in range(runner + 1, idxs_upper[runner]):
        mass2 = stats[j, 0]
        if np.abs(mass2 - mass1) <= maximum_offset:
            delta_mass2 = stats[j, 1]
            for charge in range(iso_charge_min, iso_charge_max + 1):
                if check_isotope_pattern(
                    mass1, mass2, delta_mass1, delta_mass2, charge, iso_mass_range
                ):
                    edges.append((runner, j))
                    break

    return edges


@threadpool(include_progress_callback=False)
@njit
def edge_correlation(
    idx: np.ndarray,
    to_keep: np.ndarray,
    sortindex_: np.ndarray,
    pre_edges: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    cc_cutoff: float,
):
    """Correlates two edges and flag them it they should be kept.

    Args:
        idx (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        to_keep (np.ndarray): Array with indices which edges should be kept.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        pre_edges (np.ndarray): Array with pre edges.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        cc_cutoff (float): Cutoff value for what is considered correlating.
    """
    edge = pre_edges[idx, :]

    y = sortindex_[edge[0]]
    start = hill_ptrs[y]
    end = hill_ptrs[y + 1]
    idx_ = hill_data[start:end]
    int_ = int_data[idx_]
    scans_ = scan_idx[idx_]

    con = sortindex_[edge[1]]
    start = hill_ptrs[con]
    end = hill_ptrs[con + 1]
    idx_2 = hill_data[start:end]
    int_2 = int_data[idx_2]
    scans_2 = scan_idx[idx_2]

    if correlate(scans_, scans_2, int_, int_2) > cc_cutoff:
        to_keep[idx] = 1


def get_pre_isotope_patterns(
    stats: np.ndarray,
    idxs_upper: np.ndarray,
    sortindex_: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    maximum_offset: float,
    iso_charge_min: int = 1,
    iso_charge_max: int = 6,
    iso_mass_range: float = 5,
    cc_cutoff: float = 0.6,
) -> list:
    """Function to extract pre isotope patterns.

    Args:
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        idxs_upper (np.ndarray): Upper index for comparison.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        maximum_offset (float): Maximum offset when matching.
        iso_charge_min (int, optional): Minimum isotope charge. Defaults to 1.
        iso_charge_max (int, optional): Maximum isotope charge. Defaults to 6.
        iso_mass_range (float, optional): Mass search range. Defaults to 5.
        cc_cutoff (float, optional): Correlation cutoff. Defaults to 0.6.

    Returns:
        list: List of pre isotope patterns.
    """
    pre_edges = []

    # Step 1
    for runner in range(len(stats)):
        pre_edges.extend(
            extract_edge(
                stats,
                idxs_upper,
                runner,
                idxs_upper[runner],
                maximum_offset,
                iso_charge_min,
                iso_charge_max,
                iso_mass_range,
            )
        )

    to_keep = np.zeros(len(pre_edges), dtype="int")
    pre_edges = np.array(pre_edges)
    edge_correlation(
        range(len(to_keep)),
        to_keep,
        sortindex_,
        pre_edges,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        cc_cutoff,
    )
    edges = pre_edges[to_keep.nonzero()]

    pre_isotope_patterns = find_connected_components(edges)

    return pre_isotope_patterns


@njit
def check_isotope_pattern_directed(
    mass1: float,
    mass2: float,
    delta_mass1: float,
    delta_mass2: float,
    charge: int,
    index: int,
    iso_mass_range: float,
) -> bool:
    """Check if two masses could belong to the same isotope pattern.

    Args:
        mass1 (float): Mass of the first pattern.
        mass2 (float): Mass of the second pattern.
        delta_mass1 (float): Delta mass of the first pattern.
        delta_mass2 (float): Delta mass of the second pattern.
        charge (int): Charge.
        index (int): Index (unused).
        iso_mass_range (float): Isotope mass ranges.
    Returns:
        bool: Flag if two isotope patterns belong together.
    """
    delta_mass1 = delta_mass1 * iso_mass_range
    delta_mass2 = delta_mass2 * iso_mass_range

    left_side = np.abs(mass1 - mass2 - index * DELTA_M / charge)
    right_side = np.sqrt((DELTA_S / charge) ** 2 + delta_mass1**2 + delta_mass2**2)

    return left_side <= right_side


@njit
def grow(
    trail: List,
    seed: int,
    direction: int,
    relative_pos: int,
    index: int,
    stats: np.ndarray,
    pattern: np.ndarray,
    charge: int,
    iso_mass_range: float,
    sortindex_: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    cc_cutoff: float,
) -> List:
    """Grows isotope pattern based on a seed and direction.

    Args:
        trail (List): List of hills belonging to a pattern.
        seed (int): Seed position.
        direction (int): Direction in which to grow the trail
        relative_pos (int): Relative position.
        index (int): Index.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        pattern (np.ndarray): Isotope pattern.
        charge (int): Charge.
        iso_mass_range (float): Mass range for checking isotope patterns.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        cc_cutoff (float): Cutoff value for what is considered correlating.

    Returns:
        List: List of hills belonging to a pattern.
    """
    x = pattern[seed]  # This is the seed
    mass1 = stats[x, 0]
    delta_mass1 = stats[x, 1]

    k = sortindex_[x]
    start = hill_ptrs[k]
    end = hill_ptrs[k + 1]
    idx_ = hill_data[start:end]
    int_ = int_data[idx_]
    scans_ = scan_idx[idx_]

    growing = True

    while growing:
        if direction == 1:
            if seed + relative_pos == len(pattern):
                growing = False
                break
        else:
            if seed + relative_pos < 0:
                growing = False
                break

        y = pattern[seed + relative_pos]  # This is a reference peak

        ll = sortindex_[y]

        mass2 = stats[y, 0]
        delta_mass2 = stats[y, 1]

        start = hill_ptrs[ll]
        end = hill_ptrs[ll + 1]
        idx_ = hill_data[start:end]
        int_2 = int_data[idx_]
        scans_2 = scan_idx[idx_]

        if correlate(
            scans_, scans_2, int_, int_2
        ) > cc_cutoff and check_isotope_pattern_directed(
            mass1,
            mass2,
            delta_mass1,
            delta_mass2,
            charge,
            -direction * index,
            iso_mass_range,
        ):
            if direction == 1:
                trail.append(y)
            else:
                trail.insert(0, y)
            index += 1  # Greedy matching: Only one edge for a specific distance, will not affect the following matches

        delta_mass = np.abs(mass1 - mass2)

        if (
            delta_mass > (DELTA_M + DELTA_S) * index
        ):  # the pattern is sorted so there is a maximum to look back
            break

        relative_pos += direction

    return trail


@njit
def grow_trail(
    seed: int,
    pattern: np.ndarray,
    stats: np.ndarray,
    charge: int,
    iso_mass_range: float,
    sortindex_: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    cc_cutoff: float,
) -> List:
    """Wrapper to grow an isotope pattern to the left and right side.

    Args:
        seed (int): Seed position.
        pattern (np.ndarray): Isotope pattern.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        charge (int): Charge.
        iso_mass_range (float): Mass range for checking isotope patterns.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        cc_cutoff (float): Cutoff value for what is considered correlating.

    Returns:
        List: Isotope pattern.
    """
    x = pattern[seed]
    trail = List()
    trail.append(x)
    trail = grow(
        trail,
        seed,
        -1,
        -1,
        1,
        stats,
        pattern,
        charge,
        iso_mass_range,
        sortindex_,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        cc_cutoff,
    )
    trail = grow(
        trail,
        seed,
        1,
        1,
        1,
        stats,
        pattern,
        charge,
        iso_mass_range,
        sortindex_,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        cc_cutoff,
    )

    return trail


@njit
def get_trails(
    seed: int,
    pattern: np.ndarray,
    stats: np.ndarray,
    charge_range: List,
    iso_mass_range: float,
    sortindex_: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    cc_cutoff: float,
) -> List:
    """Wrapper to extract trails for a given charge range.

    Args:
        seed (int): Seed index.
        pattern (np.ndarray): Pre isotope pattern.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        charge_range (List): Charge range.
        iso_mass_range (float): Mass range for checking isotope patterns.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        cc_cutoff (float): Cutoff value for what is considered correlating.

    Returns:
        List: Trail of consistent hills.
    """
    trails = []
    for charge in charge_range:
        trail = grow_trail(
            seed,
            pattern,
            stats,
            charge,
            iso_mass_range,
            sortindex_,
            hill_ptrs,
            hill_data,
            int_data,
            scan_idx,
            cc_cutoff,
        )

        trails.append(trail)

    return trails


# %% ../nbs/04_feature_finding.ipynb 33
def plot_pattern(
    pattern: np.ndarray,
    sorted_hills: np.ndarray,
    centroids: np.ndarray,
):
    """Helper function to plot a pattern.

    Args:
        pattern (np.ndarray): Pre isotope pattern.
        sorted_hills (np.ndarray): Hills, sorted.
        centroids (np.ndarray): 1D Array containing the masses of the centroids.
        hill_data (np.ndarray): Array containing the indices to hills.
    """
    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    centroid_dtype = [("mz", float), ("int", float), ("scan_no", int), ("rt", float)]

    for entry in pattern:
        hill = sorted_hills[entry]
        hill_data = np.array(
            [centroids[_[0]][_[1]] for _ in hill], dtype=centroid_dtype
        )

        ax1.plot(hill_data["rt"], hill_data["int"])
        ax2.scatter(hill_data["rt"], hill_data["mz"], s=hill_data["int"] / 5e5)

    ax1.set_title("Pattern")
    ax1.set_xlabel("RT (min)")
    ax1.set_ylabel("Intensity")

    ax2.set_xlabel("RT (min)")
    ax2.set_ylabel("m/z")

    plt.show()


@njit
def get_minpos(y: np.ndarray, iso_split_level: float) -> List:
    """Function to get a list of minima in a trace.
    A minimum is returned if the ratio of lower of the surrounding maxima to the minimum is larger than the splitting factor.

    Args:
        y (np.ndarray): Input array.
        iso_split_level (float): Isotope split level.

    Returns:
        List: List with min positions.
    """
    minima = get_local_minima(y)
    minima_list = List()

    for minpos in minima:
        minval = y[minpos]

        left_max = (y[:minpos]).max()
        right_max = (y[minpos:]).max()

        minimum_max = min(left_max, right_max)

        if minimum_max / minval >= iso_split_level:
            minima_list.append(minpos)

    return minima_list


@njit
def get_local_minima(y: np.ndarray) -> List:
    """Function to return all local minima of a array

    Args:
        y (np.ndarray): Input array.

    Returns:
        List: List with indices to minima.
    """
    minima = List()
    for i in range(1, len(y) - 1):
        if is_local_minima(y, i):
            minima.append(i)
    return minima


@njit
def is_local_minima(y: np.ndarray, i: int) -> bool:
    """Check if position is a local minima.

    Args:
        y (np.ndarray): Input array.
        i (int): Position to check.

    Returns:
        bool: Flag if position is minima or not.
    """
    return (y[i - 1] > y[i]) & (y[i + 1] > y[i])


@njit
def truncate(
    array: np.ndarray,
    intensity_profile: np.ndarray,
    seedpos: int,
    iso_split_level: float,
) -> np.ndarray:
    """Function to truncate an intensity profile around its seedposition.

    Args:
        array (np.ndarray):  Input array.
        intensity_profile (np.ndarray): Intensities for the input array.
        seedpos (int): Seedposition.
        iso_split_level (float): Split level.

    Returns:
        np.ndarray: Truncated array.
    """
    minima = int_list_to_array(get_minpos(intensity_profile, iso_split_level))

    if len(minima) > 0:
        left_minima = minima[minima < seedpos]
        right_minima = minima[minima > seedpos]

        # If the minimum is smaller than the seed
        minpos = left_minima[-1] if len(left_minima) > 0 else 0

        maxpos = right_minima[0] if len(right_minima) > 0 else len(array)

        array = array[minpos : maxpos + 1]

    return array


@njit
def check_averagine(
    stats: np.ndarray,
    pattern: np.ndarray,
    charge: int,
    averagine_aa: Dict,
    isotopes: Dict,
) -> float:
    """Function to compare a pattern to an averagine model.

    Args:
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        pattern (np.ndarray): Isotope pattern.
        charge (int): Charge.
        averagine_aa (Dict): Dict containing averagine masses.
        isotopes (Dict): Dict containing isotopes.

    Returns:
        float: Averagine correlation.
    """
    masses, intensity = pattern_to_mz(stats, pattern, charge)

    spec_one = np.floor(masses).astype(np.int64)
    int_one = intensity

    spec_two, int_two = mass_to_dist(
        np.min(masses), averagine_aa, isotopes
    )  # maybe change to no rounded version

    spec_two = np.floor(spec_two).astype(np.int64)

    return cosine_averagine(int_one, int_two, spec_one, spec_two)


@njit
def pattern_to_mz(
    stats: np.ndarray, pattern: np.ndarray, charge: int
) -> (np.ndarray, np.ndarray):
    """Function to calculate masses and intensities from pattern for a given charge.

    Args:
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        pattern (np.ndarray): Isotope pattern.
        charge (int): Charge of the pattern.

    Returns:
        np.ndarray: masses
        np.ndarray: intensity
    """

    mzs = np.zeros(len(pattern))
    ints = np.zeros(len(pattern))

    for i in range(len(pattern)):
        entry = pattern[i]
        mzs[i] = mz_to_mass(stats[entry, 0], charge)
        ints[i] = stats[entry, 2]

    sortindex = np.argsort(mzs)

    masses = mzs[sortindex]
    intensity = ints[sortindex]

    return masses, intensity


@njit
def cosine_averagine(
    int_one: np.ndarray, int_two: np.ndarray, spec_one: np.ndarray, spec_two: np.ndarray
) -> float:
    """Calculate the cosine correlation of two hills.

    Args:
        int_one (np.ndarray): Intensity of the first hill.
        int_two (np.ndarray): Intensity of the second hill.
        spec_one (np.ndarray): Scan numbers of the first hill.
        spec_two (np.ndarray): Scan numbers of the second hill.

    Returns:
        float: Cosine
    """

    min_one, max_one = spec_one[0], spec_one[-1]
    min_two, max_two = spec_two[0], spec_two[-1]

    min_s = np.min(np.array([min_one, min_two]))
    max_s = np.max(np.array([max_one, max_two]))

    int_one_scaled = np.zeros(int(max_s - min_s + 1))
    int_two_scaled = np.zeros(int(max_s - min_s + 1))

    int_one_scaled[spec_one - min_s] = int_one
    int_two_scaled[spec_two - min_s] = int_two

    corr = np.sum(int_one_scaled * int_two_scaled) / np.sqrt(
        np.sum(int_one_scaled**2) * np.sum(int_two_scaled**2)
    )

    return corr


@njit
def int_list_to_array(numba_list: List) -> np.ndarray:
    """Numba compatbilte function to convert a numba list with integers to a numpy array

    Args:
        numba_list (List): Input numba-typed List.

    Returns:
        np.ndarray: Output numpy array.
    """
    array = np.zeros(len(numba_list), dtype=np.int64)

    for i in range(len(array)):
        array[i] = numba_list[i]

    return array


@njit
def mz_to_mass(mz: float, charge: int) -> float:
    """Function to calculate the mass from a mz value.

    Args:
        mz (float): M/z
        charge (int): Charge.

    Raises:
        NotImplementedError: When a negative charge is used.

    Returns:
        float: mass
    """
    if charge < 0:
        raise NotImplementedError("Negative Charges not implemented.")

    mass = mz * charge - charge * M_PROTON

    return mass


@njit
def isolate_isotope_pattern(
    pre_pattern: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    stats: np.ndarray,
    sortindex_: np.ndarray,
    iso_mass_range: float,
    charge_range: List,
    averagine_aa: Dict,
    isotopes: Dict,
    iso_n_seeds: int,
    cc_cutoff: float,
    iso_split_level: float,
) -> (np.ndarray, int):
    """Isolate isotope patterns.

    Args:
        pre_pattern (np.ndarray): Pre isotope pattern.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        iso_mass_range (float): Mass range for checking isotope patterns.
        charge_range (List): Charge range.
        averagine_aa (Dict): Dict containing averagine masses.
        isotopes (Dict): Dict containing isotopes.
        iso_n_seeds (int): Number of seeds.
        cc_cutoff (float): Cutoff value for what is considered correlating.
        iso_split_level (float): Split level when isotopes are split.

    Returns:
        np.ndarray: Array with the best pattern.
        int: Charge of the best pattern.
    """
    longest_trace = 0
    champion_trace = None
    champion_charge = 0
    champion_intensity = 0

    # Sort patterns by mass

    sortindex = np.argsort(stats[pre_pattern][:, 0])  # intensity
    sorted_pattern = pre_pattern[sortindex]
    massindex = np.argsort(stats[sorted_pattern][:, 2])[::-1][:iso_n_seeds]

    # Use all the elements in the pre_pattern as seed

    for seed in massindex:  # Loop through all seeds
        seed_global = sorted_pattern[seed]

        trails = get_trails(
            seed,
            sorted_pattern,
            stats,
            charge_range,
            iso_mass_range,
            sortindex_,
            hill_ptrs,
            hill_data,
            int_data,
            scan_idx,
            cc_cutoff,
        )

        for index, trail in enumerate(trails):
            if (
                len(trail) >= longest_trace
            ):  # Needs to be longer than the current champion
                arr = int_list_to_array(trail)
                intensity_profile = stats[arr][:, 2]
                seedpos = np.nonzero(arr == seed_global)[0][0]

                # truncate around the seed...
                arr = truncate(arr, intensity_profile, seedpos, iso_split_level)
                intensity_profile = stats[arr][:, 2]

                # Remove lower masses:
                # Take the index of the maximum and remove all masses on the left side
                if charge_range[index] * stats[seed_global, 0] < 1000:
                    maxpos = np.argmax(intensity_profile)
                    arr = arr[maxpos:]
                    intensity_profile = stats[arr][:, 2]

                if (len(arr) > longest_trace) | (
                    (len(arr) == longest_trace)
                    & (intensity_profile.sum() > champion_intensity)
                ):
                    # Averagine check
                    cc = check_averagine(
                        stats, arr, charge_range[index], averagine_aa, isotopes
                    )
                    if cc > 0.6:
                        # Update the champion
                        champion_trace = arr
                        champion_charge = charge_range[index]
                        longest_trace = len(arr)
                        champion_intensity = intensity_profile.sum()

    return champion_trace, champion_charge


def get_isotope_patterns(
    pre_isotope_patterns: list,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    scan_idx: np.ndarray,
    stats: np.ndarray,
    sortindex_: np.ndarray,
    averagine_aa: Dict,
    isotopes: Dict,
    iso_charge_min: int = 1,
    iso_charge_max: int = 6,
    iso_mass_range: float = 5,
    iso_n_seeds: int = 100,
    cc_cutoff: float = 0.6,
    iso_split_level: float = 1.3,
    callback: Union[Callable, None] = None,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Wrapper function to iterate over pre_isotope_patterns.

    Args:
        pre_isotope_patterns (list): List of pre-isotope patterns.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        scan_idx (np.ndarray): Array containing the scan index for a centroid.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        averagine_aa (Dict): Dict containing averagine masses.
        isotopes (Dict): Dict containing isotopes.
        iso_charge_min (int, optional): Minimum isotope charge. Defaults to 1.
        iso_charge_max (int, optional): Maximum isotope charge. Defaults to 6.
        iso_mass_range (float, optional): Mass search range. Defaults to 5.
        iso_n_seeds (int, optional): Number of isotope seeds. Defaults to 100.
        cc_cutoff (float, optional): Cuttoff for correlation.. Defaults to 0.6.
        iso_split_level (float, optional): Isotope split level.. Defaults to 1.3.
        callback (Union[Callable, None], optional): Callback function for progress. Defaults to None.
    Returns:
        list: List of isotope patterns.
        np.ndarray: Iso idx.
        np.ndarray: Array containing isotope charges.
    """

    isotope_patterns = []
    isotope_charges = []

    charge_range = List()

    for i in range(iso_charge_min, iso_charge_max + 1):
        charge_range.append(i)

    isotope_patterns = []
    isotope_charges = []

    for idx, pre_pattern in enumerate(pre_isotope_patterns):
        extract = True
        while extract:
            isotope_pattern, isotope_charge = isolate_isotope_pattern(
                np.array(pre_pattern),
                hill_ptrs,
                hill_data,
                int_data,
                scan_idx,
                stats,
                sortindex_,
                iso_mass_range,
                charge_range,
                averagine_aa,
                isotopes,
                iso_n_seeds,
                cc_cutoff,
                iso_split_level,
            )
            length = 0 if isotope_pattern is None else len(isotope_pattern)

            if length > 1:
                isotope_charges.append(isotope_charge)
                isotope_patterns.append(isotope_pattern)

                pre_pattern = [_ for _ in pre_pattern if _ not in isotope_pattern]

                if len(pre_pattern) <= 1:
                    extract = False
            else:
                extract = False

        if callback:
            callback((idx + 1) / len(pre_isotope_patterns))

    iso_patterns = np.zeros(sum([len(_) for _ in isotope_patterns]), dtype=np.int64)

    iso_idx = np.zeros(len(isotope_patterns) + 1, dtype="int")

    start = 0
    for idx, _ in enumerate(isotope_patterns):
        iso_patterns[start : start + len(_)] = _
        start += len(_)
        iso_idx[idx + 1] = start

    return iso_patterns, iso_idx, np.array(isotope_charges)


@threadpool
@njit
def report_(
    idx: np.ndarray,
    isotope_charges: list,
    isotope_patterns: list,
    iso_idx: np.ndarray,
    stats: np.ndarray,
    sortindex_: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
    int_data: np.ndarray,
    rt_: np.ndarray,
    rt_idx: np.ndarray,
    results: np.ndarray,
    lookup_idx: np.ndarray,
):
    """Function to extract summary statstics from a list of isotope patterns and charges.

    MS1 feature intensity estimation. For each isotope envelope we interpolate the signal over the retention time
    range. All isotope enevelopes are summed up together to estimate the peak sahpe

    Lastly, we report three estimates for the intensity:

    - ms1_int_sum_apex: The intensity at the peak of the summed signal.
    - ms1_int_sum_area: The area of the summed signal
    - ms1_int_max_apex: The intensity at the peak of the most intense isotope envelope
    - ms1_int_max_area: The area of the the most intense isotope envelope

    Args:
        idx (np.ndarray): Input index. Note that we are using the performance function so this is a range.
        isotope_patterns (list): List containing isotope patterns (indices to hills).
        isotope_charges (list): List with charges assigned to the isotope patterns.
        iso_idx (np.ndarray): Index to isotope pattern.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.
        int_data (np.ndarray): Array containing the intensity to each centroid.
        rt_ (np.ndarray): Array with retention time information for each scan.
        rt_idx (np.ndarray): Lookup array to match centroid idx to rt.
        results (np.ndarray): Recordarray with isotope pattern summary statistics.
        lookup_idx (np.ndarray): Lookup array for each centroid.
    """
    pattern = isotope_patterns[iso_idx[idx] : iso_idx[idx + 1]]
    isotope_data = stats[pattern]

    mz = np.min(isotope_data[:, 0])
    mz_std = np.mean(isotope_data[:, 1])
    charge = isotope_charges[idx]
    mass = mz_to_mass(mz, charge)
    int_max_idx = np.argmax(isotope_data[:, 2])
    mz_most_abundant = isotope_data[:, 0][int_max_idx]

    rt_min_ = min(isotope_data[:, 4])
    rt_max_ = max(isotope_data[:, 5])

    rt_range = np.linspace(
        rt_min_, rt_max_, 100
    )  # TODO this is a fixed value - is there an optimum?

    trace_sum = np.zeros_like(rt_range)

    most_intense_pattern = -np.inf

    for i, k in enumerate(pattern):
        x = sortindex_[k]

        start = hill_ptrs[x]
        end = hill_ptrs[x + 1]
        idx_ = hill_data[start:end]
        int_ = int_data[idx_]
        rts = rt_[rt_idx[idx_]]

        lookup_idx[idx_, 0] = idx
        lookup_idx[idx_, 1] = i

        interpolation = np.interp(rt_range, rts, int_)

        # Filter

        interpolation[: (rt_range < rts[0]).sum()] = 0

        right_cut = (rt_range > rts[-1]).sum()
        if right_cut > 0:
            interpolation[-right_cut:] = 0

        trace_sum += interpolation

        if int_.sum() > most_intense_pattern:
            most_intense_pattern = int_.sum()
            ms1_int_max_apex = int_.max()
            ms1_int_max_area = np.trapz(int_, rts)

    rt_apex_idx = trace_sum.argmax()
    rt_apex = rt_range[rt_apex_idx]

    trace = trace_sum
    half_max = trace.max() / 2

    if rt_apex_idx == 0:
        left_apex = 0
    else:
        left_apex = np.abs(trace[:rt_apex_idx] - half_max).argmin()
    right_apex = np.abs(trace[rt_apex_idx:] - half_max).argmin() + rt_apex_idx

    ms1_int_sum_apex = trace_sum[rt_apex_idx]

    fwhm = rt_range[right_apex] - rt_range[left_apex]

    n_isotopes = len(pattern)

    rt_cutoff = 0.95  # 5%
    if rt_apex_idx == 0:
        rt_min_idx = 0
    else:
        rt_min_idx = np.abs(
            trace[:rt_apex_idx] - trace.max() * (1 - rt_cutoff)
        ).argmin()
    rt_max_idx = (
        np.abs(trace[rt_apex_idx:] - trace.max() * (1 - rt_cutoff)).argmin()
        + rt_apex_idx
    )

    # plt.xlabel('rt')
    # plt.ylabel('int')
    # plt.show()
    # plt.plot(rt_range, trace_sum)

    # plt.plot([rt_range[left_apex], rt_range[right_apex]], [(trace[left_apex] + trace[right_apex])/2]*2, 'k:')

    # plt.plot(rt_range[rt_apex_idx], trace[rt_apex_idx], 'k*')
    # plt.plot(rt_range[rt_min_idx], trace[rt_min_idx], 'k*')
    # plt.plot(rt_range[rt_max_idx], trace[rt_max_idx], 'k*')

    # plt.show()

    rt_start = rt_range[rt_min_idx]
    rt_end = rt_range[rt_max_idx]

    ms1_int_sum_area = np.trapz(
        trace_sum[rt_min_idx:rt_max_idx], rt_range[rt_min_idx:rt_max_idx]
    )

    results[idx, :] = np.array(
        [
            mz,
            mz_std,
            mz_most_abundant,
            charge,
            rt_start,
            rt_apex,
            rt_end,
            fwhm,
            n_isotopes,
            mass,
            ms1_int_sum_apex,
            ms1_int_sum_area,
            ms1_int_max_apex,
            ms1_int_max_area,
        ]
    )


def feature_finder_report(
    query_data: dict,
    isotope_patterns: list,
    isotope_charges: list,
    iso_idx: np.ndarray,
    stats: np.ndarray,
    sortindex_: np.ndarray,
    hill_ptrs: np.ndarray,
    hill_data: np.ndarray,
) -> pd.DataFrame:
    """Creates a report dataframe with summary statistics of the found isotope patterns.

    Args:
        query_data (dict): Data structure containing the query data.
        isotope_patterns (list): List containing isotope patterns (indices to hills).
        isotope_charges (list): List with charges assigned to the isotope patterns.
        iso_idx (np.ndarray): Index to the isotope pattern.
        stats (np.ndarray): Stats array that contains summary statistics of hills.
        sortindex_ (np.ndarray): Sortindex to access the hills from stats.
        hill_ptrs (np.ndarray): Array containing the bounds to the hill_data.
        hill_data (np.ndarray): Array containing the indices to hills.

    Returns:
        pd.DataFrame: DataFrame with isotope pattern summary statistics.
    """
    rt_ = np.array(query_data["rt_list_ms1"])
    indices_ = np.array(query_data["indices_ms1"])
    mass_data = np.array(query_data["mass_list_ms1"])
    rt_idx = np.searchsorted(indices_, np.arange(len(mass_data)), side="right") - 1

    lookup_idx = np.zeros((len(mass_data), 2), dtype=np.int32) - 1

    int_data = np.array(query_data["int_list_ms1"])

    results = np.zeros((len(isotope_charges), 14))

    report_(
        range(len(isotope_charges)),
        isotope_charges,
        isotope_patterns,
        iso_idx,
        stats,
        sortindex_,
        hill_ptrs,
        hill_data,
        int_data,
        rt_,
        rt_idx,
        results,
        lookup_idx,
    )

    df = pd.DataFrame(
        results,
        columns=[
            "mz",
            "mz_std",
            "mz_most_abundant",
            "charge",
            "rt_start",
            "rt_apex",
            "rt_end",
            "fwhm",
            "n_isotopes",
            "mass",
            "ms1_int_sum_apex",
            "ms1_int_sum_area",
            "ms1_int_max_apex",
            "ms1_int_max_area",
        ],
    )

    df.sort_values(["rt_start", "mz"])

    return df, lookup_idx


def get_stats(isotope_patterns, iso_idx, stats):
    columns = ["mz_average", "delta_m", "int_sum", "int_area", "rt_min", "rt_max"]

    stats_idx = np.zeros(iso_idx[-1], dtype=np.int64)
    stats_map = np.zeros(iso_idx[-1], dtype=np.int64)

    start_ = 0
    end_ = 0

    for idx in range(len(iso_idx) - 1):
        k = isotope_patterns[iso_idx[idx] : iso_idx[idx + 1]]
        end_ += len(k)
        stats_idx[start_:end_] = k
        stats_map[start_:end_] = idx
        start_ = end_

    k = pd.DataFrame(stats[stats_idx], columns=columns)

    k["feature_id"] = stats_map

    return k
