"""A module to perform centroiding of AlphaTims data."""

import alphatims.utils
import numpy as np


@alphatims.utils.njit(nogil=True)
def get_connections_within_cycle(
    scan_tolerance: int,
    scan_max_index: int,
    dia_mz_cycle: np.ndarray,
    exclude_self: bool = False,
    multiple_frames: bool = False,
    ms1: bool = True,
    ms2: bool = False,
) -> tuple:
    """Detemine how indiviudla pushes in a cycle are connected.

    Parameters
    ----------
    scan_tolerance : int
        Maximum scan distance for two pushes to be connected
    scan_max_index : int
        The maximum scan index (dia_data.scan_max_index).
    dia_mz_cycle : np.ndarray
        An np.float64[:, 2] array with upper and lower quadrupole boundaries
        per push of a cycle.
    exclude_self : bool
        Excluded connections between equal push indices
        (the default is False).
    multiple_frames : bool
        Connect scans between different frames a cycle
        (the default is False).
    ms1 : bool
        Allow connections between MS1 pushes
        (the default is True).
    ms2 : bool
        OAllow connections between MS2 pushes
        (the default is False).

    Returns
    -------
    tuple
        A tuple with indptr and indices defining the (sparse) connections.
    """
    connections = []
    connection_count = 0
    connection_counts = [connection_count]
    shape = (
        scan_max_index,
        len(dia_mz_cycle) // scan_max_index
    )

    if multiple_frames:
        frame_iterator = range(shape[1])
    for self_frame in range(shape[1]):
        if not multiple_frames:
            frame_iterator = range(self_frame, self_frame + 1)
        for self_scan in range(shape[0]):
            index = self_scan + self_frame * shape[0]
            low_quad, high_quad = dia_mz_cycle[index]
            if (not ms1) and (low_quad == -1):
                connection_counts.append(connection_count)
                continue
            if (not ms2) and (low_quad != -1):
                connection_counts.append(connection_count)
                continue
            for other_frame in frame_iterator:
                for other_scan in range(
                    self_scan - scan_tolerance,
                    self_scan + scan_tolerance + 1
                ):
                    if not (0 <= other_scan < scan_max_index):
                        continue
                    other_index = other_scan + other_frame * shape[0]
                    if exclude_self and (index == other_index):
                        continue
                    other_low_quad, other_high_quad = dia_mz_cycle[other_index]
                    if low_quad > other_high_quad:
                        continue
                    if high_quad < other_low_quad:
                        continue
                    connection_count += 1
                    connections.append(other_index)
            connection_counts.append(connection_count)
    return np.array(connection_counts), np.array(connections)


@alphatims.utils.njit(nogil=True)
def calculate_cyclic_scan_blur(
    connection_indices: np.ndarray,
    connection_indptr: np.ndarray,
    scan_max_index: int,
    sigma: float = 1,
) -> np.ndarray:
    """Short summary.

    Parameters
    ----------
    connection_indices : np.ndarray
        Connections indices from .get_connections_within_cycle.
    connection_indptr : np.ndarray
        Connections indptr from .get_connections_within_cycle.
    scan_max_index : int
        The maximum scan index (dia_data.scan_max_index).
    sigma : float
        The sigma for the Gaussian blur (default is 1).
        To make sure there are no large dropoffs, this sigma should be at most
        scan_max_index / 3 (see get_connections_within_cycle).

    Returns
    -------
    np.ndarray
        The blurred weight for all the connection_indices.

    """
    scan_blur = np.repeat(
        np.arange(len(connection_indptr) - 1),
        np.diff(connection_indptr),
    ) % scan_max_index - connection_indices % scan_max_index
    scan_blur = np.exp(-(scan_blur / sigma)**2 / 2)
    for i, start in enumerate(connection_indptr[:-1]):
        end = connection_indptr[i + 1]
        scan_blur[start: end] /= np.sum(scan_blur[start: end])
    return scan_blur


@alphatims.utils.njit(nogil=True)
def blur_cycle_in_rt_dimension(
    cycle_index: int,
    cycle_length: int,
    scan_max_index: int,
    zeroth_frame: bool,
    push_indptr: np.ndarray,
    tof_indices: np.ndarray,
    intensity_values: np.ndarray,
    cycle_tolerance: int,
    accumulation_time: np.ndarray,  # TODO
    tof_max_index: int,
    min_intensity: float = 0,
) -> tuple:
    cycle_blur = np.exp(
        -(np.arange(-cycle_tolerance, cycle_tolerance + 1))**2 / 2
    )
    cycle_blur /= np.sum(cycle_blur)
    push_max_index = len(push_indptr) - 1
    intensity_buffer = np.zeros(tof_max_index, np.float32)
    occurance_buffer = np.zeros(tof_max_index, dtype=np.uint8)
    blurred_indptr = [0]
    blurred_tof_indices = []
    blurred_intensity_values = []
    blurred_occurance_counts = []
    for self_push_index in range(
        zeroth_frame * scan_max_index + cycle_index * cycle_length,
        zeroth_frame * scan_max_index + (cycle_index + 1) * cycle_length
    ):
        tofs = []
        for i, cycle_offset in enumerate(
            range(-cycle_tolerance, cycle_tolerance + 1)
        ):
            other_push_index = self_push_index + cycle_offset * cycle_length
            if not (0 <= other_push_index < push_max_index):
                continue
            intensity_multiplier = cycle_blur[i]
            intensity_multiplier /= accumulation_time  # TODO
            for index in range(
                push_indptr[other_push_index],
                push_indptr[other_push_index + 1],
            ):
                tof = tof_indices[index]
                new_intensity = intensity_values[index]
                if new_intensity < min_intensity:
                    continue
                if intensity_buffer[tof] == 0:
                    tofs.append(tof)
                new_intensity *= intensity_multiplier
                intensity_buffer[tof] += new_intensity
                occurance_buffer[tof] += 1
        for tof in tofs:
            blurred_tof_indices.append(tof)
            blurred_occurance_counts.append(occurance_buffer[tof])
            blurred_intensity_values.append(intensity_buffer[tof])
            intensity_buffer[tof] = 0
            occurance_buffer[tof] = 0
        blurred_indptr.append(len(blurred_tof_indices))
    return (
        np.array(blurred_indptr, dtype=np.int64),
        np.array(blurred_tof_indices, dtype=np.uint32),
        np.array(blurred_occurance_counts),
        np.array(blurred_intensity_values, dtype=np.float32),
    )


@alphatims.utils.njit(nogil=True)
def blur_scans(
    blurred_indptr,
    blurred_tof_indices,
    blurred_noisy_values,
    blurred_intensity_values,
    connection_counts,
    connections,
    scan_blur,
    tof_max_index,
    tof_tolerance,
    intensity_cutoff,
    noise_level,
):
    tof_blur = np.exp(
        -(np.arange(-tof_tolerance, tof_tolerance + 1))**2 / 2
    )
    blurred_indptr_ = [0]
    blurred_tof_indices_ = []
    blurred_noisy_values_ = []
    blurred_intensity_values_ = []
    intensity_buffer = np.zeros(tof_max_index)
    noisy_buffer = np.zeros(tof_max_index, dtype=np.uint8)
    for i, start in enumerate(connection_counts[:-1]):
        tofs = []
        end = connection_counts[i + 1]
        for connection_index, blur in zip(
            connections[start: end],
            scan_blur[start: end]
        ):
            for index in range(
                blurred_indptr[connection_index],
                blurred_indptr[connection_index + 1],
            ):
                tof = blurred_tof_indices[index]
                intensity = blurred_intensity_values[index] * blur
                noisy = blurred_noisy_values[index]
                for index, tof_offset in enumerate(
                    range(-tof_tolerance, tof_tolerance + 1)
                ):
                    new_tof = tof + tof_offset
                    if not (0 <= new_tof < tof_max_index):
                        continue
                    intensity = intensity * tof_blur[index]
                    if intensity_buffer[new_tof] == 0:
                        tofs.append(new_tof)
                    noisy_buffer[new_tof] += noisy
                    intensity_buffer[new_tof] += intensity
        tof_maxima = []
        tof_maxima_intensities = []
        for tof in tofs:
            if noisy_buffer[tof] <= noise_level:
                continue
            peak_intensity = intensity_buffer[tof]
            summed_intensity = 0.
            for other_tof in range(tof - tof_tolerance, tof + tof_tolerance + 1):
                if not (0 <= other_tof < tof_max_index):
                    continue
                other_intensity = intensity_buffer[other_tof]
                if other_intensity > peak_intensity:
                    summed_intensity = -np.inf
                    break
                summed_intensity += other_intensity
            if summed_intensity >= intensity_cutoff:
                # TODO expand summed intensity beyond tof_tolerance?
                tof_maxima.append(tof)
                tof_maxima_intensities.append(summed_intensity)
        for index in np.argsort(np.array(tof_maxima)):
            tof = tof_maxima[index]
            summed_intensity = tof_maxima_intensities[index]
            blurred_tof_indices_.append(tof)
            blurred_noisy_values_.append(noisy_buffer[tof])
            blurred_intensity_values_.append(summed_intensity)

        for tof in tofs:
            intensity_buffer[tof] = 0
            noisy_buffer[tof] = 0
        blurred_indptr_.append(len(blurred_tof_indices_))
    return (
        np.array(blurred_indptr_),
        np.array(blurred_tof_indices_, dtype=np.uint32),
        np.array(blurred_noisy_values_),
        np.array(blurred_intensity_values_, dtype=np.float32),
    )


@alphatims.utils.njit(nogil=True)
def create_network(
    blurred_indptr,
    blurred_tof_indices,
    blurred_noisy_values,
    blurred_intensity_values,
    connection_counts,
    connections,
    tof_max_index,
    tof_tolerance,
    intensity_cutoff,
    noise_level,
):
    paired_indptr = np.zeros(len(blurred_tof_indices) + 1, dtype=np.int64)
    potential_peak = np.ones(len(blurred_tof_indices), dtype=np.bool_)
    large_intensity_indices = []
    small_intensity_indices = []
    for push_index, connection_start in enumerate(connection_counts[:-1]):
        self_start = blurred_indptr[push_index]
        self_end = blurred_indptr[push_index + 1]
        if self_start == self_end:
            continue
        connection_end = connection_counts[push_index + 1]
        for connected_push_index in connections[connection_start: connection_end]:
            # if connected_push_index <= push_index:
            #     continue
            if connected_push_index != push_index + 1:
                continue
            other_start = blurred_indptr[connected_push_index]
            other_end = blurred_indptr[connected_push_index + 1]
            if other_start == other_end:
                continue
            self_index = self_start
            other_index = other_start
            while (self_index < self_end) and (other_index < other_end):
                self_tof = blurred_tof_indices[self_index]
                other_tof = blurred_tof_indices[other_index]
                if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                    self_intensity = blurred_intensity_values[self_index]
                    other_intensity = blurred_intensity_values[other_index]
                    if self_intensity > other_intensity:
                        large_intensity_indices.append(self_index)
                        small_intensity_indices.append(other_index)
                        paired_indptr[self_index + 1] += 1
                        potential_peak[other_index] = False
                    elif self_intensity < other_intensity:
                        large_intensity_indices.append(other_index)
                        small_intensity_indices.append(self_index)
                        paired_indptr[other_index + 1] += 1
                        potential_peak[self_index] = False
                    else:
                        large_intensity_indices.append(self_index)
                        small_intensity_indices.append(other_index)
                        paired_indptr[self_index + 1] += 1
                        large_intensity_indices.append(other_index)
                        small_intensity_indices.append(self_index)
                        paired_indptr[other_index + 1] += 1
                if self_tof < other_tof:
                    self_index += 1
                else:
                    other_index += 1
    order = np.argsort(np.array(large_intensity_indices))
    paired_indices = np.array(small_intensity_indices)[order]
    return np.cumsum(paired_indptr), paired_indices, potential_peak
    # for index in order:
    #     paired_indices.append(small_intensity_indices[index])
    # return np.cumsum(paired_indptr), np.array(paired_indices), potential_peak


@alphatims.utils.njit(nogil=True)
def walk_network_from_peaks(
    indptr,
    indices,
    potential_peaks,
):
    assignments = np.full(len(indptr) - 1, -1, dtype=np.int64)
    for assignment_index, start in enumerate(indptr[:-1]):
        if not potential_peaks[assignment_index]:
            continue
        end = indptr[assignment_index + 1]
        if start == end:
            continue
        assignments[assignment_index] = assignment_index
        # print(f"a {assignment_index}, start {start}, end {end}")
        for connection_index in indices[start: end]:
            recursive_walk_network(
                connection_index,
                indptr,
                indices,
                assignment_index,
                assignments,
            )
    return assignments


@alphatims.utils.njit(nogil=True)
def recursive_walk_network(
    index,
    indptr,
    indices,
    assignment,
    assignments,
):
    if assignment == assignments[index]:
        return
    if assignments[index] == -2:
        return
    if assignments[index] != -1:
        assignment = -2
    assignments[index] = assignment
    start = indptr[index]
    end = indptr[index + 1]
    for connection_index in indices[start: end]:
        recursive_walk_network(
            connection_index,
            indptr,
            indices,
            assignment,
            assignments,
        )


@alphatims.utils.njit(nogil=True)
def stats(
    assignments,
    intensity_values,
    tof_values,
    push_indptr,
    scan_max_index,
):
    last_assignment = -np.inf
    assignment_indptr = [0]
    assignment_indices = np.argsort(assignments)
    push_indices = np.repeat(
        np.arange(len(push_indptr) - 1),
        np.diff(push_indptr)
    )
    for offset, index in enumerate(assignment_indices):
        assignment = assignments[index]
        if assignment < 0:
            continue
        if assignment != last_assignment:
            assignment_indptr.append(assignment_indptr[-1])
            last_assignment = assignment
        assignment_indptr[-1] += 1
    assignment_indptr = np.array(assignment_indptr)
    summed_intensities = np.empty(len(assignment_indptr) - 1, dtype=np.float32)
    average_tofs = np.empty(len(assignment_indptr) - 1, dtype=np.uint32)
    average_scans = np.empty(len(assignment_indptr) - 1, dtype=np.uint16)
    lower_scans = np.empty(len(assignment_indptr) - 1, dtype=np.uint16)
    upper_scans = np.empty(len(assignment_indptr) - 1, dtype=np.uint16)
    for index, start in enumerate(assignment_indptr[:-1]):
        end = assignment_indptr[index + 1]
        selection = assignment_indices[start: end]
        selected_intensities = intensity_values[selection]
        summed_intensity = np.sum(selected_intensities)
        selected_tofs = tof_values[selection]
        selected_scans = push_indices[selection] % scan_max_index
        summed_intensities[index] = summed_intensity
        average_tofs[index] = np.sum(selected_tofs * selected_intensities) / summed_intensity
        lower_scans[index] = np.min(selected_scans)
        upper_scans[index] = np.max(selected_scans)
        average_scans[index] = np.sum(selected_scans * selected_intensities) / summed_intensity
    # return np.array(assignment_indptr), assignment_indices, push_indices
    return (
        summed_intensities,
        average_tofs,
        average_scans,
        lower_scans,
        upper_scans,
    )



    # intensity_buffer = np.zeros_like(intensity_values)
    # tof_buffer = np.zeros_like(tof_values)
    # count_buffer = np.zeros(len(assignments), dtype=np.uint32)
    # upper_mobility = np.zeros(len(assignments), dtype=np.uint16)
    # center_mobility = np.zeros(len(assignments), dtype=np.uint16)
    # lower_mobility = np.zeros(len(assignments), dtype=np.uint16)
    # for push_index, start in enumerate(push_indptr[:-1]):
    #     end = push_indptr[push_index + 1]
    #     for index, assignment in enumerate(assignments[start: end]):
    #         if assignment < 0:
    #             continue
    #         tof = tof_values[index]
    #         intensity_value = intensity_values[index]


@alphatims.utils.njit(nogil=True)
def connect_ions(
    assignments,
    intensity_values,
    mz_centroided_indptr,
    min_overlap_length,
    min_cosine,
    min_overlap_percentage,
    min_jaccard_overlap_percentage,
):
    assignment_indptr = np.zeros(np.max(assignments) + 1, dtype=np.int64)
    for assignment in assignments:
        assignment_indptr[assignment + 1] += 1
    assignment_indptr = np.cumsum(assignment_indptr)
    assignment_counts = np.diff(assignment_indptr)
    order = np.arange(len(assignments))[np.argsort(assignments)]
    push_indices = np.repeat(
        np.arange(len(mz_centroided_indptr) - 1),
        np.diff(mz_centroided_indptr),
    )
    connection_count_buffer = np.zeros(mz_centroided_indptr[-1], dtype=np.uint16)
    connection_self_square_buffer = np.zeros(mz_centroided_indptr[-1])
    connection_other_square_buffer = np.zeros(mz_centroided_indptr[-1])
    connection_cross_product_buffer = np.zeros(mz_centroided_indptr[-1])
    assignment_connection_indptr = [0]
    assignment_connection_indices = []
    assignment_connection_counts = []
    assignment_connection_cosines = []
    for assignment, start in enumerate(assignment_indptr):
        end = assignment_indptr[assignment + 1]
        if start == end:
            assignment_connection_indptr.append(
                len(assignment_connection_indices)
            )
            continue
        self_count = assignment_counts[assignment]
        selected_assignments = []
        for index in order[start: end]:
            push_index = push_indices[index]
            self_intensity = intensity_values[index]
        # for index, push_index in enumerate(push_indices[start: end], start):
            for other_index in range(
                mz_centroided_indptr[push_index],
                mz_centroided_indptr[push_index + 1],
            ):
                other_assignment = assignments[other_index]
                if other_assignment < 0:
                    continue
                if other_assignment == assignment:
                    continue
                if connection_count_buffer[other_assignment] == 0:
                    selected_assignments.append(other_assignment)
                other_intensity = intensity_values[other_index]
                connection_count_buffer[other_assignment] += 1
                connection_self_square_buffer[other_assignment] += self_intensity * self_intensity
                connection_other_square_buffer[other_assignment] += other_intensity * other_intensity
                connection_cross_product_buffer[other_assignment] += self_intensity * other_intensity

        for other_assignment in selected_assignments:
            overlap_count = connection_count_buffer[other_assignment]
            if overlap_count >= min_overlap_length:
                other_count = assignment_counts[other_assignment]
                overlap_percentage = overlap_count / min(self_count, other_count)
                jaccard_overlap_percentage = overlap_count / max(self_count, other_count)
                if (
                    overlap_percentage >= min_overlap_percentage
                ) and (
                    jaccard_overlap_percentage >= min_jaccard_overlap_percentage
                ):
                    self_square = np.sqrt(connection_self_square_buffer[other_assignment])
                    other_square = np.sqrt(connection_other_square_buffer[other_assignment])
                    cosine = connection_cross_product_buffer[other_assignment] / (self_square * other_square)
                    if cosine >= min_cosine:
                        assignment_connection_indices.append(other_assignment)
                        assignment_connection_counts.append(overlap_count)
                        assignment_connection_cosines.append(cosine)
            connection_count_buffer[other_assignment] = 0
            connection_self_square_buffer[other_assignment] = 0
            connection_other_square_buffer[other_assignment] = 0
            connection_cross_product_buffer[other_assignment] = 0

        assignment_connection_indptr.append(len(assignment_connection_indices))
    return (
        np.array(assignment_connection_indptr),
        np.array(assignment_connection_indices),
        np.array(assignment_connection_counts),
        np.array(assignment_connection_cosines),
    )


@alphatims.utils.njit(nogil=True)
def cleanup_assignments(
    assignments,
    mz_centroided_indptr,
    minimum_assignment_size,
):
    assignment_buffer = np.zeros(len(assignments), dtype=np.uint32)
    found_assignments = []
    for assignment in assignments:
        if assignment < 0:
            continue
        assignment_buffer[assignment] += 1
        if assignment_buffer[assignment] == minimum_assignment_size:
            found_assignments.append(assignment)
    assignment_buffer[:] = 0
    for new_assignment, assignment in enumerate(found_assignments, 1):
        assignment_buffer[assignment] = new_assignment

    new_mz_centroided_indptr = [0]
    new_assignments = []
    selected_buffer = np.zeros(len(assignments), dtype=np.bool_)

    for index, start in enumerate(mz_centroided_indptr[:-1]):
        end = mz_centroided_indptr[index + 1]
        for offset, assignment in enumerate(assignments[start: end], start):
            new_assignment = assignment_buffer[assignment]
            if new_assignment != 0:
                new_assignments.append(new_assignment - 1)
                selected_buffer[offset] = True
        new_mz_centroided_indptr.append(len(new_assignments))
    return (
        np.array(new_mz_centroided_indptr),
        np.array(new_assignments),
        selected_buffer,
    )


@alphatims.utils.pjit
def create_full_network(
    push_index,
    indptr,
    tof_indices,
    intensity_values,
    tof_tolerance,
    cycle_length,
    scan_max_index,
    potential_peaks,
    potential_connections,
):
    other_push_indices = []
    if (push_index + 1) % scan_max_index != 0:
        other_push_indices.append(push_index + 1)
    if (push_index + cycle_length) < len(indptr):
        other_push_indices.append(push_index + cycle_length)
    # if (len(other_push_indices) == 2) and ((push_index + 1 + cycle_length) < len(indptr)):
    #     other_push_indices.append(push_index + cycle_length + 1)
    self_start = indptr[push_index]
    self_end = indptr[push_index + 1]
    if self_start == self_end:
        return
    for index, other_push_index in enumerate(other_push_indices):
        other_start = indptr[other_push_index]
        other_end = indptr[other_push_index + 1]
        if other_start == other_end:
            continue
        self_index = self_start
        other_index = other_start
        while (self_index < self_end) and (other_index < other_end):
            # if not potential_peaks[self_index]:
            #     self_index += 1
            #     continue
            self_tof = tof_indices[self_index]
            other_tof = tof_indices[other_index]
            if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                self_intensity = intensity_values[self_index]
                other_intensity = intensity_values[other_index]
                if self_intensity > other_intensity:
                    potential_peaks[other_index] = False
                    potential_connections[self_index, 0, index] = other_index
                elif self_intensity < other_intensity:
                    potential_peaks[self_index] = False
                    potential_connections[other_index, 1, index] = self_index
                else:
                    potential_connections[self_index, 0, index] = other_index
                    potential_connections[other_index, 1, index] = self_index
            if self_tof < other_tof:
                self_index += 1
            else:
                other_index += 1


@alphatims.utils.njit(nogil=True)
def clean_connections(
    connections,
):
    indptr = [0]
    indices = []
    for index, neighbors in enumerate(connections):
        for connection in neighbors.ravel():
            if connection != -1:
                indices.append(connection)
        indptr.append(len(indices))
    return np.array(indptr), np.array(indices)


@alphatims.utils.njit(nogil=True)
def frame_peak_connections(
    self_indptr,
    self_tof_indices,
    self_intensity_values,
    other_indptr,
    other_tof_indices,
    other_intensity_values,
    tof_tolerance,
):
    self_potential_peaks = np.ones(len(self_tof_indices), dtype=np.bool_)
    other_potential_peaks = np.ones(len(other_tof_indices), dtype=np.bool_)
    self_connections = []
    other_connections = []
    for push_index, self_start in enumerate(self_indptr[:-1]):
        self_end = self_indptr[push_index + 1]
        if self_start == self_end:
            continue
        other_start = other_indptr[push_index]
        other_end = other_indptr[push_index + 1]
        if other_start == other_end:
            continue
        self_index = self_start
        other_index = other_start
        while (self_index < self_end) and (other_index < other_end):
            self_tof = self_tof_indices[self_index]
            other_tof = other_tof_indices[other_index]
            if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                self_intensity = self_intensity_values[self_index]
                other_intensity = other_intensity_values[other_index]
                if self_intensity > other_intensity:
                    other_potential_peaks[other_index] = False
                    self_connections.append((self_index, other_index))
                elif self_intensity < other_intensity:
                    self_potential_peaks[self_index] = False
                    other_connections.append((other_index, self_index))
                else:
                    self_connections.append((self_index, other_index))
                    other_connections.append((other_index, self_index))
            if self_tof < other_tof:
                self_index += 1
            else:
                other_index += 1
    return (
        self_potential_peaks,
        other_potential_peaks,
        np.array(self_connections, dtype=np.uint32),
        np.array(other_connections, dtype=np.uint32),
    )

@alphatims.utils.njit(nogil=True)
def scan_peak_connections(
    indptr,
    tof_indices,
    intensity_values,
    tof_tolerance,
    scan_max_index,
):
    potential_peaks = np.ones(len(tof_indices), dtype=np.bool_)
    connections = []
    for push_index, self_start in enumerate(indptr[:-1]):
        self_end = indptr[push_index + 1]
        if self_start == self_end:
            continue
        if (push_index + 1) % scan_max_index == 0:
            continue
        other_start = indptr[push_index + 1]
        other_end = indptr[push_index + 2]
        if other_start == other_end:
            continue
        self_index = self_start
        other_index = other_start
        while (self_index < self_end) and (other_index < other_end):
            self_tof = tof_indices[self_index]
            other_tof = tof_indices[other_index]
            if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                self_intensity = intensity_values[self_index]
                other_intensity = intensity_values[other_index]
                if self_intensity > other_intensity:
                    potential_peaks[other_index] = False
                    connections.append((self_index, other_index))
                elif self_intensity < other_intensity:
                    potential_peaks[self_index] = False
                    connections.append((other_index, self_index))
                else:
                    connections.append((self_index, other_index))
                    connections.append((other_index, self_index))
            if self_tof < other_tof:
                self_index += 1
            else:
                other_index += 1
    return (
        potential_peaks,
        np.array(connections, dtype=np.uint32),
    )


@alphatims.utils.threadpool
def blur_and_centroid(
    cycle_index,
    scan_tolerance,
    multiple_frames_per_cycle,
    cycle_tolerance,
    tof_tolerance,
    intensity_cutoff,
    noise_level,
    cycle_count,
    cycle_length,
    dia_data,
    connection_counts,
    connections,
    scan_blur,
    hdf_file,
):
    (
        rt_blurred_indptr,
        rt_blurred_tof_indices,
        rt_blurred_occurance_counts,
        rt_blurred_intensity_values,
    ) = blur_cycle_in_rt_dimension(
        cycle_index=cycle_index,
        cycle_length=len(dia_data.dia_mz_cycle),
        scan_max_index=dia_data.scan_max_index,
        zeroth_frame=dia_data.zeroth_frame,
        push_indptr=dia_data.push_indptr,
        tof_indices=dia_data.tof_indices,
        intensity_values=dia_data.intensity_values,
        cycle_tolerance=cycle_tolerance,
        accumulation_time=1,
        tof_max_index=dia_data.tof_max_index,
    )
    (
        mz_centroided_indptr,
        mz_centroided_tof_indices,
        mz_centroided_noisy_values,
        mz_centroided_intensity_values,
    ) = blur_scans(
        rt_blurred_indptr,
        rt_blurred_tof_indices,
        rt_blurred_occurance_counts,
        rt_blurred_intensity_values,
        connection_counts,
        connections,
        scan_blur,
        dia_data.tof_max_index,
        tof_tolerance,
        intensity_cutoff,
        noise_level,
    )
    (
        potential_peaks,
        scan_connections,
    ) = scan_peak_connections(
        mz_centroided_indptr,
        mz_centroided_tof_indices,
        mz_centroided_intensity_values,
        tof_tolerance,
        dia_data.scan_max_index,
    )
    if hdf_file is not None:
        hdf_file.__setattr__(
            f"cycle_{cycle_index}",
            {
                "indptr": mz_centroided_indptr,
                "tof_indices": mz_centroided_tof_indices,
                "intensity_values": mz_centroided_intensity_values,
                "potential_peaks": potential_peaks,
                "scan_connections": scan_connections,
#                 "assignments": new_assignments,
            }
        )



@alphatims.utils.threadpool
def process_connections(
    cycle_index,
    tof_tolerance,
    scan_max_index,
    hdf_file,
):
    try:
        self_group = hdf_file.__getattribute__(f"cycle_{cycle_index}")
        self_indptr = self_group.indptr.values
        self_tof_indices = self_group.tof_indices.values
        self_intensity_values = self_group.intensity_values.values
    except AttributeError:
        return

    (
        potential_peaks,
        scan_connections,
    ) = scan_peak_connections(
        self_indptr,
        self_tof_indices,
        self_intensity_values,
        tof_tolerance,
        scan_max_index,
    )

    self_group.scan_connections = scan_connections

    if cycle_index == 0:
        self_group.potential_peaks_secondary = potential_peaks
        self_group.frame_connections_secondary = np.empty(
            (0, 2),
            dtype=scan_connections.dtype
        )
    try:
        other_group = hdf_file.__getattribute__(f"cycle_{cycle_index + 1}")
        other_indptr = other_group.indptr.values
        other_tof_indices = other_group.tof_indices.values
        other_intensity_values = other_group.intensity_values.values
    except AttributeError:
        self_group.potential_peaks = potential_peaks
        self_group.frame_connections = np.empty(
            (0, 2),
            dtype=scan_connections.dtype
        )
        return

    (
        self_potential_peaks,
        other_potential_peaks,
        self_frame_connections,
        other_frame_connections,
    ) = frame_peak_connections(
        self_indptr,
        self_tof_indices,
        self_intensity_values,
        other_indptr,
        other_tof_indices,
        other_intensity_values,
        tof_tolerance,
    )
    self_group.potential_peaks = potential_peaks & self_potential_peaks
    other_group.potential_peaks_secondary = other_potential_peaks
    self_group.frame_connections = self_frame_connections
    other_group.frame_connections_secondary = other_frame_connections


def centroid(
    dia_data,
    hdf_file,
    scan_tolerance=6,
    scan_sigma=2,
    multiple_frames_per_cycle=False,
    ms1=True,
    ms2=False,
    cycle_tolerance=3,
    tof_tolerance=3,
    intensity_cutoff=0,
    noise_level=4,
    skip_blurring=False
):

    cycle_count = len(dia_data.push_indptr) // len(dia_data.dia_mz_cycle)
    cycle_length = len(dia_data.dia_mz_cycle)
    if not skip_blurring:
        connection_counts, connections = get_connections_within_cycle(
            scan_tolerance=scan_tolerance,
            scan_max_index=dia_data.scan_max_index,
            dia_mz_cycle=dia_data.dia_mz_cycle,
            multiple_frames=multiple_frames_per_cycle,
            ms1=ms1,
            ms2=ms2,
        )
        scan_blur = calculate_cyclic_scan_blur(
            connections,
            connection_counts,
            dia_data.scan_max_index,
            sigma=scan_sigma,
        )
        blur_and_centroid(
            range(cycle_count + 1),
            scan_tolerance,
            multiple_frames_per_cycle,
            cycle_tolerance,
            tof_tolerance,
            intensity_cutoff,
            noise_level,
            cycle_count,
            cycle_length,
            dia_data,
            connection_counts,
            connections,
            scan_blur,
            hdf_file,
        )
    process_connections(
        range(cycle_count + 1),
        tof_tolerance,
        dia_data.scan_max_index,
        hdf_file,
    )


@alphatims.utils.njit(nogil=True)
def update_connection_indptr(
    index,
    indptr,
    cycle_indptr,
    connections,
):
    start = cycle_indptr[index]
    for connection in connections[:, 0]:
        indptr[1 + start + connection] += 1


# @alphatims.utils.njit(nogil=True)
def update_connection_indices(
    index,
    cycle_indptr,
    connection_indices,
    connection_indptr,
    scan_connections,
    frame_connections,
    frame_connections_secondary,
):
    high_connections = []
    low_connections = []
    if len(frame_connections) > 0:
        start = cycle_indptr[index + 1]
        for high_connection, low_connection in frame_connections:
            high_connections.append(high_connection)
            low_connections.append(low_connection + start)
    start = cycle_indptr[index]
    offset = connection_indptr[start]
    # print(start, offset)
    for high_connection, low_connection in scan_connections:
        high_connections.append(high_connection)
        low_connections.append(low_connection + start)
    if len(frame_connections_secondary) > 0:
        start = cycle_indptr[index - 1]
        for high_connection, low_connection in frame_connections_secondary:
            high_connections.append(high_connection)
            low_connections.append(low_connection + start)
    # print(len(low_connections))
    # print(len(high_connections))
    order = np.argsort(np.array(high_connections))
    # print(len(order))
    for i, index in enumerate(order, offset):
        connection_indices[i] = low_connections[index]


@alphatims.utils.threadpool
def create_connection_indices(
    index,
    cycle_indptr,
    scan_connections,
    frame_connections,
    frame_connections_secondary,
    connection_indices,
    connection_indptr,
):
    # scan_connection = scan_connections[index].astype(np.int64)
    # try:
    #     frame_connection = frame_connections[index]
    # except IndexError:
    #     frame_connection = np.empty((0, 2), dtype=scan_connection.dtype)
    # if index != 0:
    #     frame_connection_secondary = frame_connections_secondary[index - 1]
    # else:
    #     frame_connection_secondary = np.empty((0, 2), dtype=scan_connection.dtype)
    # frame_connection = frame_connections[index].astype(np.int64)
    # frame_connection_secondary = frame_connections_secondary[index].astype(np.int64)
    update_connection_indices(
        index,
        cycle_indptr,
        connection_indices,
        connection_indptr,
        scan_connections[index].astype(np.int64),
        frame_connections[index].astype(np.int64),
        frame_connections_secondary[index].astype(np.int64),
    )


@alphatims.utils.threadpool
def create_connection_indptr(
    index,
    indptr,
    cycle_indptr,
    scan_connections,
    frame_connections,
    frame_connections_secondary,
):
    update_connection_indptr(
        index,
        indptr,
        cycle_indptr,
        scan_connections[index],
    )
    update_connection_indptr(
        index,
        indptr,
        cycle_indptr,
        frame_connections[index],
    )
    update_connection_indptr(
        index,
        indptr,
        cycle_indptr,
        frame_connections_secondary[index],
    )



def load_connections(
    hdf_file
):
    groups = np.array(hdf_file.groups)
    group_names = [int(group.name.split("_")[1]) for group in groups]
    groups = groups[np.argsort(group_names)]
    sizes = [group.potential_peaks.shape[0] for group in groups]
    sizes
    potential_peaks = np.concatenate(
        [group.potential_peaks.values for group in groups]
    )
    start = sizes[0]
    for size, group in zip(
        sizes[1:],
        groups[1:],
    ):
        end = start + size
        potential_peaks[start: end] &= group.potential_peaks_secondary.values
        start = end
    cycle_indptr = np.empty(len(sizes) + 1, dtype=np.int64)
    cycle_indptr[0] = 0
    cycle_indptr[1:] = np.cumsum(sizes)
    return potential_peaks, cycle_indptr
