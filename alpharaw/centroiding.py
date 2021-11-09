import alphatims.utils
import numpy as np


@alphatims.utils.njit(cache=False, nogil=True)
def connection_cycle(
    scan_tolerance,
    scan_max_index,
    dia_mz_cycle,
    exclude_self=False,
):
    connections = []
    connection_count = 0
    connection_counts = [connection_count]
    shape = (
        scan_max_index,
        len(dia_mz_cycle) // scan_max_index
    )

    for j in range(shape[1]):
        for i in range(shape[0]):
            low_i = max(i - scan_tolerance, 0)
            high_i = min(i + scan_tolerance + 1, shape[0])
            index = i + j * shape[0]
            low_quad, high_quad = dia_mz_cycle[index]
            for l in range(shape[1]):
                for k in range(low_i, high_i):
                    other_index = k + l * shape[0]
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
    connection_counts = np.array(connection_counts)
    connections = np.array(connections)
    return connection_counts, connections


@alphatims.utils.njit(cache=False, nogil=True)
def calculate_scan_blur(
    connections,
    connection_counts,
    # TODO set sigma?
    scan_max_index,
):
    scan_blur = np.repeat(
        np.arange(len(connection_counts) - 1),
        np.diff(connection_counts),
    ) % scan_max_index - connections % scan_max_index
    scan_blur = np.exp(-(scan_blur)**2 / 2)
    for i, start in enumerate(connection_counts[:-1]):
        end = connection_counts[i + 1]
        scan_blur[start: end] /= np.sum(scan_blur[start: end])
    return scan_blur


# @alphatims.utils.njit(cache=False, nogil=True)
def blur_cycles2(
    cycle_length,
    scan_max_index,
    zeroth_frame,
    push_indptr,
    tof_indices,
    intensity_values,
    cycle_index,
    cycle_tolerance,
    accumulation_time,
    tof_max_index,
):
    cycle_blur = np.exp(
        -(np.arange(-cycle_tolerance, cycle_tolerance + 1))**2 / 2
    )
    cycle_blur /= np.sum(cycle_blur)
    start_push_index = zeroth_frame * scan_max_index + cycle_index * cycle_length
    end_push_index = start_push_index + cycle_length
    blurred_indptr = np.zeros(cycle_length + 1, dtype=np.int64)
    push_max_index = len(push_indptr)
    for cycle_offset in range(-cycle_tolerance, cycle_tolerance + 1):
        offset_start_push_index = start_push_index + cycle_offset * cycle_length
        if offset_start_push_index < 0:
            continue
        offset_end_push_index = offset_start_push_index + cycle_length
        if offset_end_push_index >= push_max_index:
            continue
        blurred_indptr[1:] += np.diff(
            push_indptr[offset_start_push_index: offset_end_push_index + 1]
        )
    blurred_indptr = np.cumsum(blurred_indptr)
    blurred_tof_indices = np.empty(blurred_indptr[-1], dtype=tof_indices.dtype)
    blurred_intensity_values = np.empty(blurred_indptr[-1])
    offset = 0
    for self_push_index in range(start_push_index, end_push_index):
        for i, cycle_offset in enumerate(
            range(-cycle_tolerance, cycle_tolerance + 1)
        ):
            other_push_index = self_push_index + cycle_offset * cycle_length
            if other_push_index < 0:
                continue
            if other_push_index >= push_max_index:
                continue
            intensity_multiplier = cycle_blur[i]
            intensity_multiplier /= accumulation_time # TODO
            index_start = push_indptr[other_push_index]
            index_end = push_indptr[other_push_index + 1]
            target_slice = slice(offset, offset + index_end - index_start)
            blurred_tof_indices[target_slice] = tof_indices[
                index_start: index_end
            ]
            blurred_intensity_values[target_slice] = intensity_values[
                index_start: index_end
            ] * intensity_multiplier
            offset += index_end - index_start
    return (
        blurred_indptr,
        blurred_tof_indices,
        blurred_intensity_values,
    )


@alphatims.utils.njit(cache=False, nogil=True)
def blur_scans2(
    blurred_indptr,
    blurred_tof_indices,
    blurred_intensity_values,
    connection_counts,
    connections,
    scan_blur,
):
    count = 0
    blurred_indptr_ = [count]
    blurred_tof_indices_ = []
    blurred_intensity_values_ = []
    for i, start in enumerate(connection_counts[:-1]):
        end = connection_counts[i + 1]
        scan_blur_ = scan_blur[start: end]
        connections_ = connections[start: end]
        for connection_index, blur in zip(connections_, scan_blur_):
            start_index = blurred_indptr[connection_index]
            end_index = blurred_indptr[connection_index + 1]
            blurred_tof_indices_.append(
                blurred_tof_indices[start_index: end_index]
            )
            blurred_intensity_values_.append(
                blurred_intensity_values[start_index: end_index] * blur
            )
            count += end_index - start_index
        # TODO input tof blur here?
        blurred_indptr_.append(count)
    return (
        np.array(blurred_indptr_),
        concatenate(blurred_tof_indices_),
        concatenate(blurred_intensity_values_),
    )


@alphatims.utils.njit(cache=False, nogil=True)
def concatenate(array_list):
    concatenated_array = np.empty(
        sum([len(a) for a in array_list]),
        array_list[0].dtype
    )
    offset = 0
    for i in array_list:
        concatenated_array[offset: offset + len(i)] = i
        offset += len(i)
    return concatenated_array


@alphatims.utils.njit(cache=False, nogil=True)
def blur_tof(
    blurred_indptr,
    blurred_tof_indices,
    blurred_intensity_values,
    tof_tolerance,
):
    tof_blur = np.exp(
        -(np.arange(-tof_tolerance, tof_tolerance + 1))**2 / 2
    )
    tof_blur /= np.sum(tof_blur)
    count = 0
    blurred_indptr_ = [count]
    blurred_tof_indices_ = []
    blurred_intensity_values_ = []
    for i, start_index in enumerate(blurred_indptr[:-1]):
        end_index = blurred_indptr[i + 1]
        intensity_values = blurred_intensity_values[start_index: end_index]
        tof_indices = blurred_tof_indices[start_index: end_index]
        for i, tof_offset in enumerate(
            range(-tof_tolerance, tof_tolerance + 1)
        ):
            blurred_tof_indices_.append(
                tof_indices + tof_offset
            )
            blurred_intensity_values_.append(
                intensity_values * tof_blur[i]
            )
            count += end_index - start_index
        blurred_indptr_.append(count)
    # # order = np.argsort(blurred_tof_indices[start_offset: offset])
    # # blurred_tof_indices[start_offset: offset] = blurred_tof_indices[start_offset: offset][order]
    # # blurred_intensity_values[start_offset: offset] = blurred_intensity_values[start_offset: offset][order]
    return (
        np.array(blurred_indptr_),
        concatenate(blurred_tof_indices_),
        concatenate(blurred_intensity_values_),
    )


@alphatims.utils.njit(cache=False, nogil=True)
def blur(
    cycle_index,
    cycle_length,
    scan_max_index,
    zeroth_frame,
    push_indptr,
    tof_indices,
    intensity_values,
    cycle_tolerance,
    connection_counts,
    connections,
    scan_blur,
    tof_tolerance,
    accumulation_time,
    intensity_cutoff,
    tof_max_index,
    noise_level,
):
    (
        blurred_indptr,
        blurred_tof_indices,
        blurred_noisy_values,
        blurred_intensity_values,
    ) = blur_cycles(
        cycle_length,
        scan_max_index,
        zeroth_frame,
        push_indptr,
        tof_indices,
        intensity_values,
        cycle_index,
        cycle_tolerance,
        accumulation_time,
        tof_max_index,
    )
    # if blurred_indptr[-1] == 0:
    #     return (
    #         blurred_indptr,
    #         blurred_tof_indices,
    #         blurred_intensity_values,
    #     )
    #     # TODO test for empty blurred_indptr
    (
        blurred_indptr,
        blurred_tof_indices,
        blurred_noisy_values,
        blurred_intensity_values,
    # ) = blur_scans_full_width(
    ) = blur_scans(
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
    )
    return (
        blurred_indptr,
        blurred_tof_indices,
        blurred_noisy_values,
        blurred_intensity_values,
    )


@alphatims.utils.njit(cache=False, nogil=True)
def centroid_tof2(
    blurred_indptr,
    blurred_tof_indices,
    blurred_intensity_values,
    tof_tolerance,
    intensity_cutoff,
):
    tof_blur = np.exp(
        -(np.arange(-tof_tolerance, tof_tolerance + 1))**2 / 2
    )
    tof_blur /= np.sum(tof_blur)
    count = 0
    blurred_indptr_ = [count]
    blurred_tof_indices_ = []
    blurred_intensity_values_ = []
    max_tof = np.max(blurred_tof_indices)
    intensities = np.zeros(max_tof + 2 * tof_tolerance)
    for i, start_index in enumerate(blurred_indptr[:-1]):
        end_index = blurred_indptr[i + 1]
        if start_index == end_index:
            blurred_indptr_.append(count)
            continue
        intensity_values = blurred_intensity_values[start_index: end_index]
        tof_indices = blurred_tof_indices[start_index: end_index] + tof_tolerance
        unique_tofs = np.array(list(set(list(tof_indices))))
        # unique_tofs = np.unique(tof_indices)
        for index, tof_offset in enumerate(
            range(-tof_tolerance, tof_tolerance + 1)
        ):
            for tof, intensity in zip(
                tof_indices + tof_offset,
                intensity_values,
            ):
                intensities[tof] += intensity * tof_blur[index]
        for tof in unique_tofs:
            peak_intensity = intensities[tof]
            if peak_intensity < intensity_cutoff:
                continue
            if intensities[tof - 1] > peak_intensity:
                continue
            if intensities[tof + 1] > peak_intensity:
                continue
            blurred_tof_indices_.append(tof - tof_tolerance)
            blurred_intensity_values_.append(peak_intensity)
            count += 1
            left_border = tof - 1
            while intensities[left_border] <= peak_intensity:
                blurred_intensity_values_[-1] += intensities[left_border]
                peak_intensity = intensities[left_border]
                left_border -= 1
            right_border = tof + 1
            while intensities[right_border] <= peak_intensity:
                blurred_intensity_values_[-1] += intensities[right_border]
                peak_intensity = intensities[right_border]
                right_border += 1
        blurred_indptr_.append(count)
        for tof_offset in range(-tof_tolerance, tof_tolerance + 1):
            for tof in (unique_tofs + tof_offset):
                intensities[tof] = 0
    return (
        np.array(blurred_indptr_),
        np.array(blurred_tof_indices_, dtype=blurred_tof_indices.dtype),
        np.array(blurred_intensity_values_),
    )

















@alphatims.utils.njit(cache=False, nogil=True)
def blur_cycles(
    cycle_length,
    scan_max_index,
    zeroth_frame,
    push_indptr,
    tof_indices,
    intensity_values,
    cycle_index,
    cycle_tolerance,
    accumulation_time,
    tof_max_index,
):
    cycle_blur = np.exp(
        -(np.arange(-cycle_tolerance, cycle_tolerance + 1))**2 / 2
    )
    cycle_blur /= np.sum(cycle_blur)
    start_push_index = zeroth_frame * scan_max_index + cycle_index * cycle_length
    end_push_index = start_push_index + cycle_length
    push_max_index = len(push_indptr) - 1
    intensity_buffer = np.zeros(tof_max_index, np.float32)
    noisy_buffer = np.zeros(tof_max_index, dtype=np.uint8)
    blurred_indptr = [0]
    blurred_tof_indices = []
    blurred_intensity_values = []
    blurred_noisy_values = []
    for self_push_index in range(start_push_index, end_push_index):
        tofs = []
        for i, cycle_offset in enumerate(
            range(-cycle_tolerance, cycle_tolerance + 1)
        ):
            other_push_index = self_push_index + cycle_offset * cycle_length
            if not (0 <= other_push_index < push_max_index):
                continue
            intensity_multiplier = cycle_blur[i]
            intensity_multiplier /= accumulation_time # TODO
            for index in range(
                push_indptr[other_push_index],
                push_indptr[other_push_index + 1],
            ):
                tof = tof_indices[index]
                if intensity_buffer[tof] == 0:
                    tofs.append(tof)
                noisy_buffer[tof] += 1
                intensity_buffer[tof] += intensity_values[index] * intensity_multiplier
        for tof in tofs:
            blurred_tof_indices.append(tof)
            blurred_noisy_values.append(noisy_buffer[tof])
            blurred_intensity_values.append(intensity_buffer[tof])
            intensity_buffer[tof] = 0
            noisy_buffer[tof] = 0
        blurred_indptr.append(len(blurred_tof_indices))
    return (
        np.array(blurred_indptr),
        np.array(blurred_tof_indices, dtype=np.uint32),
        np.array(blurred_noisy_values),
        np.array(blurred_intensity_values),
    )



@alphatims.utils.njit(cache=False, nogil=True)
def blur_scans3(
    blurred_indptr,
    blurred_tof_indices,
    blurred_intensity_values,
    connection_counts,
    connections,
    scan_blur,
    tof_max_index,
):
    blurred_indptr_ = [0]
    blurred_tof_indices_ = []
    blurred_intensity_values_ = []
    intensity_buffer = np.zeros(tof_max_index, np.float32)
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
                tofs.append(tof)
                intensity_buffer[tof] += blurred_intensity_values[index] * blur
            #
            # start_index = blurred_indptr[connection_index]
            # end_index = blurred_indptr[connection_index + 1]
            # blurred_tof_indices_.append(
            #     blurred_tof_indices[start_index: end_index]
            # )
            # blurred_intensity_values_.append(
            #     blurred_intensity_values[start_index: end_index] * blur
            # )
            # count += end_index - start_index
        for tof in np.unique(np.array(tofs)):
            blurred_tof_indices_.append(tof)
            blurred_intensity_values_.append(intensity_buffer[tof])
            intensity_buffer[tof] = 0
        blurred_indptr_.append(len(blurred_tof_indices_))
        # TODO input tof blur here?
    return (
        np.array(blurred_indptr_),
        np.array(blurred_tof_indices_),
        np.array(blurred_intensity_values_),
    )





@alphatims.utils.njit(cache=False, nogil=True)
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
        np.array(blurred_intensity_values_),
    )


@alphatims.utils.njit(cache=False, nogil=True)
def cluster_cycle2(
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
    potential_peaks = np.ones(len(blurred_tof_indices), dtype=np.bool_)
    summed_intensities = blurred_intensity_values.copy()
    for push_index, connection_start in enumerate(connection_counts[:-1]):
        self_start = blurred_indptr[push_index]
        self_end = blurred_indptr[push_index + 1]
        if self_start == self_end:
            continue
        connection_end = connection_counts[push_index + 1]
        for connected_push_index in connections[connection_start: connection_end]:
            if connected_push_index == push_index:
                continue
            other_start = blurred_indptr[connected_push_index]
            other_end = blurred_indptr[connected_push_index + 1]
            if other_start == other_end:
                continue
            self_index = self_start
            other_index = other_start
            while (self_index < self_end) and (other_index < other_end):
                if blurred_noisy_values[self_index] <= noise_level:
                    potential_peaks[self_index] = False
                if not potential_peaks[self_index]:
                    self_index += 1
                    continue
                self_tof = blurred_tof_indices[self_index]
                other_tof = blurred_tof_indices[other_index]
                if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                    self_intensity = blurred_intensity_values[self_index]
                    other_intensity = blurred_intensity_values[other_index]
                    summed_intensities[self_index] += other_intensity
                    summed_intensities[other_index] += self_intensity
                    if self_intensity < other_intensity:
                        potential_peaks[self_index] = False
                    elif other_intensity < self_intensity:
                        potential_peaks[other_index] = False
                if self_tof < other_tof:
                    self_index += 1
                else:
                    other_index += 1
    return potential_peaks, summed_intensities



@alphatims.utils.njit(cache=False, nogil=True)
def cluster_cycle(
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
    clusters = np.arange(len(blurred_tof_indices))
    for push_index, connection_start in enumerate(connection_counts[:-1]):
        self_start = blurred_indptr[push_index]
        self_end = blurred_indptr[push_index + 1]
        if self_start == self_end:
            continue
        connection_end = connection_counts[push_index + 1]
        for connected_push_index in connections[connection_start: connection_end]:
            if connected_push_index <= push_index:
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
                    clusters[other_index] = clusters[self_index]
                    # update_clusters(
                    #     clusters,
                    #     self_index,
                    #     other_index,
                    # )
                if self_tof < other_tof:
                    self_index += 1
                else:
                    other_index += 1
    return clusters


@alphatims.utils.njit(cache=False, nogil=True)
def update_clusters(
    clusters,
    self_index,
    other_index,
):
    self_cluster = clusters[self_index]
    other_cluster = clusters[other_index]
    if self_cluster < other_cluster:
        # other_cluster = trace_clusters(clusters, other_index, self_cluster)
        clusters[other_index] = self_cluster
    elif self_cluster > other_cluster:
        # self_cluster = trace_clusters(clusters, self_index, other_cluster)
        clusters[self_index] = other_cluster
    # if self_cluster > other_cluster:
    #     self_index, other_index = other_index, self_index
    #     self_cluster, other_cluster = other_cluster, self_cluster
    # if self_cluster > other_cluster:
    #     clusters[self_index] = other_cluster
    # else:
    #     clusters[other_index] = self_cluster
    # if self_clu


@alphatims.utils.njit(cache=False, nogil=True)
def trace_clusters(
    clusters,
    index,
    new_cluster,
):
    trace = []
    cluster = clusters[index]
    while cluster < index:
        trace.append(index)
        index = cluster
        cluster = clusters[index]
    if new_cluster < cluster:
        cluster = new_cluster
    for index in trace:
        clusters[index] = cluster
    return cluster




@alphatims.utils.njit(cache=False, nogil=True)
def blur_scans_full_width(
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
    blurred_left_tof_indices_ = []
    # blurred_center_tof_indices_ = []
    blurred_right_tof_indices_ = []
    blurred_noisy_values_ = []
    blurred_intensity_values_ = []
    intensity_buffer = np.zeros(tof_max_index)
    noisy_buffer = np.zeros(tof_max_index, dtype=np.uint8)
    for push_index, start in enumerate(connection_counts[:-1]):
        tofs = []
        end = connection_counts[push_index + 1]
        for connected_push_index, blur in zip(
            connections[start: end],
            scan_blur[start: end]
        ):
            for index in range(
                blurred_indptr[connected_push_index],
                blurred_indptr[connected_push_index + 1],
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
        if len(tofs) == 0:
            blurred_indptr_.append(len(blurred_right_tof_indices_))
            continue
        order = np.argsort(np.array(tofs))
        previous_tof = -np.inf
        summed_intensity = 0.
        noisy = True
        for index_order in order:
            tof = tofs[index_order]
            if (tof - previous_tof) != 1:
                blurred_left_tof_indices_.append(tof)
                if previous_tof != -np.inf:
                    blurred_right_tof_indices_.append(previous_tof)
                    blurred_intensity_values_.append(summed_intensity)
                    blurred_noisy_values_.append(noisy)
                summed_intensity = 0.
                noisy = True
            summed_intensity += intensity_buffer[tof]
            previous_tof = tof
            if noisy_buffer[tof] > noise_level:
                noisy = False
        blurred_right_tof_indices_.append(previous_tof)
        blurred_intensity_values_.append(summed_intensity)
        blurred_noisy_values_.append(noisy)
        for tof in tofs:
            intensity_buffer[tof] = 0
            noisy_buffer[tof] = 0
        blurred_indptr_.append(len(blurred_right_tof_indices_))
    return (
        np.array(blurred_indptr_),
        np.array(blurred_left_tof_indices_, dtype=np.uint32),
        np.array(blurred_right_tof_indices_, dtype=np.uint32),
        # np.array(blurred_tof_indices_, dtype=np.uint32),
        np.array(blurred_noisy_values_),
        # np.array(blurred_intensity_values_),
    )




@alphatims.utils.njit(cache=False, nogil=True)
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
            if connected_push_index <= push_index:
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


@alphatims.utils.njit(cache=False, nogil=True)
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


@alphatims.utils.njit(cache=False, nogil=True)
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
