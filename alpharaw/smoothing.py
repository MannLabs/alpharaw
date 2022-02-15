"""A module to perform smoothing of TOF data."""

import alphatims.utils
import numpy as np
# import scipy.ndimage
# import pandas as pd


def smooth(
    dia_data,
    # hdf_file,
    scan_tolerance=6,
    scan_sigma=2,
    multiple_frames_per_cycle=False,
    ms1=True,
    ms2=False,
    cycle_tolerance=3,
    tof_tolerance=3,
    intensity_cutoff=0,
    noise_level=4,
    thread_count=None,
):
    import multiprocessing.pool
    import h5py
    import tempfile
    import os
    cycle_count = len(dia_data.push_indptr) // len(dia_data.dia_mz_cycle)
    cycle_length = len(dia_data.dia_mz_cycle)
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

    def starfunc(iterable):
        return smoothen_frame(
            iterable,
            dia_data,
            cycle_tolerance,
            connection_counts,
            connections,
            scan_blur,
            tof_tolerance,
            intensity_cutoff,
            neighbor_type=(2 + 4 + 8 + 16),
        )

    if thread_count is None:
        current_thread_count = alphatims.utils.MAX_THREADS
    else:
        current_thread_count = alphatims.utils.set_threads(
            thread_count,
            set_global=False
        )
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_file_name = f"{dia_data.sample_name}_temp_smooth.hdf"
        # temp_file_name = os.path.join(
        #     temp_dir_name,
        #     f"{dia_data.sample_name}_temp_smooth.hdf"
        # )
        intensities = []
        tof_indices = []
        neighbor_types = []
        offsets = [0]
        indptr = np.empty(
            ((cycle_count + 1) * cycle_length + 1,),
            dtype=np.int64,
        )
        iterable = range(cycle_count + 1)
        with h5py.File(temp_file_name, 'w') as temp_hdf_file:
            start = 0
            indptr_offset = 0
            indptr_start = 0
            with multiprocessing.pool.ThreadPool(current_thread_count) as pool:
                for cycle_index, (
                    mz_centroided_indptr,
                    mz_centroided_tof_indices,
                    mz_centroided_noisy_values,
                    mz_centroided_intensity_values,
                ) in alphatims.utils.progress_callback(
                    enumerate(pool.imap(starfunc, iterable)),
                    total=len(iterable),
                    include_progress_callback=True
                ):
                    end = start + len(mz_centroided_intensity_values)
                    indptr_end = indptr_start + len(mz_centroided_indptr)
                    tof_indices_ = temp_hdf_file.create_dataset(
                        f"mz_centroided_tof_indices_{cycle_index}",
                        data=mz_centroided_tof_indices
                    )
                    neighbor_types_ = temp_hdf_file.create_dataset(
                        f"mz_centroided_noisy_values_{cycle_index}",
                        data=mz_centroided_noisy_values
                    )
                    intensities_ = temp_hdf_file.create_dataset(
                        f"mz_centroided_intensity_values_{cycle_index}",
                        data=mz_centroided_intensity_values
                    )
                    tof_indices.append(tof_indices_)
                    neighbor_types.append(neighbor_types_)
                    intensities.append(intensities_)
                    mz_centroided_indptr += indptr_offset
                    indptr[indptr_start: indptr_end] = mz_centroided_indptr
                    indptr_start = indptr_end - 1
                    indptr_offset = mz_centroided_indptr[-1]
                    start = end
                    offsets.append(end)
            dia_data2 = alphatims.bruker.TimsTOF(
                os.path.join(
                    dia_data.directory,
                    f"{dia_data.sample_name}.hdf",
                ),
                mmap_detector_events=True,
            )
            dia_data2._push_indptr[
                dia_data.scan_max_index: len(dia_data.push_indptr)
            ] = indptr[:len(dia_data.push_indptr) - dia_data.scan_max_index]
            dia_data2._quad_indptr = dia_data2.push_indptr[dia_data2.raw_quad_indptr]
            del dia_data2._tof_indices
            del dia_data2._intensity_values
            hdf_file_name = dia_data2.save_as_hdf(
                directory="/Users/swillems/Documents/software/alphadia/nbs",
                file_name=f"{dia_data2.sample_name}_smoothed.hdf",
                overwrite=True,
            )
            with h5py.File(hdf_file_name, 'a') as hdf_file:
                intensities_ = hdf_file.create_dataset(
                    "raw/_intensity_values",
                    (end,),
                    dtype=np.uint16
                )
                tof_indices_ = hdf_file.create_dataset(
                    "raw/_tof_indices",
                    (end,),
                    dtype=np.uint32
                )
                for index, start in alphatims.utils.progress_callback(
                    enumerate(offsets[:-1]),
                    total=len(offsets) - 1,
                    include_progress_callback=True
                ):
                    end = offsets[index + 1]
                    intensities_[start:end] = intensities[index][...]
                    tof_indices_[start:end] = tof_indices[index][...]

    # iterable = range(cycle_count + 1)
    # # iterable = range(300, 310)
    # indptr = hdf_file.create_dataset(
    #     "indptr",
    #     ((cycle_count + 1) * cycle_length + 1,),
    #     dtype=np.int64,
    # )
    # intensities_ = hdf_file.create_dataset(
    #     "intensities_",
    #     (len(dia_data) * (1 + scan_tolerance) * (1 + cycle_tolerance),),
    #     dtype=np.float32,
    # )
    # mz_values_ = hdf_file.create_dataset(
    #     "mz_values_",
    #     (len(dia_data) * (1 + scan_tolerance) * (1 + cycle_tolerance),),
    #     dtype=np.float32,
    # )
    # neighbor_type_ = hdf_file.create_dataset(
    #     "neighbor_type_",
    #     (len(dia_data) * (1 + scan_tolerance) * (1 + cycle_tolerance),),
    #     np.uint8
    # )
    # start = 0
    # indptr_offset = 0
    # indptr_start = 0
    # with multiprocessing.pool.ThreadPool(current_thread_count) as pool:
    #     for (
    #         mz_centroided_indptr,
    #         mz_centroided_tof_indices,
    #         mz_centroided_noisy_values,
    #         mz_centroided_intensity_values,
    #     ) in alphatims.utils.progress_callback(
    #         pool.imap(starfunc, iterable),
    #         total=len(iterable),
    #         include_progress_callback=True
    #     ):
    #         end = start + len(mz_centroided_intensity_values)
    #         indptr_end = indptr_start + len(mz_centroided_indptr)
    #         mz_values_[start: end] = mz_centroided_tof_indices
    #         neighbor_type_[start: end] = mz_centroided_noisy_values
    #         intensities_[start: end] = mz_centroided_intensity_values
    #         mz_centroided_indptr += indptr_offset
    #         indptr[indptr_start: indptr_end] = mz_centroided_indptr
    #         indptr_start = indptr_end - 1
    #         indptr_offset = mz_centroided_indptr[-1]
    #         start = end
    # hdf_file.create_dataset(
    #     "intensities",
    #     data=hdf_file["intensities_"][:end]
    # )
    # del hdf_file["intensities_"]
    # hdf_file.create_dataset(
    #     "mz_values",
    #     data=hdf_file["mz_values_"][:end]
    # )
    # del hdf_file["mz_values_"]
    # hdf_file.create_dataset(
    #     "neighbor_type",
    #     data=hdf_file["neighbor_type_"][:end]
    # )
    # del hdf_file["neighbor_type_"]


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
    """Determine how individual pushes in a cycle are connected.

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


def smoothen_frame(
    cycle_index,
    dia_data,
    cycle_tolerance,
    connection_counts,
    connections,
    scan_blur,
    tof_tolerance,
    intensity_cutoff,
    neighbor_type,
):
    (
        rt_blurred_indptr,
        rt_blurred_tof_indices,
        rt_neighbor_values,
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
    # return (
    #     rt_blurred_indptr,
    #     rt_blurred_tof_indices,
    #     rt_neighbor_values,
    #     rt_blurred_intensity_values,
    # )
    (
        mz_centroided_indptr,
        mz_centroided_tof_indices,
        mz_centroided_noisy_values,
        mz_centroided_intensity_values,
    ) = blur_scans(
        rt_blurred_indptr,
        rt_blurred_tof_indices,
        rt_neighbor_values,
        rt_blurred_intensity_values,
        connection_counts,
        connections,
        scan_blur,
        dia_data.tof_max_index,
        tof_tolerance,
        intensity_cutoff,
        neighbor_type,
    )
    return (
        mz_centroided_indptr,
        mz_centroided_tof_indices,
        mz_centroided_noisy_values,
        mz_centroided_intensity_values,
    )



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
    neighbor_buffer = np.zeros(tof_max_index, dtype=np.uint8)
    indptr_ = [0]
    tof_indices_ = []
    intensity_values_ = []
    neighbor_values = []
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
            if other_push_index < self_push_index:
                neighbor = 2
            elif other_push_index > self_push_index:
                neighbor = 4
            else:
                neighbor = 1
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
                neighbor_buffer[tof] |= neighbor
        for tof in tofs:
            tof_indices_.append(tof)
            neighbor_values.append(neighbor_buffer[tof])
            intensity_values_.append(intensity_buffer[tof])
            intensity_buffer[tof] = 0
            neighbor_buffer[tof] = 0
        indptr_.append(len(tof_indices_))
    return (
        np.array(indptr_, dtype=np.int64),
        np.array(tof_indices_, dtype=np.uint32),
        np.array(neighbor_values, dtype=np.uint8),
        np.array(intensity_values_, dtype=np.float32),
    )


@alphatims.utils.njit(nogil=True)
def blur_scans(
    indptr,
    tof_indices,
    neighbor_values,
    intensity_values,
    connection_counts,
    connections,
    scan_blur,
    tof_max_index,
    tof_tolerance,
    intensity_cutoff,
    neighbor_type,
):
    tof_blur = np.exp(
        -(np.arange(-tof_tolerance, tof_tolerance + 1))**2 / 2
    )
    indptr_ = [0]
    tof_indices_ = []
    neighbor_values_ = []
    intensity_values_ = []
    intensity_buffer = np.zeros(tof_max_index)
    neighbor_buffer = np.zeros(tof_max_index, dtype=np.uint8)
    for self_push_index, start in enumerate(connection_counts[:-1]):
        tofs = []
        end = connection_counts[self_push_index + 1]
        for other_push_index, blur in zip(
            connections[start: end],
            scan_blur[start: end]
        ):
            if other_push_index < self_push_index:
                neighbor = 8
                internal_im = False
            elif other_push_index > self_push_index:
                neighbor = 16
                internal_im = False
            else:
                neighbor = 0
                internal_im = True
            for index in range(
                indptr[other_push_index],
                indptr[other_push_index + 1],
            ):
                tof = tof_indices[index]
                intensity = intensity_values[index] * blur
                neighbor |= neighbor_values[index]
                internal_rt = (neighbor & 1)
                if internal_rt:
                    neighbor -= 1
                for index, tof_offset in enumerate(
                    range(-tof_tolerance, tof_tolerance + 1)
                ):
                    new_tof = tof + tof_offset
                    if not (0 <= new_tof < tof_max_index):
                        continue
                    if intensity_buffer[new_tof] == 0:
                        tofs.append(new_tof)
                    neighbor_buffer[new_tof] |= neighbor
                    intensity_buffer[new_tof] += intensity * tof_blur[index]
                    if tof_offset < 0:
                        neighbor_buffer[new_tof] |= 32
                    elif tof_offset > 0:
                        neighbor_buffer[new_tof] |= 64
                    elif (internal_rt and internal_im):
                        neighbor_buffer[new_tof] |= 1
        tof_maxima = []
        tof_maxima_intensities = []
        for tof in tofs:
            if (neighbor_buffer[tof] & neighbor_type) != neighbor_type:
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
            # if True:
            #     summed_intensity = intensity_buffer[tof]
                # TODO expand summed intensity beyond tof_tolerance?
                tof_maxima.append(tof)
                tof_maxima_intensities.append(summed_intensity)
        for index in np.argsort(np.array(tof_maxima)):
            tof = tof_maxima[index]
            summed_intensity = tof_maxima_intensities[index]
            tof_indices_.append(tof)
            neighbor_values_.append(neighbor_buffer[tof])
            intensity_values_.append(summed_intensity)

        for tof in tofs:
            intensity_buffer[tof] = 0
            neighbor_buffer[tof] = 0
        indptr_.append(len(tof_indices_))
    return (
        np.array(indptr_),
        np.array(tof_indices_, dtype=np.uint32),
        np.array(neighbor_values_, dtype=np.uint8),
        np.array(intensity_values_, dtype=np.float32),
    )
