"""A module to perform centroiding of AlphaTims data."""

import alphatims.utils
import numpy as np
import scipy.ndimage
import pandas as pd
import alpharaw.centroiding


@alphatims.utils.njit(nogil=True)
def merge_cyclic_pushes(
    cyclic_push_index,
    intensity_values,
    tof_indices,
    push_indptr,
    zeroth_frame,
    cycle_length,
    tof_max_index,
    scan_max_index,
    return_sparse=False,
):
    offset = scan_max_index * zeroth_frame + cyclic_push_index
    intensity_buffer = np.zeros(tof_max_index)
    tofs = []
    for push_index in range(offset, len(push_indptr) - 1, cycle_length):
        start = push_indptr[push_index]
        end = push_indptr[push_index + 1]
        for index in range(start, end):
            tof = tof_indices[index]
            intensity = intensity_values[index]
            if intensity_buffer[tof] == 0:
                tofs.append(tof)
            intensity_buffer[tof] += intensity
    tofs = np.array(tofs, dtype=tof_indices.dtype)
    if return_sparse:
        tofs = np.sort(tofs)
        intensity_buffer = intensity_buffer[tofs]
    return tofs, intensity_buffer


def smooth_buffer(
    intensity_buffer,
    smooth_window=100,
    gaussian_blur=5,
    normalize=True,
):
    smoothed_intensity_buffer = np.cumsum(intensity_buffer)
    smoothed_intensity_buffer = (
        smoothed_intensity_buffer[smooth_window::smooth_window] - smoothed_intensity_buffer[:-smooth_window:smooth_window]
    )
    if gaussian_blur > 0:
        smoothed_intensity_buffer = scipy.ndimage.gaussian_filter(smoothed_intensity_buffer, gaussian_blur)
    if normalize:
        max_smooth = np.max(smoothed_intensity_buffer)
        max_raw = np.max(intensity_buffer)
        smoothed_intensity_buffer *= max_raw / max_smooth
    return smoothed_intensity_buffer


def guesstimate_quad_settings(
    dia_data,
    smooth_window=100,
    gaussian_blur=5,
    percentile=50,
    regresion_mz_lower_cutoff=400,
    regresion_mz_upper_cutoff=1000,
):
    import tqdm
    dia_mz_cycle = np.empty_like(dia_data.dia_mz_cycle)
    for cyclic_push_index, (low_quad, high_quad) in tqdm.tqdm(
        enumerate(dia_data.dia_mz_cycle),
        total=len(dia_data.dia_mz_cycle)
    ):
        if (low_quad == -1) and (high_quad == -1):
            dia_mz_cycle[cyclic_push_index] = (low_quad, high_quad)
            continue
        tofs, intensity_buffer = merge_cyclic_pushes(
            cyclic_push_index=cyclic_push_index,
            intensity_values=dia_data.intensity_values,
            tof_indices=dia_data.tof_indices,
            push_indptr=dia_data.push_indptr,
            zeroth_frame=dia_data.zeroth_frame,
            cycle_length=len(dia_data.dia_mz_cycle),
            tof_max_index=dia_data.tof_max_index,
            scan_max_index=dia_data.scan_max_index
        )
        intensity_buffer = smooth_buffer(
            intensity_buffer,
            smooth_window=smooth_window,
            gaussian_blur=gaussian_blur,
            normalize=False,
        )
        low_quad_estimate, high_quad_estimate = find_border_indices(
            intensity_buffer,
            percentile / 100,
            low_quad,
            high_quad,
            dia_data.mz_values[
                smooth_window // 2: -smooth_window // 2: smooth_window
            ],
        )
        dia_mz_cycle[cyclic_push_index] = (
            low_quad_estimate,
            high_quad_estimate
        )
    predicted_dia_mz_cycle = predict_dia_mz_cycle(
        dia_mz_cycle,
        dia_data,
        regresion_mz_lower_cutoff,
        regresion_mz_upper_cutoff,
    )
    return dia_mz_cycle, predicted_dia_mz_cycle


def predict_dia_mz_cycle(
    dia_mz_cycle,
    dia_data,
    regresion_mz_lower_cutoff,
    regresion_mz_upper_cutoff,
):
    import sklearn.linear_model
    df = pd.DataFrame(
        {
            "set_lower": dia_data.dia_mz_cycle[:, 0],
            "set_upper": dia_data.dia_mz_cycle[:, 1],
            "detected_lower": dia_mz_cycle[:, 0],
            "detected_upper": dia_mz_cycle[:, 1],
            "frame": np.arange(len(dia_mz_cycle)) // dia_data.scan_max_index,
            "scan": np.arange(len(dia_mz_cycle)) % dia_data.scan_max_index,
        }
    )
    selected = regresion_mz_lower_cutoff < dia_data.dia_mz_cycle[:, 0]
    selected &= regresion_mz_lower_cutoff < dia_mz_cycle[:, 0]
    selected &= regresion_mz_lower_cutoff < dia_data.dia_mz_cycle[:, 1]
    selected &= regresion_mz_lower_cutoff < dia_mz_cycle[:, 1]
    selected &= dia_data.dia_mz_cycle[:, 0] < regresion_mz_upper_cutoff
    selected &= dia_mz_cycle[:, 0] < regresion_mz_upper_cutoff
    selected &= dia_data.dia_mz_cycle[:, 1] < regresion_mz_upper_cutoff
    selected &= dia_mz_cycle[:, 1] < regresion_mz_upper_cutoff
    df2 = df[selected]
    frame_reg_lower = {}
    frame_reg_upper = {}
    for frame in np.unique(df2.frame):
        frame_reg_lower[frame] = sklearn.linear_model.LinearRegression().fit(
            df2[df2.frame == frame].scan.values.reshape(-1, 1),
            df2[df2.frame == frame].detected_lower.values.reshape(-1, 1),
        )
        frame_reg_upper[frame] = sklearn.linear_model.LinearRegression().fit(
            df2[df2.frame == frame].scan.values.reshape(-1, 1),
            df2[df2.frame == frame].detected_upper.values.reshape(-1, 1),
        )
    predicted_upper = []
    predicted_lower = []
    for index, frame in enumerate(df.frame.values):
        if frame == 0:
            predicted_upper.append(-1)
            predicted_lower.append(-1)
            continue
        predicted_lower_ = frame_reg_lower[frame].predict(
            df.scan.values[index: index + 1].reshape(-1, 1)
        )
        predicted_upper_ = frame_reg_upper[frame].predict(
            df.scan.values[index: index + 1].reshape(-1, 1)
        )
        predicted_lower.append(predicted_lower_[0, 0])
        predicted_upper.append(predicted_upper_[0, 0])
    return np.vstack(
        [predicted_lower, predicted_upper]
    ).T


@alphatims.utils.njit(nogil=True)
def find_border_indices(
    intensity_buffer,
    percentile,
    low_quad,
    high_quad,
    mz_values,
):
    argmax = np.argmax(intensity_buffer)
    max_intensity = intensity_buffer[argmax]
    low_index = argmax
    high_index = argmax
    threshold_intensity = max_intensity * percentile
    while low_index >= 0:
        if intensity_buffer[low_index] > threshold_intensity:
            low_index -= 1
        else:
            break
    while high_index < len(intensity_buffer):
        if intensity_buffer[high_index] > threshold_intensity:
            high_index += 1
        else:
            break
    return mz_values[low_index], mz_values[high_index]


def animate_quad(
    dia_data,
    *,
    smooth_window=100,
    gaussian_blur=5,
    normalize=True,
    save_path=None,
    fps=30,
    figsize=(10, 6),
    dpi=80,
    predicted_dia_mz_cycle=None
):
    from matplotlib import pyplot as plt
    from matplotlib import animation

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(
        xlim=(dia_data.mz_min_value, dia_data.mz_max_value),
        ylim=(0, 1)
    )
    raw_plot, = ax.plot([], [], lw=1, color="lightgrey")
    smooth_plot, = ax.plot([], [], lw=2, color="orange")
    quad_left_line = ax.axvline(0, ls='-', color='red', lw=2, zorder=10)
    quad_right_line = ax.axvline(0, ls='-', color='red', lw=2, zorder=10)
    legend_data = [
        "raw signal",
        "semi_smooth signal",
        "set_quad_left",
        "set_quad_right",
        # "10th_percentile",
        # "25th_percentile",
        # "50th_percentile",
    ]
    if predicted_dia_mz_cycle is not None:
        predicted_quad_left_line = ax.axvline(
            0,
            ls='-',
            color='green',
            lw=2,
            zorder=10
        )
        predicted_quad_right_line = ax.axvline(
            0,
            ls='-',
            color='green',
            lw=2,
            zorder=10
        )
        legend_data += [
            "real_quad_left",
            "real_quad_right",
        ]
    # percentile10_line = ax.axhline(0, ls=':', color='black', lw=1, zorder=10)
    # percentile25_line = ax.axhline(0, ls=':', color='black', lw=1, zorder=10)
    # percentile50_line = ax.axhline(0, ls=':', color='black', lw=1, zorder=10)
    title = ax.text(
        0.5,
        1.05,
        "",
        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
        transform=ax.transAxes,
        ha="center"
    )
    ax.grid(
        visible=True,
        which="major",
        ls=':',
        color='black',
        lw=0.25,
        alpha=0.25,
        axis="y"
    )
    ax.legend(
        legend_data,
        loc="upper right"
    )
    max_intensity = 1
    ax.set_yticks(np.linspace(0, max_intensity, 5))

    # initialization function: plot the background of each frame
    def init():
        raw_plot.set_data([], [])
        smooth_plot.set_data([], [])
        title.set_text("")
        quad_left_line.set_xdata([])
        quad_right_line.set_xdata([])
        if predicted_dia_mz_cycle is not None:
            predicted_quad_left_line.set_xdata([])
            predicted_quad_right_line.set_xdata([])
        # percentile10_line.set_ydata([])
        # percentile25_line.set_ydata([])
        # percentile50_line.set_ydata([])
        max_intensity = 1
        ax.set_ylim(0, max_intensity)
        ax.set_yticks(np.linspace(0, max_intensity, 5))
        ax.grid(
            visible=True,
            which="major",
            ls=':',
            color='black',
            lw=0.25,
            alpha=0.25,
            axis="y"
        )
        return raw_plot,

    # animation function.  This is called sequentially
    def animate_internal(i):
        cyclic_push_index = i + dia_data.scan_max_index
    #     print(cyclic_push_index)
        quad_vals = dia_data.dia_mz_cycle[cyclic_push_index]
        tofs, intensity_buffer = merge_cyclic_pushes(
            cyclic_push_index=cyclic_push_index,
            intensity_values=dia_data.intensity_values,
            tof_indices=dia_data.tof_indices,
            push_indptr=dia_data.push_indptr,
            zeroth_frame=dia_data.zeroth_frame,
            cycle_length=len(dia_data.dia_mz_cycle),
            tof_max_index=dia_data.tof_max_index,
            scan_max_index=dia_data.scan_max_index
        )
        smoothed_intensity_buffer = smooth_buffer(
            intensity_buffer,
            smooth_window=smooth_window,
            gaussian_blur=gaussian_blur,
            normalize=normalize,
        )
        raw_plot.set_data(
            dia_data.mz_values,
            intensity_buffer,
        )
        smooth_plot.set_data(
            dia_data.mz_values[
                smooth_window // 2: -smooth_window // 2: smooth_window
            ],
            smoothed_intensity_buffer,
        )
        title.set_text(
            f"{dia_data.sample_name}\n"
            f"framegroup: {cyclic_push_index // dia_data.scan_max_index}, mobility: "
            f"{dia_data.mobility_values[cyclic_push_index % dia_data.scan_max_index]:,.3f}"
        )
        quad_left_line.set_xdata([quad_vals[0], quad_vals[0]])
        quad_right_line.set_xdata([quad_vals[1], quad_vals[1]])
        if predicted_dia_mz_cycle is not None:
            predicted_quad_vals = predicted_dia_mz_cycle[cyclic_push_index]
            predicted_quad_left_line.set_xdata(
                [predicted_quad_vals[0], predicted_quad_vals[0]]
            )
            predicted_quad_right_line.set_xdata(
                [predicted_quad_vals[1], predicted_quad_vals[1]]
            )
        max_intensity = np.max(smoothed_intensity_buffer)
        if not np.isfinite(max_intensity):
            max_intensity = 1
        # percentile10_line.set_ydata([max_intensity / 10, max_intensity / 10])
        # percentile25_line.set_ydata([max_intensity / 4, max_intensity / 4])
        # percentile50_line.set_ydata([max_intensity / 2, max_intensity / 2])
        ax.set_xlim(quad_vals[0] - 100, quad_vals[1] + 100)
        ax.set_ylim(0, max_intensity)
        ax.set_yticks(np.linspace(0, max_intensity, 5))
        return raw_plot,

    anim = animation.FuncAnimation(
        fig,
        animate_internal,
        init_func=init,
        blit=True,
        frames=len(dia_data.dia_mz_cycle) - dia_data.scan_max_index,
        interval=10
    )

    if save_path is not None:
        anim.save(save_path, fps=fps)  # , extra_args=['-vcodec', 'libx264'])

    return anim


@alphatims.utils.threadpool
def deconvolute_frame(
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
    # ) = alpharaw.centroiding.blur_cycle_in_rt_dimension(
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
    # ) = alpharaw.centroiding.blur_scans(
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


# @alphatims.utils.njit(nogil=True)
# def connect_pushes(
#     indptr,
#     tof_indices,
#     intensity_values,
#     cycle_length,
#     tof_tolerance,
#     scan_max_index,
# ):
#     connection_indptr = np.zeros(indptr[-1] + 1, dtype=np.int64)
#     first_connections = []
#     second_connections = []
#     frame_connections = []
#
#     for self_push_index, self_start in enumerate(indptr[:-1]):
#         self_end = indptr[self_push_index + 1]
#         self_frame = self_push_index // scan_max_index
#         if self_start == self_end:
#             continue
#         for other_push_index in range(
#             self_push_index + scan_max_index,
#             cycle_length,
#             scan_max_index
#         ):
#             other_start = indptr[other_push_index]
#             other_end = indptr[other_push_index + 1]
#             if other_start == other_end:
#                 continue
#             other_frame = other_push_index // scan_max_index
#             self_index = self_start
#             other_index = other_start
#             while (self_index < self_end) and (other_index < other_end):
#                 self_tof = tof_indices[self_index]
#                 other_tof = tof_indices[other_index]
#                 if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
#                     first_connections.append(self_index)
#                     second_connections.append(other_index)
#                     frame_connections.append(other_frame)
#                     first_connections.append(other_index)
#                     second_connections.append(self_index)
#                     frame_connections.append(self_frame)
#                     connection_indptr[self_index + 1] += 1
#                     connection_indptr[other_index + 1] += 1
#                 if self_tof < other_tof:
#                     self_index += 1
#                 else:
#                     other_index += 1
#     order = np.argsort(np.array(first_connections))
#     connection_indices = np.array(second_connections, dtype=np.int64)[order]
#     frame_indices = np.array(frame_connections, dtype=np.int8)[order]
#     return np.cumsum(connection_indptr), connection_indices, frame_indices


@alphatims.utils.njit(nogil=True)
def match_within_cycle(
    indptr,
    tof_indices,
    cycle_length,
    tof_tolerance,
    scan_max_index,
    dia_mz_cycle,
    include_fragments=True,
    include_precursors=False,
):
    # assignments = np.full(len(indptr) - 1, -1, dtype=np.int64)
    connections = np.arange(indptr[-1])

    for self_push_index, self_start in enumerate(indptr[:-scan_max_index]):
        is_precursor = (dia_mz_cycle[self_push_index, 0] == -1)
        if is_precursor:
            if not include_precursors:
                continue
        elif not include_fragments:
            continue
        self_end = indptr[self_push_index + 1]
        # self_frame = self_push_index // scan_max_index
        if self_start == self_end:
            continue
        for other_push_index in range(
            self_push_index + scan_max_index,
            cycle_length,
            scan_max_index
        ):
            is_precursor = (dia_mz_cycle[other_push_index, 0] == -1)
            if is_precursor:
                if not include_precursors:
                    continue
            elif not include_fragments:
                continue
            other_start = indptr[other_push_index]
            other_end = indptr[other_push_index + 1]
            if other_start == other_end:
                continue
            # other_frame = other_push_index // scan_max_index
            self_index = self_start
            other_index = other_start
            while (self_index < self_end) and (other_index < other_end):
                self_tof = tof_indices[self_index]
                other_tof = tof_indices[other_index]
                if (self_tof - tof_tolerance) <= other_tof <= (self_tof + tof_tolerance):
                    self_connection = connections[self_index]
                    loop_connection = self_connection
                    other_connection = connections[other_index]
                    already_set = False
                    while loop_connection != self_index:
                        if loop_connection == other_connection:
                            already_set = True
                        loop_connection = connections[loop_connection]
                    if not already_set:
                        connections[self_index] = other_connection
                        connections[other_index] = self_connection
                if self_tof < other_tof:
                    self_index += 1
                else:  # what if equal?
                    other_index += 1
    return connections


@alphatims.utils.njit(nogil=True)
def deconvolute_quad(
    loop_assignments,
    indptr,
    dia_mz_cycle
):
    low_quads = np.repeat(
        # np.repeat(
        #     np.arange(cycle_length / scan_max_index, dtype=np.uint8),
        #     scan_max_index,
        # ),
        dia_mz_cycle[:, 0],
        np.diff(indptr)
    )
    high_quads = np.repeat(
        # np.repeat(
        #     np.arange(cycle_length / scan_max_index, dtype=np.uint8),
        #     scan_max_index,
        # ),
        dia_mz_cycle[:, 1],
        np.diff(indptr)
    )
    deconvoluted_quad = np.full((indptr[-1], 2), -1, dtype=np.float32)
    for index, connection in enumerate(loop_assignments):
        if low_quads[index] == -1:
            continue
        if deconvoluted_quad[index, 0] != -1:
            continue
        selection = [index]
        lower_quad = low_quads[index]
        upper_quad = high_quads[index]
        while connection != index:
            if low_quads[connection] > lower_quad:
                lower_quad = low_quads[connection]
            if high_quads[connection] < upper_quad:
                upper_quad = high_quads[connection]
            selection.append(connection)
            connection = loop_assignments[connection]
        # if len(selection) < 4:
        #     continue
        for selected_index in selection:
            deconvoluted_quad[selected_index] = (lower_quad, upper_quad)
    return deconvoluted_quad


def merge_all(
    dia_data,
    smooth_window=100,
    gaussian_blur=5,
    percentile=50,
):
    import tqdm
    indptr = np.empty(len(dia_data.dia_mz_cycle) + 1, dtype=np.int64)
    total = 0
    indptr[0] = total
    merged_intensities = []
    merged_tofs = []
    for cyclic_push_index, (low_quad, high_quad) in tqdm.tqdm(
        enumerate(dia_data.dia_mz_cycle),
        total=len(dia_data.dia_mz_cycle)
    ):
        tofs, intensity_buffer = merge_cyclic_pushes(
            cyclic_push_index=cyclic_push_index,
            intensity_values=dia_data.intensity_values,
            tof_indices=dia_data.tof_indices,
            push_indptr=dia_data.push_indptr,
            zeroth_frame=dia_data.zeroth_frame,
            cycle_length=len(dia_data.dia_mz_cycle),
            tof_max_index=dia_data.tof_max_index,
            scan_max_index=dia_data.scan_max_index,
            return_sparse=True,
        )
        merged_intensities.append(intensity_buffer)
        merged_tofs.append(tofs)
        total += len(tofs)
        indptr[cyclic_push_index + 1] = total
    return (
        indptr,
        np.concatenate(merged_tofs),
        np.concatenate(merged_intensities),
    )


def calculate_stats(
    loop_assignments,
    indptr,
    tof_indices,
    intensity_values,
    mz_values,
    mobility_values,
    scan_max_index,
    dia_mz_cycle,
    return_fragments_only=True,
):
    push_indices = expand_indptr(indptr)
    frame_indptr = indptr[::scan_max_index]
    frame_indices = expand_indptr(frame_indptr)
    (
        assignment_indptr,
        assignment_indices,
    ) = group_assignments(loop_assignments)
    (
        frame_intensity_values,
        new_mz_values,
    ) = make_dense_group(
        assignment_indptr,
        assignment_indices,
        frame_indices,
        tof_indices,
        intensity_values,
        mz_values,
    )
    df = pd.DataFrame(
        {
            "mz_values": new_mz_values,
            "scan_indices": push_indices[
                assignment_indices[assignment_indptr[:-1]]
            ] % scan_max_index,
        }
    )
    df["mobility_values"] = mobility_values[df["scan_indices"]]
    consistent = calculate_consistency_of_df(
        frame_intensity_values,
        df["scan_indices"].values,
        dia_mz_cycle,
        scan_max_index,
    )
    df["consistent"] = consistent
    df["frame_count"] = 0

    for index in range(frame_intensity_values.shape[1]):
        df[f"frame_{index}_intensity"] = frame_intensity_values[:, index]
        in_frame = new_mz_values >= dia_mz_cycle[
            (df["scan_indices"].values + index * scan_max_index), 0
        ]
        df["frame_count"] += frame_intensity_values[:, index] > 0
        if index == 0:
            continue
        in_frame &= new_mz_values <= dia_mz_cycle[
            (df["scan_indices"].values + index * scan_max_index), 1
        ]
        df[f"frame_{index}_theoretical"] = in_frame

        # df[f"frame_{index}_drop_in"] = (~in_frame) & (frame_intensity_values[:, index] > 0)
        # df[f"frame_{index}_drop_out"] = (in_frame) & (frame_intensity_values[:, index] == 0)
        # df[f"frame_{index}_correct"] = ~(df[f"frame_{index}_drop_in"] | df[f"frame_{index}_drop_out"])
    if return_fragments_only:
        df = df[~((df.frame_count == 1) & (df.frame_0_intensity > 0))]
    return df


@alphatims.utils.njit(nogil=True)
def group_assignments(
    loop_assignments
):
    assignment_indices = []
    assignment_indptr = [0]
    already_set = np.zeros(len(loop_assignments), dtype=np.bool_)
    for index, assignment in enumerate(loop_assignments):
        if already_set[index]:
            continue
        assignment_indices.append(assignment)
        already_set[index] = True
        while index != assignment:
            already_set[assignment] = True
            assignment = loop_assignments[assignment]
            assignment_indices.append(assignment)
        assignment_indptr.append(len(assignment_indices))
    return np.array(assignment_indptr), np.array(assignment_indices)


@alphatims.utils.njit(nogil=True)
def expand_indptr(indptr):
    return np.repeat(
        np.arange(len(indptr) - 1),
        np.diff(indptr)
    )


@alphatims.utils.njit(nogil=True)
def make_dense_group(
    assignment_indptr,
    assignment_indices,
    frame_indices,
    tof_indices,
    intensity_values,
    mz_values,
):
    frame_intensity_values = np.zeros(
        (len(assignment_indptr) - 1, frame_indices[-1] + 1),
    )
    new_mz_values = np.empty(len(assignment_indptr) - 1)
    for index, start in enumerate(assignment_indptr[:-1]):
        end = assignment_indptr[index + 1]
        mz_sum = 0
        intensity_sum = 0
        for assignment_index in assignment_indices[start: end]:
            frame = frame_indices[assignment_index]
            intensity = intensity_values[assignment_index]
            frame_intensity_values[index, frame] = intensity
            intensity_sum += intensity
            tof = tof_indices[assignment_index]
            mz_value = mz_values[tof]
            mz_sum += intensity * mz_value
        new_mz_values[index] = mz_sum / intensity_sum
    return frame_intensity_values, new_mz_values


@alphatims.utils.njit(nogil=True)
def calculate_consistency_of_df(
    frame_intensity_values,
    scans,
    dia_mz_cycle,
    scan_max_index,
    precision=0.5,
):
    consistent = np.zeros(len(scans), dtype=np.bool_)
    # max_frame_index = frame_intensity_values.shape[1] - 1
    for index, scan in enumerate(scans):
        border_indptr = np.empty(frame_intensity_values.shape[1] * 2 - 1)
        # border_indptr[0] = -np.inf
        border_indptr[1] = np.inf
        # idx = 2
        idx = 1
        for lower, upper in dia_mz_cycle[scan + scan_max_index::scan_max_index]:
            border_indptr[idx] = lower
            border_indptr[idx + 1] = upper
            idx += 2
        border_indptr = np.unique(border_indptr)
        selection = np.diff(border_indptr) > precision
        border_indptr = border_indptr[:-1][selection]
        included = np.zeros(
            len(border_indptr) - 1,
            dtype=np.bool_
        )
        excluded = np.zeros(
            len(border_indptr) - 1,
            dtype=np.bool_
        )
        for frame_id, intensity in enumerate(
            frame_intensity_values[index, 1:],
            1
        ):
            lower, upper = dia_mz_cycle[scan + frame_id * scan_max_index]
            for border_index, border in enumerate(border_indptr[:-1]):
                if lower <= border < upper:
                    if intensity > 0:
                        included[border_index] = True
                    else:
                        excluded[border_index] = True
        if 1 <= sum(np.diff(included & ~excluded)) <= 2:
            consistent[index] = True
        # else:
        #     print(index)
        #     print(included)
        #     print(excluded)
        #     print(included & ~excluded)
        #     print(border_indptr)
        #     return consistent
    return consistent



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


@alphatims.utils.njit(nogil=True)
def calculate_consistency_matrix(
    dia_mz_cycle,
    scan_max_index,
    precision=1,
):
    consistency_matrix = np.full(
        (scan_max_index, 2**16),
        -1,
        # dtype=np.uint8
    )
    new_borders = []
    for scan_index in range(scan_max_index):
        border_indptr = []
        for lower, upper in dia_mz_cycle[scan_index::scan_max_index]:
            if lower > -1:
                border_indptr.append(lower)
                border_indptr.append(upper)
        border_indptr.append(np.inf)
        border_indptr = np.unique(np.array(border_indptr))
        selection = np.diff(border_indptr) > precision  # TODO, what if multiple close?
        border_indptr = border_indptr[:-1][selection]
        # print(len(border_indptr))#, np.diff(border_indptr))
        for border_index, border_lower in enumerate(border_indptr[:-1]):
            border_upper = border_indptr[border_index + 1]
            bit_reference = 0
            new_borders.append((border_lower, border_upper))
            for index, (scan_lower, scan_upper) in enumerate(
                dia_mz_cycle[scan_index::scan_max_index]
            ):
                if (scan_lower <= border_lower < scan_upper):
                    bit_reference += 2**(index + 1)
            consistency_matrix[scan_index, bit_reference] = len(new_borders) - 1
    return consistency_matrix, np.array(new_borders)




@alphatims.utils.njit(nogil=True)
def calculate_consistency(
    loop_indices,
    indptr,
    consistency_matrix,
    scan_max_index,
):
    push_indices = expand_indptr(indptr)
    border_indices = np.full_like(loop_indices, -1)
    for index, loop_index in enumerate(loop_indices):
        bit_number = 2**((push_indices[loop_index] // scan_max_index) + 1)
        while index != loop_index:
            loop_index = loop_indices[loop_index]
            bit_number |= 2**((push_indices[loop_index] // scan_max_index) + 1)
        scan_index = push_indices[loop_index] % scan_max_index
        border_indices[index] = consistency_matrix[scan_index, bit_number]
    return border_indices
