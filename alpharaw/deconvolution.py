"""A module to perfortm centroiding of AlphaTims data."""

import alphatims.utils
import numpy as np
import scipy.ndimage
import pandas as pd


@alphatims.utils.njit
def merge_cyclic_pushes(
    cyclic_push_index,
    intensity_values,
    tof_indices,
    push_indptr,
    zeroth_frame,
    cycle_length,
    tof_max_index,
    scan_max_index,
):
    offset = scan_max_index * zeroth_frame + cyclic_push_index
    intensity_buffer = np.zeros(tof_max_index)
    for push_index in range(offset, len(push_indptr) - 1, cycle_length):
        start = push_indptr[push_index]
        end = push_indptr[push_index + 1]
        for index in range(start, end):
            tof = tof_indices[index]
            intensity = intensity_values[index]
            intensity_buffer[tof] += intensity
    return intensity_buffer


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
    lower_cutoff=400,
    upper_cutoff=1000,
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
        intensity_buffer = merge_cyclic_pushes(
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
        lower_cutoff,
        upper_cutoff,
    )
    return dia_mz_cycle, predicted_dia_mz_cycle


def predict_dia_mz_cycle(
    dia_mz_cycle,
    dia_data,
    lower_cutoff,
    upper_cutoff,
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
    selected = lower_cutoff < dia_data.dia_mz_cycle[:, 0]
    selected &= lower_cutoff < dia_mz_cycle[:, 0]
    selected &= lower_cutoff < dia_data.dia_mz_cycle[:, 1]
    selected &= lower_cutoff < dia_mz_cycle[:, 1]
    selected &= dia_data.dia_mz_cycle[:, 0] < upper_cutoff
    selected &= dia_mz_cycle[:, 0] < upper_cutoff
    selected &= dia_data.dia_mz_cycle[:, 1] < upper_cutoff
    selected &= dia_mz_cycle[:, 1] < upper_cutoff
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


@alphatims.utils.njit
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
        intensity_buffer = merge_cyclic_pushes(
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
