import alphatims
import numpy as np
import pandas as pd
from alphatims.bruker import TimsTOF

from ..ms_data_base import MSData_Base


class AlphaTimsReader(MSData_Base):
    """
    TimsTOF data are too large, do not use this class
    """

    def import_raw(self, burker_d_folder: str):
        tims = TimsTOF(
            burker_d_folder,
        )

        self.raw_file_path = burker_d_folder
        self.file_type = "bruker"

        self.spectrum_df["precursor_mz"] = tims.fragment_frames.IsolationMz.values
        isolations = tims.fragment_frames.IsolationWidth.values / 2
        self.spectrum_df["isolation_lower_mz"] = (
            self.spectrum_df.precursor_mz - isolations
        )
        self.spectrum_df["isolation_upper_mz"] = (
            self.spectrum_df.precursor_mz + isolations
        )
        self.spectrum_df["peak_start_idx"] = tims._push_indptr[:-1]
        self.spectrum_df["peak_stop_idx"] = tims._push_indptr[1:]
        self.spectrum_df["rt"] = tims.rt_values / 60
        self.spectrum_df["mobility"] = tims.mobility_values
        self.spectrum_df["ms_level"] = (tims.precursor_indices > 0).astype(np.int8) + 1
        self.spectrum_df["spec_idx"] = self.spectrum_df.index.values
        self.spectrum_df["tims_frame"] = (
            self.spectrum_df.spec_idx // tims.scan_max_index
        )
        self.spectrum_df["tims_scan"] = self.spectrum_df.spec_idx % tims.scan_max_index

        self.peak_df["mz"] = tims.mz_values[tims.tof_indices]
        self.peak_df["intensity"] = tims.intensity_values.astype(np.float32)


class AlphaTimsWrapper(TimsTOF):
    """Create a AlphaTims object containing
    all data in-memory (or memory mapping).

    Attribute
    ---------
    slice_as_dataframe
        Attribute from AlphaTims.
        If True, AlphaTims slicing returns a pd.DataFrame by default.
        If False, AlphaTims slicing provides a np.int64[:] with raw indices.
        The value can be modified on-the-fly.
    """

    def __init__(self, msdata: MSData_Base, dda: bool, slice_as_dataframe: bool = True):
        """
        Parameters
        ----------
        msdata : MSData_Base
            The AlphaRaw data object.

        dda : bool
            If DDA, precursor indices will be equal to scan numbers.
            If not DDA (i.e. DIA), precursor indices will be equal to the
            scan number within a DIA cycle.

        slice_as_dataframe : bool
            If True, slicing returns a pd.DataFrame by default.
            If False, slicing provides a np.int64[:] with raw indices.
            Default is True.
        """
        self._use_calibrated_mz_values_as_default = False
        self._import_alpharaw_object(msdata, dda)
        self.thermo_raw_file_name = msdata.raw_file_path
        self.bruker_d_folder_name = self.thermo_raw_file_name
        self.slice_as_dataframe = (
            slice_as_dataframe  # This value can be modified after creation.
        )
        # Precompile
        self[0, "raw"]

    def _import_alpharaw_object(
        self,
        msdata: MSData_Base,
        dda: bool,
    ):
        self._version = alphatims.__version__
        mz_values = msdata.peak_df.mz.values
        self._intensity_values = msdata.peak_df.intensity.values

        if (msdata.spectrum_df.peak_start_idx < 0).any():
            spectrum_df = msdata.spectrum_df.query("peak_start_idx!=-1")
        else:
            spectrum_df = msdata.spectrum_df

        self._push_indptr = np.zeros(len(spectrum_df) + 1, dtype=np.int64)
        self._push_indptr[1:] = spectrum_df.peak_stop_idx.values
        self._rt_values = spectrum_df.rt.values * 60
        self._quad_mz_values = spectrum_df[
            ["isolation_lower_mz", "isolation_upper_mz"]
        ].values
        if dda:
            self._precursor_indices = np.zeros_like(self._rt_values, dtype=np.int64)
            ms2s = spectrum_df.ms_level.values == 2
            self._precursor_indices[ms2s] = np.cumsum(ms2s, dtype=np.int64)[ms2s]
        else:
            precursor_indices = []
            prev_mz = -1
            prev_idx = 0
            for mz, ms_level in spectrum_df[["precursor_mz", "ms_level"]].values:
                if ms_level == 1:
                    precursor_indices.append(0)
                elif prev_mz >= mz:  # TODO if DIA mz windows are not in order
                    prev_mz = mz
                    prev_idx = 1
                    precursor_indices.append(prev_idx)
                else:
                    prev_idx += 1
                    prev_mz = mz
                    precursor_indices.append(prev_idx)
            self._precursor_indices = np.array(precursor_indices, dtype=np.int64)

        scan_count = len(self._precursor_indices)
        self._frame_max_index = scan_count
        self._scan_max_index = 1
        self._mobility_max_value = 0.0
        self._mobility_min_value = 0.0
        self._mobility_values = np.array([0.0])
        self._quad_indptr = self._push_indptr
        self._raw_quad_indptr = np.arange(scan_count + 1)
        self._intensity_min_value = float(np.min(self._intensity_values))
        self._intensity_max_value = float(np.max(self._intensity_values))
        self._intensity_corrections = np.ones(self._frame_max_index)
        _q_mzs = self._quad_mz_values[self._quad_mz_values != -1]
        if len(_q_mzs) > 0:
            self._quad_min_mz_value = float(np.min(_q_mzs))
            self._quad_max_mz_value = float(np.max(_q_mzs))
        else:
            self._quad_min_mz_value = 0
            self._quad_max_mz_value = 0
        self._precursor_max_index = int(np.max(self._precursor_indices)) + 1
        self._acquisition_mode = (
            msdata.file_type + " " + ("DDA" if dda else "DIA")
        )  # TODO
        self._mz_min_value = int(np.min(mz_values))
        self._mz_max_value = int(np.max(mz_values)) + 1
        self._decimals = 4
        self._mz_values = (
            np.arange(
                10**self._decimals * self._mz_min_value,
                10**self._decimals * (self._mz_max_value + 1),
            )
            / 10**self._decimals
        )
        self._tof_indices = (mz_values * 10**self._decimals).astype(
            np.int32
        ) - 10**self._decimals * self._mz_min_value
        self._tof_max_index = len(self._mz_values)
        self._meta_data = {"SampleName": msdata.raw_file_path}
        msmstype = np.array([0 if s == -1 else 1 for s, e in self._quad_mz_values])
        summed_intensities_ = np.cumsum(self._intensity_values)
        summed_intensities = -summed_intensities_[self._push_indptr[:-1]]
        summed_intensities[:-1] += summed_intensities_[self._push_indptr[1:-1]]
        summed_intensities[-1] += summed_intensities_[-1]
        max_intensities = [
            np.max(
                self._intensity_values[self._push_indptr[i] : self._push_indptr[i + 1]]
            )
            if self._push_indptr[i + 1] != -1
            and self._push_indptr[i] != -1
            and self._push_indptr[i] != self._push_indptr[i + 1]
            else 0
            for i in range(len(self._rt_values))
        ]
        self._frames = pd.DataFrame(
            {
                "MsMsType": msmstype,
                "Time": self._rt_values,
                "SummedIntensities": summed_intensities,
                "MaxIntensity": max_intensities,
                "Id": np.arange(len(self._rt_values)),
            }
        )
        frame_numbers = np.arange(len(self._rt_values), dtype=np.int32)
        isolation_widths = self._quad_mz_values[:, 1] + self._quad_mz_values[:, 0]
        isolation_centers = self._quad_mz_values[:, 1] - self._quad_mz_values[:, 0]
        self._fragment_frames = pd.DataFrame(
            {
                "Frame": frame_numbers[msmstype == 1],
                "ScanNumBegin": 0,
                "ScanNumEnd": 0,
                "IsolationWidth": isolation_widths[msmstype == 1],
                "IsolationMz": isolation_centers[msmstype == 1],
                "Precursor": self._precursor_indices[msmstype == 1],
            }
        )
        self._zeroth_frame = False
        offset = int(self.zeroth_frame)
        cycle_index = (
            np.searchsorted(
                self.raw_quad_indptr,
                (self.scan_max_index) * (self.precursor_max_index + offset),
                "r",
            )
            + 1
        )
        repeats = np.diff(self.raw_quad_indptr[:cycle_index])
        if self.zeroth_frame:
            repeats[0] -= self.scan_max_index
        cycle_length = self.scan_max_index * self.precursor_max_index
        repeat_length = np.sum(repeats)
        if repeat_length != cycle_length:
            repeats[-1] -= repeat_length - cycle_length
        self._dia_mz_cycle = np.empty((cycle_length, 2))
        self._dia_mz_cycle[:, 0] = np.repeat(
            self.quad_mz_values[: cycle_index - 1, 0], repeats
        )
        self._dia_mz_cycle[:, 1] = np.repeat(
            self.quad_mz_values[: cycle_index - 1, 1], repeats
        )
        self._dia_precursor_cycle = np.repeat(
            self.precursor_indices[: cycle_index - 1], repeats
        )
