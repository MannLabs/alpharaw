"""This module provides functions to handle Bruker data.
It primarily implements the TimsTOF class, that acts as an in-memory container
for Bruker data storage.

Note: this code has been moved from the AlphaTims package and does not comply to the MSData_Base contract.
For the full functionality in terms of fast data access, please use the TimsTOF class from the AlphaTims package, which inherits from this TimsTOFBase class.
"""

import os
import logging

import alpharaw
import numpy as np
from alpharaw.bruker.dll import BRUKER_DLL_FILE_NAME, open_bruker_d_folder
from alpharaw.bruker.read import read_bruker_sql, read_bruker_binary


class TimsTOFBase(object):
    """A class that reads Bruker TimsTOF data.

    Data can be read directly from a Bruker .d folder.
    All OS's are supported,
    but reading mz_values and mobility_values from a .d folder
    requires Windows or Linux due to availability of Bruker libraries.
    On MacOS, they are estimated based on metadata,
    but these values are not guaranteed to be correct.
    Often they fall within 0.02 Th, but errors up to 6 Th have already
    been observed!
    """

    @property
    def sample_name(self):
        """: str : The sample name of this TimsTOF object."""
        file_name = os.path.basename(self.bruker_d_folder_name)
        return '.'.join(file_name.split('.')[:-1])

    @property
    def directory(self):
        """: str : The directory of this TimsTOF object."""
        return os.path.dirname(self.bruker_d_folder_name)


    @property
    def version(self):
        """: str : AlphaTims version used to create this TimsTOF object."""
        return self._version

    @property
    def acquisition_mode(self):
        """: str : The acquisition mode."""
        return self._acquisition_mode

    @property
    def meta_data(self):
        """: dict : The metadata for the acquisition."""
        return self._meta_data

    @property
    def rt_values(self):
        """: np.ndarray : np.float64[:] : The rt values."""
        return self._rt_values

    @property
    def mobility_values(self):
        """: np.ndarray : np.float64[:] : The mobility values."""
        return self._mobility_values

    @property
    def cycle(self):
        """: np.ndarray : np.float64[:,:,:,:] : The quad values."""
        return self._cycle

    @property
    def mz_values(self):
        """: np.ndarray : np.float64[:] : The mz values."""
        return self._mz_values

    @property
    def quad_mz_values(self):
        """: np.ndarray : np.float64[:, 2] : The (low, high) quad mz values."""
        return self._quad_mz_values

    @property
    def intensity_values(self):
        """: np.ndarray : np.uint16[:] : The intensity values."""
        return self._intensity_values

    @property
    def frame_max_index(self):
        """: int : The maximum frame index."""
        return self._frame_max_index

    @property
    def scan_max_index(self):
        """: int : The maximum scan index."""
        return self._scan_max_index

    @property
    def tof_max_index(self):
        """: int : The maximum tof index."""
        return self._tof_max_index

    @property
    def precursor_max_index(self):
        """: int : The maximum precursor index."""
        return self._precursor_max_index

    @property
    def mz_min_value(self):
        """: float : The minimum mz value."""
        return self.mz_values[0]

    @property
    def mz_max_value(self):
        """: float : The maximum mz value."""
        return self.mz_values[-1]

    @property
    def rt_max_value(self):
        """: float : The maximum rt value."""
        return self.rt_values[-1]

    @property
    def quad_mz_min_value(self):
        """: float : The minimum quad mz value."""
        return self._quad_min_mz_value

    @property
    def quad_mz_max_value(self):
        """: float : The maximum quad mz value."""
        return self._quad_max_mz_value

    @property
    def mobility_min_value(self):
        """: float : The minimum mobility value."""
        return self._mobility_min_value

    @property
    def mobility_max_value(self):
        """: float : The maximum mobility value."""
        return self._mobility_max_value

    @property
    def intensity_min_value(self):
        """: float : The minimum intensity value."""
        return self._intensity_min_value

    @property
    def intensity_max_value(self):
        """: float : The maximum intensity value."""
        return self._intensity_max_value

    @property
    def frames(self):
        """: pd.DataFrame : The frames table of the analysis.tdf SQL."""
        return self._frames

    @property
    def fragment_frames(self):
        """: pd.DataFrame : The fragment frames table."""
        return self._fragment_frames

    @property
    def precursors(self):
        """: pd.DataFrame : The precursor table."""
        return self._precursors

    @property
    def tof_indices(self):
        """: np.ndarray : np.uint32[:] : The tof indices."""
        return self._tof_indices

    @property
    def push_indptr(self):
        """: np.ndarray : np.int64[:] : The tof indptr."""
        return self._push_indptr

    @property
    def quad_indptr(self):
        """: np.ndarray : np.int64[:] : The quad indptr (tof_indices)."""
        return self._quad_indptr

    @property
    def raw_quad_indptr(self):
        """: np.ndarray : np.int64[:] : The raw quad indptr (push indices)."""
        return self._raw_quad_indptr

    @property
    def precursor_indices(self):
        """: np.ndarray : np.int64[:] : The precursor indices."""
        return self._precursor_indices

    @property
    def dia_precursor_cycle(self):
        """: np.ndarray : np.int64[:] : The precursor indices of a DIA cycle."""
        return self._dia_precursor_cycle

    @property
    def dia_mz_cycle(self):
        """: np.ndarray : np.float64[:, 2] : The mz_values of a DIA cycle."""
        return self._dia_mz_cycle

    @property
    def zeroth_frame(self):
        """: bool : A blank zeroth frame is present so frames are 1-indexed."""
        return self._zeroth_frame

    @property
    def max_accumulation_time(self):
        """: float : The maximum accumulation time of all frames."""
        return self._max_accumulation_time

    @property
    def accumulation_times(self):
        """: np.ndarray : The accumulation times of all frames."""
        return self._accumulation_times

    @property
    def intensity_corrections(self):
        """: np.ndarray : The intensity_correction per frame."""
        return self._intensity_corrections

    def __init__(
        self,
        bruker_d_folder_name: str,
        *,
        mz_estimation_from_frame: int = 1,
        mobility_estimation_from_frame: int = 1,
        drop_polarity: bool = True,
        convert_polarity_to_int: bool = True,
    ):
        """Create a Bruker TimsTOF object that contains all data in-memory.

        Parameters
        ----------
        bruker_d_folder_name : str
            The full file name to a Bruker .d folder.
            Alternatively, the full file name of an already exported .hdf
            can be provided as well.
        mz_estimation_from_frame : int
            If larger than 0, mz_values from this frame are read as
            default mz_values with the Bruker library.
            If 0, mz_values are being estimated with the metadata
            based on "MzAcqRangeLower" and "MzAcqRangeUpper".
            IMPORTANT NOTE: MacOS defaults to 0, as no Bruker library
            is available.
            Default is 1.
        mobility_estimation_from_frame : int
            If larger than 0, mobility_values from this frame are read as
            default mobility_values with the Bruker library.
            If 0, mobility_values are being estimated with the metadata
            based on "OneOverK0AcqRangeLower" and "OneOverK0AcqRangeUpper".
            IMPORTANT NOTE: MacOS defaults to 0, as no Bruker library
            is available.
            Default is 1.
        drop_polarity : bool
            The polarity column of the frames table contains "+" or "-" and
            is not numerical.
            If True, the polarity column is dropped from the frames table.
            this ensures a fully numerical pd.DataFrame.
            If False, this column is kept, resulting in a pd.DataFrame with
            dtype=object.
            Default is True.
        convert_polarity_to_int : bool
            Convert the polarity to int (-1 or +1).
            This allows to keep it in numerical form.
            This is ignored if the polarity is dropped.
            Default is True.
        """
        #Log a warning if there was not a valid DLL filename
        if BRUKER_DLL_FILE_NAME == "":
            logging.warning(
                "WARNING: "
                "No Bruker libraries are available for this operating system. "
                "Mobility and m/z values need to be estimated. "
                "While this estimation often returns acceptable results with errors "
                "< 0.02 Th, huge errors (e.g. offsets of 6 Th) have already been "
                "observed for some samples!"
            )
            logging.info("")

        self._load_data(bruker_d_folder_name,
                        mz_estimation_from_frame,
                        mobility_estimation_from_frame,
                        drop_polarity,
                        convert_polarity_to_int)

        logging.info(f"Successfully imported data from {bruker_d_folder_name}")

    def _load_data(self,
                   bruker_d_folder_name: str,
                   mz_estimation_from_frame: int,
                   mobility_estimation_from_frame: int,
                   drop_polarity: bool,
                   convert_polarity_to_int: bool) -> None:
        """Load data from disk."""

        if bruker_d_folder_name.endswith("/"):
            bruker_d_folder_name = bruker_d_folder_name[:-1]

        logging.info(f"Importing data from {bruker_d_folder_name}")
        if bruker_d_folder_name.endswith(".d"):

            self.bruker_d_folder_name = os.path.abspath(
                bruker_d_folder_name
            )
            self._import_data_from_d_folder(
                bruker_d_folder_name,
                mz_estimation_from_frame,
                mobility_estimation_from_frame,
                drop_polarity,
                convert_polarity_to_int,
            )
        else:
            raise NotImplementedError(
                "WARNING: file extension not understood. This class only supports import from .d folders. Use TimsTOF class from alphatims to enable import from .hdf files."
            )

    def __len__(self):
        return len(self.intensity_values)

    def __hash__(self):
        return hash(self.bruker_d_folder_name)

    def _import_data_from_d_folder(
        self,
        bruker_d_folder_name: str,
        mz_estimation_from_frame: int,
        mobility_estimation_from_frame: int,
        drop_polarity: bool = True,
        convert_polarity_to_int: bool = True,
    ):
        logging.info(f"Using .d import for {bruker_d_folder_name}")
        self._version = alpharaw.__version__
        self._zeroth_frame = True
        (
            self._acquisition_mode,
            global_meta_data,
            self._frames,
            self._fragment_frames,
            self._precursors,
            calibration_available,
        ) = read_bruker_sql(
            bruker_d_folder_name,
            self._zeroth_frame,
            drop_polarity,
            convert_polarity_to_int,
        )
        self._meta_data = dict(
            zip(global_meta_data.Key, global_meta_data.Value)
        )
        (
            self._push_indptr,
            self._tof_indices,
            self._intensity_values,
        ) = read_bruker_binary(
            self.frames,
            bruker_d_folder_name,
            int(self._meta_data["TimsCompressionType"]),
            int(self._meta_data["MaxNumPeaksPerScan"]),
        )
        logging.info(f"Indexing {bruker_d_folder_name}...")
        self._use_calibrated_mz_values_as_default = False
        self._frame_max_index = self.frames.shape[0]
        self._scan_max_index = int(self.frames.NumScans.max()) + 1
        self._tof_max_index = int(self.meta_data["DigitizerNumSamples"]) + 1
        self._rt_values = self.frames.Time.values.astype(np.float64)
        self._mobility_min_value = float(
            self.meta_data["OneOverK0AcqRangeLower"]
        )
        self._mobility_max_value = float(
            self.meta_data["OneOverK0AcqRangeUpper"]
        )
        self._accumulation_times = self.frames.AccumulationTime.values.astype(
            np.float64
        )
        self._max_accumulation_time = np.max(self._accumulation_times)
        self._intensity_corrections = self._max_accumulation_time / self._accumulation_times
        if (mobility_estimation_from_frame != 0) and calibration_available:
            import ctypes
            with open_bruker_d_folder(
                bruker_d_folder_name
            ) as (bruker_dll, bruker_d_folder_handle):
                logging.info(
                    f"Fetching mobility values from {bruker_d_folder_name}"
                )
                indices = np.arange(self.scan_max_index).astype(np.float64)
                self._mobility_values = np.empty_like(indices)
                bruker_dll.tims_scannum_to_oneoverk0(
                    bruker_d_folder_handle,
                    mobility_estimation_from_frame,
                    indices.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.mobility_values.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.scan_max_index
                )
        else:
            if (mobility_estimation_from_frame != 0):
                logging.info(
                    "Bruker DLL not available, estimating mobility values"
                )
            self._mobility_values = self.mobility_max_value - (
                self.mobility_max_value - self.mobility_min_value
            ) / self.scan_max_index * np.arange(self.scan_max_index)
        mz_min_value = float(self.meta_data["MzAcqRangeLower"])
        mz_max_value = float(self.meta_data["MzAcqRangeUpper"])
        if self.meta_data["AcquisitionSoftware"] == "Bruker otofControl":
            logging.warning(
                "WARNING: Acquisition software is Bruker otofControl, "
                "mz min/max values are assumed to be 5 m/z wider than "
                "defined in analysis.tdf!"
            )
            mz_min_value -= 5
            mz_max_value += 5
        tof_intercept = np.sqrt(mz_min_value)
        tof_slope = (
            np.sqrt(mz_max_value) - tof_intercept
        ) / self.tof_max_index
        if (mz_estimation_from_frame != 0) and calibration_available:
            import ctypes
            with open_bruker_d_folder(
                bruker_d_folder_name
            ) as (bruker_dll, bruker_d_folder_handle):
                logging.info(
                    f"Fetching mz values from {bruker_d_folder_name}"
                )
                indices = np.arange(self.tof_max_index).astype(np.float64)
                self._mz_values = np.empty_like(indices)
                bruker_dll.tims_index_to_mz(
                    bruker_d_folder_handle,
                    mz_estimation_from_frame,
                    indices.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self._mz_values.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.tof_max_index
                )
        else:
            if (mz_estimation_from_frame != 0):
                logging.info(
                    "Bruker DLL not available, estimating mz values"
                )
            self._mz_values = (
                tof_intercept + tof_slope * np.arange(self.tof_max_index)
            )**2
        self._parse_quad_indptr()
        self._intensity_min_value = int(np.min(self.intensity_values))
        self._intensity_max_value = int(np.max(self.intensity_values))
        if self.acquisition_mode == "diaPASEF":
            self.set_cycle()


    def _parse_quad_indptr(self) -> None:
        logging.info("Indexing quadrupole dimension")
        frame_ids = self.fragment_frames.Frame.values + 1
        scan_begins = self.fragment_frames.ScanNumBegin.values
        scan_ends = self.fragment_frames.ScanNumEnd.values
        isolation_mzs = self.fragment_frames.IsolationMz.values
        isolation_widths = self.fragment_frames.IsolationWidth.values
        precursors = self.fragment_frames.Precursor.values
        if (precursors[0] is None):
            if self.zeroth_frame:
                frame_groups = self.frames.MsMsType.values[1:]
            else:
                frame_groups = self.frames.MsMsType.values
            precursor_frames = np.flatnonzero(frame_groups == 0)
            group_sizes = np.diff(precursor_frames)
            group_size = group_sizes[0]
            if np.any(group_sizes != group_size):
                raise ValueError("Sample type not understood")
            precursors = (1 + frame_ids - frame_ids[0]) % group_size
            if self.zeroth_frame:
                precursors[0] = 0
            self.fragment_frames.Precursor = precursors
            self._acquisition_mode = "diaPASEF"
        scan_max_index = self.scan_max_index
        frame_max_index = self.frame_max_index
        quad_indptr = [0]
        quad_low_values = []
        quad_high_values = []
        precursor_indices = []
        high = -1
        for (
            frame_id,
            scan_begin,
            scan_end,
            isolation_mz,
            isolation_width,
            precursor
        ) in zip(
            frame_ids - 1,
            scan_begins,
            scan_ends,
            isolation_mzs,
            isolation_widths / 2,
            precursors
        ):
            low = frame_id * scan_max_index + scan_begin
            # TODO: CHECK?
            # if low < high:
            #     print(frame_id, low, frame_id * scan_max_index + scan_end, high, low - high)
            if low != high:
                quad_indptr.append(low)
                quad_low_values.append(-1)
                quad_high_values.append(-1)
                precursor_indices.append(0)
            high = frame_id * scan_max_index + scan_end
            quad_indptr.append(high)
            quad_low_values.append(isolation_mz - isolation_width)
            quad_high_values.append(isolation_mz + isolation_width)
            precursor_indices.append(precursor)
        quad_max_index = scan_max_index * frame_max_index
        if high < quad_max_index:
            quad_indptr.append(quad_max_index)
            quad_low_values.append(-1)
            quad_high_values.append(-1)
            precursor_indices.append(0)
        self._quad_mz_values = np.stack([quad_low_values, quad_high_values]).T
        self._precursor_indices = np.array(precursor_indices)
        self._raw_quad_indptr = np.array(quad_indptr)
        self._quad_indptr = self.push_indptr[self._raw_quad_indptr]
        self._quad_max_mz_value = np.max(self.quad_mz_values[:, 1])
        self._quad_min_mz_value = np.min(
            self.quad_mz_values[
                self.quad_mz_values[:, 0] >= 0,
                0
            ]
        )
        self._precursor_max_index = int(np.max(self.precursor_indices)) + 1
        if self._acquisition_mode == "diaPASEF":
            offset = int(self.zeroth_frame)
            cycle_index = np.searchsorted(
                self.raw_quad_indptr,
                (self.scan_max_index) * (self.precursor_max_index + offset),
                "r"
            ) + 1
            repeats = np.diff(self.raw_quad_indptr[: cycle_index])
            if self.zeroth_frame:
                repeats[0] -= self.scan_max_index
            cycle_length = self.scan_max_index * self.precursor_max_index
            repeat_length = np.sum(repeats)
            if repeat_length != cycle_length:
                repeats[-1] -= repeat_length - cycle_length
            self._dia_mz_cycle = np.empty((cycle_length, 2))
            self._dia_mz_cycle[:, 0] = np.repeat(
                self.quad_mz_values[: cycle_index - 1, 0],
                repeats
            )
            self._dia_mz_cycle[:, 1] = np.repeat(
                self.quad_mz_values[: cycle_index - 1, 1],
                repeats
            )
            self._dia_precursor_cycle = np.repeat(
                self.precursor_indices[: cycle_index - 1],
                repeats
            )
        else:
            self._dia_mz_cycle = np.empty((0, 2))
            self._dia_precursor_cycle = np.empty(0, dtype=np.int64)


    def set_cycle(self) -> None:
        """Set the quad cycle for diaPASEF data.
        """
        ms1_diffs = np.diff(
            np.flatnonzero(self.frames.MsMsType[int(self.zeroth_frame):]==0)
        )
        subcycle_length_count = np.bincount(ms1_diffs)
        if np.all(subcycle_length_count[:-1]!=0):
            raise ValueError("No consistent subcycle length")
        subcycle_length = len(subcycle_length_count) - 1
        max_precursor = len(self.fragment_frames.Precursor.unique())
        subcycle_count = max_precursor // (subcycle_length - 1)
        frame_count = subcycle_length * subcycle_count
        cycle = np.zeros(
            (
                frame_count,
                self.scan_max_index,
                2,
            )
        )
        precursor_frames = np.ones(frame_count, dtype=np.bool_)

        subframes = self.fragment_frames.drop("Frame", axis=1)
        for max_index in range(1, len(subframes)):
            subframe = subframes.iloc[max_index]
            if subframe.equals(subframes.iloc[0]):
                break
        for index, row in self.fragment_frames[:max_index].iterrows():
            frame = int(row.Frame - self.zeroth_frame)
            scan_begin = int(row.ScanNumBegin)
            scan_end = int(row.ScanNumEnd)
            low_mz = row.IsolationMz - row.IsolationWidth / 2
            high_mz = row.IsolationMz + row.IsolationWidth / 2
            cycle[
                frame,
                scan_begin: scan_end,
            ] = (low_mz, high_mz)
            precursor_frames[frame] = False

        cycle[precursor_frames] = (-1, -1)
        cycle = cycle.reshape(
            (
                subcycle_count,
                subcycle_length,
                *cycle.shape[1:]
            )
        )
        self._cycle = cycle


    def use_calibrated_mz_values_as_default(
        self,
        use_calibrated_mz_values: int
    ) -> None:
        """Override the default mz_values with the global calibrated_mz_values."""
        raise NotImplementedError("Not implemented for TimsTOFBase. Use TimsTOF class from alphatims to enable use_calibrated_mz_values_as_default.")

    def _import_data_from_hdf_file(
        self,
        bruker_d_folder_name: str,
    ):
        raise NotImplementedError("Not implemented for TimsTOFBase. Use TimsTOF class from alphatims to enable import_data_from_hdf_file.")