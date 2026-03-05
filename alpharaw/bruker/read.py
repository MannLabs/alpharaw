# builtin
import os
import sys
import contextlib
import logging
# external
import numpy as np
import pandas as pd
import h5py
# local
import alphatims
import alphatims.utils
import alphatims.tempmmap as tm
from alpharaw.bruker.dll import BRUKER_DLL_FILE_NAME, open_bruker_d_folder


def read_bruker_sql(
    bruker_d_folder_name: str,
    add_zeroth_frame: bool = True,
    drop_polarity: bool = True,
    convert_polarity_to_int: bool = True,
) -> tuple:
    """Read metadata, (fragment) frames and precursors from a Bruker .d folder.

    Parameters
    ----------
    bruker_d_folder_name : str
        The name of a Bruker .d folder.
    add_zeroth_frame : bool
        Bruker uses 1-indexing for frames.
        If True, a zeroth frame is added without any TOF detections to
        make Python simulate this 1-indexing.
        If False, frames are 0-indexed.
        Default is True.
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

    Returns
    -------
    : tuple
        (str, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool).
        The acquisition_mode, global_meta_data, frames, fragment_frames,
        precursors and calibration availability.
        For diaPASEF, precursors is None.
    """
    import sqlite3
    logging.info(f"Reading frame metadata for {bruker_d_folder_name}")
    with sqlite3.connect(
        os.path.join(bruker_d_folder_name, "analysis.tdf")
    ) as sql_database_connection:
        global_meta_data = pd.read_sql_query(
            "SELECT * from GlobalMetaData",
            sql_database_connection
        )
        frames = pd.read_sql_query(
            "SELECT * FROM Frames",
            sql_database_connection
        )
        if 9 in frames.MsMsType.values:
            acquisition_mode = "diaPASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * FROM DiaFrameMsMsInfo",
                sql_database_connection
            )
            fragment_frame_groups = pd.read_sql_query(
                "SELECT * from DiaFrameMsMsWindows",
                sql_database_connection
            )
            fragment_frames = fragment_frames.merge(
                fragment_frame_groups,
                how="left"
            )
            fragment_frames.rename(
                columns={"WindowGroup": "Precursor"},
                inplace=True
            )
            precursors = None
        elif 8 in frames.MsMsType.values:
            acquisition_mode = "ddaPASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * from PasefFrameMsMsInfo",
                sql_database_connection
            )
            precursors = pd.read_sql_query(
                "SELECT * from Precursors",
                sql_database_connection
            )
        else:
            acquisition_mode = "noPASEF"
            fragment_frames = pd.DataFrame(
                {
                    "Frame": np.array([0]),
                    "ScanNumBegin": np.array([0]),
                    "ScanNumEnd": np.array([0]),
                    "IsolationWidth": np.array([0]),
                    "IsolationMz": np.array([0]),
                    "Precursor": np.array([0]),
                }
            )
            precursors = None
            # raise ValueError("Scan mode is not ddaPASEF or diaPASEF")
        calibration_available = BRUKER_DLL_FILE_NAME != ""
        try:
            pd.read_sql_query(
                "SELECT * from CalibrationInfo",
                sql_database_connection
            )
        except pd.io.sql.DatabaseError:
            calibration_available = False
        if add_zeroth_frame:
            frames = pd.concat(
                [
                    pd.DataFrame(frames.iloc[0]).T,
                    frames,
                ],
                ignore_index=True
            )
            frames.Id[0] = 0
            frames.Time[0] = 0
            frames.MaxIntensity[0] = 0
            frames.SummedIntensities[0] = 0
            frames.NumPeaks[0] = 0
            frames.MsMsType[0] = 0
        polarity_col = frames["Polarity"].copy()
        frames = pd.DataFrame(
            {
                col: pd.to_numeric(
                    frames[col]
                ) for col in frames if col != "Polarity"
            }
        )
        if not drop_polarity:
            if convert_polarity_to_int:
                frames['Polarity'] = polarity_col.apply(
                    lambda x: 1 if x == "+" else -1
                ).astype(np.int8)
            else:
                frames['Polarity'] = polarity_col
        return (
            acquisition_mode,
            global_meta_data,
            frames,
            fragment_frames,
            precursors,
            calibration_available,
        )


def read_bruker_binary(
    frames: np.ndarray,
    bruker_d_folder_name: str,
    compression_type: int,
    max_peaks_per_scan: int,
    mmap_detector_events: bool = None,
) -> tuple:
    """Read all data from an "analysis.tdf_bin" of a Bruker .d folder.

    Parameters
    ----------
    frames : pd.DataFrame
        The frames from the "analysis.tdf" SQL database of a Bruker .d folder.
        These can be acquired with e.g. alphatims.bruker.read_bruker_sql.
    bruker_d_folder_name : str
        The full path to a Bruker .d folder.
    compression_type : int
        The compression type. This must be either 1 or 2.
    max_peaks_per_scan : int
        The maximum number of peaks per scan.
        Should be treieved from the global metadata.
    mmap_detector_events : bool
        Do not save the intensity_values and tof_indices in memory,
        but use an mmap instead.
        Default is True

    Returns
    -------
    : tuple (np.int64[:], np.uint32[:], np.uint16[:]).
        The scan_indptr, tof_indices and intensities.
    """
    frame_indptr = np.empty(frames.shape[0] + 1, dtype=np.int64)
    frame_indptr[0] = 0
    frame_indptr[1:] = np.cumsum(frames.NumPeaks.values)
    max_scan_count = frames.NumScans.max() + 1
    scan_count = max_scan_count * frames.shape[0]
    scan_indptr = np.zeros(scan_count + 1, dtype=np.int64)
    if mmap_detector_events:
        intensities = tm.empty(int(frame_indptr[-1]), dtype=np.uint16)
        tof_indices = tm.empty(int(frame_indptr[-1]), dtype=np.uint32)
    else:
        intensities = np.empty(int(frame_indptr[-1]), dtype=np.uint16)
        tof_indices = np.empty(int(frame_indptr[-1]), dtype=np.uint32)
    tdf_bin_file_name = os.path.join(bruker_d_folder_name, "analysis.tdf_bin")
    tims_offset_values = frames.TimsId.values
    logging.info(
        f"Reading {frame_indptr.size - 2:,} frames with "
        f"{frame_indptr[-1]:,} detector events for {bruker_d_folder_name}"
    )
    if compression_type == 1:
        process_frame_func = alphatims.utils.threadpool(
            process_frame,
            thread_count=1
        )
    else:
        process_frame_func = alphatims.utils.threadpool(process_frame)
    process_frame_func(
        range(1, len(frames)),
        tdf_bin_file_name,
        tims_offset_values,
        scan_indptr,
        intensities,
        tof_indices,
        frame_indptr,
        max_scan_count,
        compression_type,
        max_peaks_per_scan,
    )
    scan_indptr[1:] = np.cumsum(scan_indptr[:-1])
    scan_indptr[0] = 0
    return scan_indptr, tof_indices, intensities



def process_frame(
    frame_id: int,
    tdf_bin_file_name: str,
    tims_offset_values: np.ndarray,
    scan_indptr: np.ndarray,
    intensities: np.ndarray,
    tof_indices: np.ndarray,
    frame_indptr: np.ndarray,
    max_scan_count: int,
    compression_type: int,
    max_peaks_per_scan: int,
) -> None:
    """Read and parse a frame directly from a Bruker .d.analysis.tdf_bin.

    Parameters
    ----------
    frame_id : int
        The frame number that should be processed.
        Note that this is interpreted as 1-indixed instead of 0-indexed,
        so that it is compatible with Bruker.
    tdf_bin_file_name : str
        The full file name of the SQL database "analysis.tdf_bin" in a Bruker
        .d folder.
    tims_offset_values : np.int64[:]
        The offsets that indicate the starting indices of each frame in the
        binary.
        These are contained in the "TimsId" column of the frames table in
        "analysis.tdf_bin".
    scan_indptr : np.int64[:]
        A buffer containing zeros that can store the cumulative number of
        detections per scan.
        The size should be equal to max_scan_count * len(frames) + 1.
        A dummy 0-indexed frame is required to be present for len(frames).
        The last + 1 allows to explicitly interpret the end of a scan as
        the start of a subsequent scan.
    intensities : np.uint16[:]
        A buffer that can store the intensities of all detections.
        It's size can be determined by summing the "NumPeaks" column from
        the frames table in "analysis.tdf_bin".
    tof_indices : np.uint32[:]
        A buffer that can store the tof indices of all detections.
        It's size can be determined by summing the "NumPeaks" column from
        the frames table in "analysis.tdf_bin".
    frame_indptr : np.int64[:]
        The cumulative sum of the number of detections per frame.
        The size should be equal to len(frames) + 1.
        A dummy 0-indexed frame is required to be present for len(frames).
        The last + 1 allows to explicitly interpret the end of a frame as
        the start of a subsequent frame.
    max_scan_count : int
        The maximum number of scans a single frame can have.
    compression_type : int
        The compression type. This must be either 1 or 2.
        Should be treieved from the global metadata.
    max_peaks_per_scan : int
        The maximum number of peaks per scan.
        Should be retrieved from the global metadata.
    """
    with open(tdf_bin_file_name, "rb") as infile:
        frame_start = frame_indptr[frame_id]
        frame_end = frame_indptr[frame_id + 1]
        if frame_start != frame_end:
            offset = tims_offset_values[frame_id]
            infile.seek(offset)
            bin_size = int.from_bytes(infile.read(4), "little")
            scan_count = int.from_bytes(infile.read(4), "little")
            max_peak_count = min(
                max_peaks_per_scan,
                frame_end - frame_start
            )
            if compression_type == 1:
                import lzf
                compression_offset = 8 + (scan_count + 1) * 4
                scan_offsets = np.frombuffer(
                    infile.read((scan_count + 1) * 4),
                    dtype=np.int32
                ) - compression_offset
                compressed_data = infile.read(bin_size - compression_offset)
                scan_indices_ = np.zeros(scan_count, dtype=np.int64)
                tof_indices_ = np.empty(
                    frame_end - frame_start,
                    dtype=np.uint32
                )
                intensities_ = np.empty(
                    frame_end - frame_start,
                    dtype=np.uint16
                )
                scan_start = 0
                for scan_index in range(scan_count):
                    start = scan_offsets[scan_index]
                    end = scan_offsets[scan_index + 1]
                    if start == end:
                        continue
                    decompressed_bytes = lzf.decompress(
                        compressed_data[start: end],
                        max_peak_count * 4 * 2
                    )
                    scan_start += parse_decompressed_bruker_binary_type1(
                        decompressed_bytes,
                        scan_indices_,
                        tof_indices_,
                        intensities_,
                        scan_start,
                        scan_index,
                    )
            elif compression_type == 2:
                import pyzstd
                compressed_data = infile.read(bin_size - 8)
                decompressed_bytes = pyzstd.decompress(compressed_data)
                (
                    scan_indices_,
                    tof_indices_,
                    intensities_
                ) = parse_decompressed_bruker_binary_type2(decompressed_bytes)
            else:
                raise ValueError("TimsCompressionType is not 1 or 2.")
            scan_start = frame_id * max_scan_count
            scan_end = scan_start + scan_count
            scan_indptr[scan_start: scan_end] = scan_indices_
            tof_indices[frame_start: frame_end] = tof_indices_
            intensities[frame_start: frame_end] = intensities_



@alphatims.utils.njit(nogil=True)
def parse_decompressed_bruker_binary_type1(
    decompressed_bytes: bytes,
    scan_indices_: np.ndarray,
    tof_indices_: np.ndarray,
    intensities_: np.ndarray,
    scan_start: int,
    scan_index: int,
) -> int:
    """Parse a Bruker binary scan buffer into tofs and intensities.

    Parameters
    ----------
    decompressed_bytes : bytes
        A Bruker scan binary buffer that is already decompressed with lzf.
    scan_indices_ : np.ndarray
        The scan_indices_ buffer array.
    tof_indices_ : np.ndarray
        The tof_indices_ buffer array.
    intensities_ : np.ndarray
        The intensities_ buffer array.
    scan_start : int
        The offset where to start new tof_indices and intensity_values.
    scan_index : int
        The scan index.

    Returns
    -------
    : int
        The number of peaks in this scan.
    """
    buffer = np.frombuffer(decompressed_bytes, dtype=np.int32)
    tof_index = 0
    previous_was_intensity = True
    current_index = scan_start
    for value in buffer:
        if value >= 0:
            if previous_was_intensity:
                tof_index += 1
            tof_indices_[current_index] = tof_index
            intensities_[current_index] = value
            previous_was_intensity = True
            current_index += 1
        else:
            tof_index -= value
            previous_was_intensity = False
    scan_size = current_index - scan_start
    scan_indices_[scan_index] = scan_size
    return scan_size



@alphatims.utils.njit(nogil=True)
def parse_decompressed_bruker_binary_type2(decompressed_bytes: bytes) -> tuple:
    """Parse a Bruker binary frame buffer into scans, tofs and intensities.

    Parameters
    ----------
    decompressed_bytes : bytes
        A Bruker frame binary buffer that is already decompressed with pyzstd.

    Returns
    -------
    : tuple (np.uint32[:], np.uint32[:], np.uint32[:]).
        The scan_indices, tof_indices and intensities present in this binary
        array
    """
    temp = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    buffer = np.frombuffer(temp.reshape(4, -1).T.flatten(), dtype=np.uint32)
    scan_count = buffer[0]
    scan_indices = buffer[:scan_count].copy() // 2
    intensities = buffer[scan_count + 1::2]
    last_scan = len(intensities) - np.sum(scan_indices[1:])
    scan_indices[:-1] = scan_indices[1:]
    scan_indices[-1] = last_scan
    tof_indices = buffer[scan_count::2].copy()
    index = 0
    for size in scan_indices:
        current_sum = 0
        for i in range(size):
            current_sum += tof_indices[index]
            tof_indices[index] = current_sum
            index += 1
    return scan_indices, tof_indices - 1, intensities
