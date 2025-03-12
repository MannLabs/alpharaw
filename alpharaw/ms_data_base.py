import numpy as np
import pandas as pd
from alphabase.constants._const import PEAK_INTENSITY_DTYPE, PEAK_MZ_DTYPE
from alphabase.io.hdf import HDF_File


class MSData_Base:
    """
    The base data structure for MS RAW Data, other MSData loaders inherit this class.
    """

    column_dtypes = {
        "rt": np.float64,
        "ms_level": np.int8,
        "precursor_mz": np.float64,
        "isolation_lower_mz": np.float64,
        "isolation_upper_mz": np.float64,
        "precursor_charge": np.int8,
        "nce": np.float32,
        "injection_time": np.float32,
        "activation": "U",
    }

    spectrum_df: pd.DataFrame
    """
    Spectrum dataframe containing the following columns:

    - `rt` (float64): in minutes. `rt_sec` will be RT in seconds, which is not included by default.
    - `precursor_mz` (float64): mono_mz (DDA) or isolation center mz
    - `isolation_lower_mz` (float64): left of the isolation window
    - `isolation_upper_mz` (float64): right of the isolation window
    - `spec_idx` (int64): spectrum index. For thermo, it is `scan_num - 1`
    - `peak_start_idx` (int64): peak start position pointing to `self.peak_df`
    - `peak_stop_idx` (int64): peak stop position pointing to `self._peak_df`
    - `ms_level` (int8): =1 for MS1, =2 for MS2, ...
    - [`scan_num`] (int64): Thermo scan number
    - [`mobility`] (float64): Bruker timsTOF mobility

    Other columns depends on different vendors and file formats
    """

    peak_df: pd.DataFrame
    """
    Peak list dataframe containing the follow columns:
    - `mz` (PEAK_MZ_DTYPE in alphabase, float32 by default): m/z values of the peak
    - `intensity` (PEAK_INTENSITY_DTYPE in alphabase, float32 by default): intensity values of the peak
    """

    vocab: list = [
        "CID",
        "HCD",
        "ETD",
        "ECD",
        "EAD",
        "EXD",
        "UVPD",
        "ETHCD",
        "ETCID",
        "EXCID",
        "NETD",
        "IT",
        "FT",
        "TOF",
    ]
    """
    These spectrum infomation items in str format can be one-to-one mapped into
    unique token IDs (indices), for exampel "CID"=0, "HCD"=1, ...
    Token IDs are better for storage in HDF5 format.
    """

    def __init__(self, centroided: bool = True, save_as_hdf: bool = False, **kwargs):
        """
        Parameters
        ----------
        centroided : bool, optional
            If peaks will be centroided after loading, by default True
        save_as_hdf : bool, optional
            If automatically save the data into HDF5 format, by default False
        """
        # A spectrum contains peaks
        self.spectrum_df: pd.DataFrame = pd.DataFrame()
        # A peak contains mz, intensity, and ...
        self.peak_df: pd.DataFrame = pd.DataFrame()
        self._raw_file_path = ""
        self.centroided = centroided
        self._save_as_hdf = save_as_hdf
        self.creation_time = ""
        self.file_type = ""
        self.instrument = "none"

    def _get_term_id(self, terminology: str) -> int:
        """
        Get terminology ID from :attr:`.MSData_Base.vocab`, -1 if not exist.

        Parameters
        ----------
        terminology : str
            The terminology name from :attr:`.MSData_Base.vocab`, such as "CID", "HCD", ...

        Returns
        -------
        int
            Terminology ID, which is the index in :attr:`.MSData_Base.vocab`.
        """
        try:
            return self.vocab.index(terminology)
        except ValueError:
            return -1

    @property
    def raw_file_path(self) -> str:
        return self._raw_file_path

    @raw_file_path.setter
    def raw_file_path(self, raw_file_path: str):
        self._raw_file_path = raw_file_path

    def import_raw(self, raw_file_path: str):
        """
        Import a raw file. It involves three steps:
        ```
        raw_data_dict = self._import(raw_file_path)
        self._set_dataframes(raw_data_dict)
        self._check_df()
        ```

        Parameters
        ----------
        raw_file_path : str
            Absolute or relative path of the raw file.
        """
        self.raw_file_path = raw_file_path
        raw_data_dict = self._import(raw_file_path)
        self._set_dataframes(raw_data_dict)
        self._check_df()

        if self._save_as_hdf:
            self.save_hdf(raw_file_path + ".hdf")

    def load_raw(self, raw_file_path: str):
        """
        Wrapper of :func:`.MSData_Base.import_raw`
        """
        self.import_raw(raw_file_path)

    def _save_meta_to_hdf(self, hdf: HDF_File):
        hdf.ms_data.meta = {
            "creation_time": self.creation_time,
            "raw_file_path": self.raw_file_path,
            "file_type": self.file_type,
            "centroided": self.centroided,
            "instrument": self.instrument,
        }

    def _load_meta_from_hdf(self, hdf: HDF_File):
        self.creation_time = hdf.ms_data.meta.creation_time
        self.raw_file_path = hdf.ms_data.meta.raw_file_path
        self.file_type = hdf.ms_data.meta.file_type
        self.centroided = hdf.ms_data.meta.centroided
        self.instrument = hdf.ms_data.meta.instrument

    def save_hdf(self, hdf_file_path: str):
        """
        Save data into HDF5 file

        Parameters
        ----------
        hdf_file_path : str
            Absolute or relative path of HDF5 file.
        """
        hdf = HDF_File(
            hdf_file_path, read_only=False, truncate=True, delete_existing=True
        )

        hdf.ms_data = {"spectrum_df": self.spectrum_df, "peak_df": self.peak_df}

        self._save_meta_to_hdf(hdf)

    def load_hdf(self, hdf_file_path: str):
        """
        Load data from HDF5 file.

        Parameters
        ----------
        hdf_file_path : str
            Absolute or relative path of HDF5 file.
        """
        hdf = HDF_File(
            hdf_file_path, read_only=True, truncate=False, delete_existing=False
        )

        self.spectrum_df = hdf.ms_data.spectrum_df.values
        self.peak_df = hdf.ms_data.peak_df.values

        if hasattr(hdf.ms_data, "meta"):
            self._load_meta_from_hdf(hdf)

    def reset_spec_idxes(self):
        """
        Reset spec indexes to make sure spec_idx values are continuous ranging from 0 to N.
        """
        self.spectrum_df.reset_index(drop=True, inplace=True)
        self.spectrum_df["spec_idx"] = self.spectrum_df.index.values

    def _import(self, _path: str) -> dict:
        """
        Parameters
        ----------
        _path : str
            Path of raw file.

        Returns
        -------
        dict
            Example:
            ```
            spec_dict = {
                "_peak_indices": _peak_indices,
                "peak_mz": np.concatenate(mz_values).copy(),
                "peak_intensity": np.concatenate(intensity_values).copy(),
                "rt": np.array(rt_values).copy(),
                "precursor_mz": np.array(precursor_mz_values).copy(),
                "precursor_charge": np.array(precursor_charges, dtype=np.int8).copy(),
                "isolation_lower_mz": np.array(isolation_mz_lowers).copy(),
                "isolation_upper_mz": np.array(isolation_mz_uppers).copy(),
                "ms_level": np.array(ms_order_list, dtype=np.int8).copy(),
                "nce": np.array(ce_list, dtype=np.float32).copy(),
            }
            ```

        Raises
        ------
        NotImplementedError
            Sub-class of `MSData_Base` must implement this method.
        """
        raise NotImplementedError(f"{self.__class__} must implement `_import()`")

    def _set_dataframes(self, raw_data: dict):
        self.create_spectrum_df(len(raw_data["rt"]))
        self.set_peak_df_by_indexed_array(
            raw_data["peak_mz"],
            raw_data["peak_intensity"],
            raw_data["peak_indices"][:-1],
            raw_data["peak_indices"][1:],
        )

        for col, val in raw_data.items():
            if col in self.column_dtypes:
                if self.column_dtypes[col] == "O":
                    self.spectrum_df[col] = list(val)
                else:
                    self.spectrum_df[col] = np.array(val, dtype=self.column_dtypes[col])

    def _read_creation_time(self, raw_data):
        pass

    def _check_df(self):
        self._check_rt()
        # self._check_mobility()
        self._check_precursor_mz()
        self._check_peak_dtypes()

    def _check_peak_dtypes(self):
        if self.peak_df.mz.dtype != PEAK_MZ_DTYPE:
            self.peak_df.mz = self.peak_df.mz.astype(PEAK_MZ_DTYPE)
        if self.peak_df.intensity.dtype != PEAK_INTENSITY_DTYPE:
            self.peak_df.intensity = self.peak_df.intensity.astype(PEAK_INTENSITY_DTYPE)

    def _check_rt(self):
        assert "rt" in self.spectrum_df.columns
        # self.spectrum_df['rt_sec'] = self.spectrum_df.rt*60

    def _check_mobility(self):
        if "mobility" not in self.spectrum_df.columns:
            self.spectrum_df["mobility"] = 0.0

    def _check_precursor_mz(self):
        if "isolation_lower_mz" not in self.spectrum_df.columns:
            self.spectrum_df["isolation_lower_mz"] = -1.0
            self.spectrum_df["isolation_upper_mz"] = -1.0
        if "precursor_mz" not in self.spectrum_df.columns:
            self.spectrum_df["precursor_mz"] = -1.0

    def create_spectrum_df(
        self,
        spectrum_num: int,
    ):
        """
        Create a empty spectrum dataframe from the number of spectra.

        Parameters
        ----------
        spectrum_num : int
            The number of spectra.
        """
        self.spectrum_df = pd.DataFrame(index=np.arange(spectrum_num, dtype=np.int64))
        self.spectrum_df["spec_idx"] = self.spectrum_df.index.values

    def set_peak_df_by_indexed_array(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        peak_start_indices: np.ndarray,
        peak_stop_indices: np.ndarray,
    ):
        self.peak_df = pd.DataFrame(
            {
                "mz": mz_array.astype(PEAK_MZ_DTYPE),
                "intensity": intensity_array.astype(PEAK_INTENSITY_DTYPE),
            }
        )
        self.spectrum_df["peak_start_idx"] = peak_start_indices
        self.spectrum_df["peak_stop_idx"] = peak_stop_indices

    def set_peak_df_by_array_list(
        self,
        mz_array_list: list,
        intensity_array_list: list,
    ):
        indices = index_ragged_list(mz_array_list)
        self.set_peak_df_by_indexed_array(
            np.concatenate(mz_array_list),
            np.concatenate(intensity_array_list),
            indices[:-1],
            indices[1:],
        )

    def add_column_in_spec_df_by_spec_idxes(
        self,
        column_name: str,
        values: np.ndarray,
        spec_idxes: np.ndarray,
        dtype: np.dtype = np.float64,
        na_value=np.nan,
    ):
        self.spectrum_df.loc[spec_idxes, column_name] = values
        self.spectrum_df[column_name].fillna(na_value, inplace=True)
        self.spectrum_df[column_name] = self.spectrum_df[column_name].astype(dtype)

    def add_column_in_df_by_scan_num(
        self,
        column_name: str,
        values: np.ndarray,
        scan_nums: np.ndarray,
        dtype: np.dtype = np.float64,
        na_value=np.nan,
    ):
        """
        scan num starts from 1 not 0
        """
        self.add_column_in_spec_df_by_spec_idxes(
            column_name, values, scan_nums - 1, dtype, na_value
        )

    def get_peaks(self, spec_idx):
        start, end = self.spectrum_df[["peak_start_idx", "peak_stop_idx"]].values[
            spec_idx, :
        ]
        return (
            self.peak_df.mz.values[start:end],
            self.peak_df.intensity.values[start:end],
        )

    def set_precursor_mz_by_spec_idxes(
        self,
        precursor_mz_values: np.ndarray,
        spec_idxes: np.ndarray,
    ):
        self.add_column_in_spec_df_by_spec_idxes(
            "precursor_mz", precursor_mz_values, spec_idxes, np.float64, -1.0
        )

    def set_isolation_mz_windows_by_spec_idxes(
        self,
        precursor_lower_mz_values: np.ndarray,
        precursor_upper_mz_values: np.ndarray,
        spec_idxes: np.ndarray,
    ):
        self.add_column_in_spec_df_by_spec_idxes(
            "isolation_lower_mz",
            precursor_lower_mz_values,
            spec_idxes,
            np.float64,
            -1.0,
        )
        self.add_column_in_spec_df_by_spec_idxes(
            "isolation_upper_mz",
            precursor_upper_mz_values,
            spec_idxes,
            np.float64,
            -1.0,
        )

    def _sort_rt(self):
        """
        Used by :class:`alpharaw.wrappers.alphatims_wrapper.AlphaTims_Wrapper`.
        """
        # if 'mobility' in self.spectrum_df.columns:
        #     self.spectrum_df['_mobility'] = -self.spectrum_df.mobility
        #     self.spectrum_df.sort_values(['rt','_mobility'],inplace=True)
        #     self.spectrum_df.drop(columns=['_mobility'],inplace=True)
        # else:
        #     self.spectrum_df.sort_values('rt',inplace=True)
        self.spectrum_df.sort_values("rt", inplace=True)
        self.spectrum_df.reset_index(drop=True, inplace=True)
        self.spectrum_df["spec_idx_old"] = self.spectrum_df.spec_idx
        self.spectrum_df["spec_idx"] = self.spectrum_df.index
        mzs_list = []
        intens_list = []
        idx_list = []
        for start, end in self.spectrum_df[["peak_start_idx", "peak_stop_idx"]].values:
            mzs_list.append(self.peak_df.mz.values[start:end])
            intens_list.append(self.peak_df.intensity.values[start:end])
            idx_list.append(end - start)
        peak_indices = np.empty(len(idx_list) + 1, dtype=np.int64)
        peak_indices[0] = 0
        peak_indices[1:] = np.cumsum(idx_list)
        self.peak_df.mz.values[:] = np.concatenate(mzs_list)
        self.peak_df.intensity.values[:] = np.concatenate(intens_list)
        self.spectrum_df["peak_start_idx"] = peak_indices[:-1]
        self.spectrum_df["peak_stop_idx"] = peak_indices[1:]


def index_ragged_list(ragged_list: list) -> np.ndarray:
    """Create lookup indices for a list of arrays for concatenation.

    Args:
        value (list): Input list of arrays.

    Returns:
        indices: A numpy array with indices.
    """
    indices = np.zeros(len(ragged_list) + 1, np.int64)
    indices[1:] = [len(i) for i in ragged_list]
    indices = np.cumsum(indices, dtype=np.int64)

    return indices


class MSData_HDF(MSData_Base):
    """
    Wrapper of reader for alpharaw's HDF5 spectrum file.
    This class will be registered as file formats (types)
    "alpharaw", "raw.hdf", "alpharaw_hdf", "hdf" and "hdf5"
    in :obj:`ms_reader_provider` instance by :func:`register_readers`.
    """

    def import_raw(self, _path: str):
        self.raw_file_path = _path
        self.load_hdf(_path)


class MSReaderProvider:
    """Factory class to register and get MS Readers"""

    def __init__(self):
        self.ms_reader_dict = {}

    def register_reader(self, ms_file_type: str, reader_class: type):
        """
        Register a new reader for `ms_file_type` format with `reader_class`.

        Parameters
        ----------
        ms_file_type : str
            AlphaRaw supported MS file types.
        reader_class : type
            AlphaRaw supported MS class types.
        """
        self.ms_reader_dict[ms_file_type.lower()] = reader_class

    def get_reader(
        self, ms_file_type: str, *, centroided: bool = True, **kwargs
    ) -> MSData_Base:
        """
        Get the MS reader for the given `ms_file_type`.

        Parameters
        ----------
        ms_file_type : str
            AlphaRaw supported MS file types.
        centroided : bool, optional
            If peaks will be centroided after loading, by default True.

        Returns
        -------
        MSData_Base
            Instance of corresponding sub-class of `MSData_Base`.
        """
        ms_file_type = ms_file_type.lower()
        if ms_file_type not in self.ms_reader_dict:
            return None
        else:
            return self.ms_reader_dict[ms_file_type](centroided=centroided, **kwargs)


ms_reader_provider = MSReaderProvider()
"""
MS data register (:class:`MSReaderProvider`) performs as a factory to
produce different readers for given file formats.
"""


def register_readers():
    """
    Register :class:`MSData_HDF` for file formats (types):
    "alpharaw", "raw.hdf", "alpharaw_hdf", "hdf", "hdf5"
    in :obj:`ms_reader_provider`.
    """
    ms_reader_provider.register_reader("alpharaw", MSData_HDF)
    ms_reader_provider.register_reader("raw.hdf", MSData_HDF)
    ms_reader_provider.register_reader("alpharaw_hdf", MSData_HDF)
    ms_reader_provider.register_reader("hdf", MSData_HDF)
    ms_reader_provider.register_reader("hdf5", MSData_HDF)


register_readers()
