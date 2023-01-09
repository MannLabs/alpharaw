import pandas as pd
import numpy as np
from alphabase.io.hdf import HDF_File

class MSData_Base:
    """
    The base data structure for MS Data, other MSData loader inherit
    """
    def __init__(self, centroided:bool=True):
        """
        Parameters
        ----------
        centroided : bool, optional
            if peaks will be centroided after loading, 
            by default True
        """
        # A spectrum contains peaks
        self.spectrum_df:pd.DataFrame = pd.DataFrame()
        # A peak contains mz, intensity, and ...
        self.peak_df:pd.DataFrame = pd.DataFrame()
        self._raw_file_path = ''
        self.centroided = centroided
        self.creation_time = ''
        self.file_type = ''

    @property
    def raw_file_path(self)->str:
        return self._raw_file_path

    @raw_file_path.setter
    def raw_file_path(self, _path:str):
        self._raw_file_path = _path

    def import_raw(self, _path:str):
        self.raw_file_path = _path
        raw_data = self._import(_path)
        self._set_dataframes(raw_data)
        self._check_df()

    def load_raw(self, _path:str):
        self.import_raw(_path)

    def _save_meta_to_hdf(self, hdf:HDF_File):
        hdf.ms_data.creation_time = self.creation_time
        hdf.ms_data.raw_file_path = self.raw_file_path
        hdf.ms_data.file_type = self.file_type
        hdf.ms_data.centroided = self.centroided

    def _load_meta_from_hdf(self, hdf:HDF_File):
        self.creation_time = hdf.ms_data.creation_time
        self.raw_file_path = hdf.ms_data.raw_file_path
        self.file_type = hdf.ms_data.file_type
        self.centroided = hdf.ms_data.centroided

    def save_hdf(self, _path:str):
        hdf = HDF_File(
            _path, read_only=False,
            truncate=True, delete_existing=True
        )

        hdf.ms_data = {
            'spectrum_df': self.spectrum_df,
            'peak_df': self.peak_df
        }

        self._save_meta_to_hdf(hdf)
        

    def load_hdf(self, _path:str):
        hdf = HDF_File(
            _path, read_only=True,
            truncate=False, delete_existing=False
        )

        self.spectrum_df = hdf.ms_data.spectrum_df.values
        self.peak_df = hdf.ms_data.peak_df.values

        self._load_meta_from_hdf(hdf)

    def reset_spec_idxes(self):
        self.spectrum_df.reset_index(drop=True, inplace=True)
        self.spectrum_df['spec_idx'] = self.spectrum_df.index.values

    def _import(self, _path):
        raise NotImplementedError(
            f"{self.__class__} must implement `_import()`"
        )
    
    def _set_dataframes(self, raw_data):
        raise NotImplementedError(
            f"{self.__class__} must implement `_set_dataframes()`"
        )

    def _read_creation_time(self, raw_data):
        pass

    def _check_df(self):
        self._check_rt()
        # self._check_mobility()
        self._check_precursor_mz()
    
    def _check_rt(self):
        assert 'rt' in self.spectrum_df.columns
        # self.spectrum_df['rt_sec'] = self.spectrum_df.rt*60

    def _check_mobility(self):
        if 'mobility' not in self.spectrum_df.columns:
            self.spectrum_df['mobility'] = 0.0

    def _check_precursor_mz(self):
        if 'isolation_lower_mz' not in self.spectrum_df.columns:
            self.spectrum_df['isolation_lower_mz'] = -1.0
            self.spectrum_df['isolation_upper_mz'] = -1.0
        if 'precursor_mz' not in self.spectrum_df.columns:
            self.spectrum_df['precursor_mz'] = -1.0

    def create_spectrum_df(self,
        spectrum_num:int,
    ):
        self.spectrum_df = pd.DataFrame(
            index=np.arange(spectrum_num, dtype=np.int64)
        )
        self.spectrum_df['spec_idx'] = self.spectrum_df.index.values

    def set_peaks_by_cat_array(self, 
        mz_array:np.ndarray, 
        intensity_array:np.ndarray,
        peak_start_indices:np.ndarray,
        peak_stop_indices:np.ndarray,
    ):
        self.peak_df = pd.DataFrame({
            'mz': mz_array,
            'intensity': intensity_array
        })
        self.spectrum_df['peak_start_idx'] = peak_start_indices
        self.spectrum_df['peak_stop_idx'] = peak_stop_indices

    def set_peaks_by_array_list(self,
        mz_array_list:list,
        intensity_array_list:list,
    ):
        indices = index_ragged_list(mz_array_list)
        self.set_peaks_by_cat_array(
            np.concatenate(mz_array_list),
            np.concatenate(intensity_array_list),
            indices[:-1],
            indices[1:]
        )

    def add_column_in_spec_df(self, 
        column_name:str, 
        values:np.ndarray, 
        spec_idxes:np.ndarray = None,
        dtype:np.dtype=np.float64, na_value=np.nan,
    ):
        if spec_idxes is None:
            self.spectrum_df[
                column_name
            ] = np.array(values, dtype=dtype)
        else:
            self.spectrum_df.loc[
                spec_idxes, column_name
            ] = values
            self.spectrum_df[column_name].fillna(
                na_value, inplace=True
            )
            self.spectrum_df[
                column_name
            ] = self.spectrum_df[column_name].astype(dtype)

    def add_column_in_df_by_thermo_scan_num(self, 
        column_name:str, 
        values:np.ndarray, 
        scan_nums:np.ndarray,
        dtype:np.dtype=np.float64, 
        na_value=np.nan,
    ):
        self.add_column_in_spec_df(
            column_name, values,
            scan_nums-1, 
            dtype, na_value
        )

    def get_peaks(self, spec_idx):
        start, end = self.spectrum_df[[
            'peak_start_idx','peak_stop_idx'
        ]].values[spec_idx,:]
        return (
            self.peak_df.mz.values[start:end],
            self.peak_df.intensity.values[start:end],
        )

    def set_precursor_mz(self, 
        precursor_mz_values:np.ndarray,
        spec_idxes:np.ndarray=None, 
    ):
        self.add_column_in_spec_df(
            'precursor_mz', 
            precursor_mz_values, 
            spec_idxes, np.float64, -1.0
        )
    
    def set_precursor_mz_windows(self,
        precursor_lower_mz_values:np.ndarray,
        precursor_upper_mz_values:np.ndarray,
        spec_idxes:np.ndarray=None, 
    ):
        self.add_column_in_spec_df(
            'isolation_lower_mz',
            precursor_lower_mz_values, 
            spec_idxes, np.float64, -1.0
        )
        self.add_column_in_spec_df(
            'isolation_upper_mz',
            precursor_upper_mz_values, 
            spec_idxes, np.float64, -1.0
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
        self.spectrum_df.sort_values('rt',inplace=True)
        self.spectrum_df.reset_index(drop=True,inplace=True)
        self.spectrum_df['spec_idx_old'] = self.spectrum_df.spec_idx
        self.spectrum_df['spec_idx'] = self.spectrum_df.index
        mzs_list = []
        intens_list = []
        idx_list = []
        for start,end in self.spectrum_df[
            ['peak_start_idx','peak_stop_idx']
        ].values:
            mzs_list.append(self.peak_df.mz.values[start:end])
            intens_list.append(self.peak_df.intensity.values[start:end])
            idx_list.append(end-start)
        peak_indices = np.empty(len(idx_list)+1,dtype=np.int64)
        peak_indices[0] = 0
        peak_indices[1:] = np.cumsum(idx_list)
        self.peak_df.mz.values[:] = np.concatenate(mzs_list)
        self.peak_df.intensity.values[:] = np.concatenate(intens_list)
        self.spectrum_df['peak_start_idx'] = peak_indices[:-1]
        self.spectrum_df['peak_stop_idx'] = peak_indices[1:]

def index_ragged_list(ragged_list: list)  -> np.ndarray:
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
    def import_raw(self, _path:str):
        self.raw_file_path = _path
        self.load_hdf(_path)

class MSReaderProvider:
    """Factory class to register and get MS Readers"""
    def __init__(self):
        self.ms_reader_dict = {}

    def register_reader(self, ms2_type:str, reader_class):
        self.ms_reader_dict[ms2_type.lower()] = reader_class

    def get_reader(
        self, file_type:str, 
        *, 
        centroided:bool=True,
        **kwargs
    )->MSData_Base:
        file_type = file_type.lower()
        if file_type not in self.ms_reader_dict: return None
        else: return self.ms_reader_dict[file_type](
            centroided=centroided, **kwargs
        )

ms_reader_provider = MSReaderProvider()
ms_reader_provider.register_reader('alpharaw', MSData_HDF)
ms_reader_provider.register_reader('raw.hdf', MSData_HDF)
ms_reader_provider.register_reader('alpharaw_hdf', MSData_HDF)
