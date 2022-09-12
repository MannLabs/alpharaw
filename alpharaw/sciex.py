import numpy as np
import pandas as pd
import os
import alpharaw.pysciexwifffilereader as pysciexwifffilereader
from alpharaw.ms_data_base import MSData_Base
    
class SciexWiffData(MSData_Base):
    def __init__(self, centroided:bool=True):
        super().__init__(centroided)
        self.centroid_mz_tol = 0.06
        self.ignore_empty_scans = True
        self.keep_k_peaks_per_spec = 2000
        self.sample_id = 0
        self.file_type = 'Sciex'

    def _import(self,
        _wiff_file_path:str
    )->dict:
        wiff_reader = pysciexwifffilereader.WillFileReader(
            _wiff_file_path
        )
        data_dict = wiff_reader.load_sample(self.sample_id,
            centroid = self.centroided,
            centroid_mz_tol = self.centroid_mz_tol,
            ignore_empty_scans=self.ignore_empty_scans,
            keep_k_peaks=self.keep_k_peaks_per_spec,
        )
        self.creation_time = wiff_reader.wiffSample.Details.AcquisitionDateTime.ToString("O")
        wiff_reader.close()
        return data_dict
    
    def _set_dataframes(self, raw_data:dict):
        self.create_spectrum_df(len(raw_data['rt']))
        self.set_peaks_by_cat_array(
            raw_data['peak_mz'],
            raw_data['peak_intensity'],
            raw_data['peak_indices'][:-1],
            raw_data['peak_indices'][1:],
        )
        self.peak_df['peak_start_mz'] = raw_data['peak_start_mz']
        self.peak_df['peak_end_mz'] = raw_data['peak_end_mz']
        self.add_column_in_spec_df(
            'rt', raw_data['rt']
        )
        self.add_column_in_spec_df(
            'ms_level', raw_data['ms_level'],
            dtype=np.int8
        )
        self.set_precursor_mz(
            raw_data['precursor_mz']
        )
        self.add_column_in_spec_df(
            'charge', raw_data['precursor_charge'],
            dtype=np.int8
        )
        self.set_precursor_mz_windows(
            raw_data['precursor_mz_lower'],
            raw_data['precursor_mz_upper'],
        )
        self.add_column_in_spec_df(
            'experiment_id', raw_data['experiment_id'],
            dtype=np.int32
        )
        self.add_column_in_spec_df(
            'cycle_id', raw_data['cycle_id'],
            dtype=np.int32
        )