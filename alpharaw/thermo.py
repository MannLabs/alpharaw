import numpy as np
import pandas as pd
import os
import alpharaw.raw_access.pythermorawfilereader as pyrawfilereader
from .ms_data_base import MSData_Base
from .ms_data_base import ms_reader_provider

class ThermoRawData(MSData_Base):
    """
    Loading Thermo Raw data as MSData_Base data structure.
    """
    def __init__(self, centroided:bool=True):
        """
        Parameters
        ----------
        centroided : bool, optional
            if peaks will be centroided after loading, 
            by default True
        """
        super().__init__(centroided)
        self.file_type = 'thermo'

    def _import(self,
        raw_file_path: str,
    ) -> dict:
        rawfile = pyrawfilereader.RawFileReader(raw_file_path)

        self.creation_time = rawfile.GetCreationDate()
        _peak_indices = []
        mz_values = []
        intensity_values = []
        rt_values = []
        precursor_mz_values = []
        isolation_mz_lowers = []
        isolation_mz_uppers = []
        precursor_charges = []
        ms_order_list = []
        for i in range(
            rawfile.FirstSpectrumNumber,
            rawfile.LastSpectrumNumber + 1
        ):
            if not self.centroided:
                masses, intensities = rawfile.GetProfileMassListFromScanNum(i)
            else:
                masses, intensities = rawfile.GetCentroidMassListFromScanNum(i)
            mz_values.append(masses)
            intensity_values.append(intensities.astype(np.float32))
            _peak_indices.append(len(masses))
            rt = rawfile.RTFromScanNum(i)
            rt_values.append(rt)
            ms_order = rawfile.GetMSOrderForScanNum(i)
            ms_order_list.append(ms_order)
            if ms_order == 1:
                precursor_mz_values.append(-1.0)
                isolation_mz_lowers.append(-1.0)
                isolation_mz_uppers.append(-1.0)
                precursor_charges.append(0)
            else:
                isolation_center = rawfile.GetPrecursorMassForScanNum(i)
                isolation_width = rawfile.GetIsolationWidthForScanNum(i)

                mono_mz, charge = rawfile.GetMS2MonoMzAndChargeFromScanNum(i)

                precursor_mz_values.append(mono_mz)
                precursor_charges.append(charge)
                isolation_mz_lowers.append(isolation_center - isolation_width / 2)
                isolation_mz_uppers.append(isolation_center + isolation_width / 2)
        rawfile.Close()
        peak_indices = np.empty(rawfile.LastSpectrumNumber + 1, np.int64)
        peak_indices[0] = 0
        peak_indices[1:] = np.cumsum(_peak_indices)
        return {
            'peak_indices': peak_indices,
            'peak_mz': np.concatenate(mz_values),
            'peak_intensity': np.concatenate(intensity_values),
            'rt': np.array(rt_values),
            'precursor_mz': np.array(precursor_mz_values),
            'precursor_charge': np.array(precursor_charges, dtype=np.int8),
            'isolation_mz_lower': np.array(isolation_mz_lowers),
            'isolation_mz_upper': np.array(isolation_mz_uppers),
            'ms_level': np.array(ms_order_list, dtype=np.int8),
        }

    def _set_dataframes(self, raw_data:dict):
        self.create_spectrum_df(len(raw_data['rt']))
        self.set_peaks_by_cat_array(
            raw_data['peak_mz'],
            raw_data['peak_intensity'],
            raw_data['peak_indices'][:-1],
            raw_data['peak_indices'][1:],
        )
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
            raw_data['isolation_mz_lower'],
            raw_data['isolation_mz_upper'],
        )

ms_reader_provider.register_reader('thermo', ThermoRawData)
ms_reader_provider.register_reader('thermo_raw', ThermoRawData)
