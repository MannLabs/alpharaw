import numpy as np
import pandas as pd
import platform
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

import alpharaw.raw_access.pythermorawfilereader as pyrawfilereader
from .ms_data_base import (
    MSData_Base, PEAK_MZ_DTYPE, PEAK_INTENSITY_DTYPE
)
from .ms_data_base import ms_reader_provider

def _import_batch(
    raw_file_path: str,
    centroided: bool,
    start_stop_tuple: tuple,
) -> dict:
    """Collect spectra from a batch of scans.

    Parameters
    ----------
    raw_file_path : str
        path to the raw file

    centroided : bool
        if centroided peaks should be collected

    start_stop_tuple : tuple
        tuple of start and stop scan numbers

    Returns
    -------
    dict
        dictionary of collected spectra
    
    """

    rawfile = pyrawfilereader.RawFileReader(raw_file_path)
    start, stop = start_stop_tuple
    
    _peak_indices = []
    mz_values = []
    intensity_values = []
    rt_values = []
    precursor_mz_values = []
    isolation_mz_lowers = []
    isolation_mz_uppers = []
    precursor_charges = []
    ms_order_list = []
    ce_list = []
    for i in range(
        start,
        stop
    ):
        if not centroided:
            masses, intensities = rawfile.GetProfileMassListFromScanNum(i)
        else:
            masses, intensities = rawfile.GetCentroidMassListFromScanNum(i)
        mz_values.append(masses.astype(PEAK_MZ_DTYPE))
        intensity_values.append(intensities.astype(PEAK_INTENSITY_DTYPE))
        _peak_indices.append(len(masses))
        rt = rawfile.RTFromScanNum(i)
        rt_values.append(rt)
        ms_order = rawfile.GetMSOrderForScanNum(i)
        ms_order_list.append(ms_order)

        if ms_order == 1:
            ce_list.append(0)
            precursor_mz_values.append(-1.0)
            isolation_mz_lowers.append(-1.0)
            isolation_mz_uppers.append(-1.0)
            precursor_charges.append(0)
        else:
            ce_list.append(rawfile.GetCollisionEnergyForScanNum(i))

            isolation_center = rawfile.GetPrecursorMassForScanNum(i)
            isolation_width = rawfile.GetIsolationWidthForScanNum(i)

            mono_mz, charge = rawfile.GetMS2MonoMzAndChargeFromScanNum(i)
            if mono_mz <= 0:
                mono_mz = isolation_center

            # In case that: ms1 = ms_order==2&NCE==0?
            if mono_mz <= 0:
                precursor_mz_values.append(-1.0)
                isolation_mz_lowers.append(-1.0)
                isolation_mz_uppers.append(-1.0)
                precursor_charges.append(0)
                ms_order_list[-1] = 1
            else:
                precursor_mz_values.append(mono_mz)
                isolation_mz_lowers.append(isolation_center - isolation_width / 2)
                isolation_mz_uppers.append(isolation_center + isolation_width / 2)
                precursor_charges.append(charge)
    rawfile.Close()
    return {
        '_peak_indices': _peak_indices,
        'peak_mz': np.concatenate(mz_values),
        'peak_intensity': np.concatenate(intensity_values),
        'rt': np.array(rt_values),
        'precursor_mz': np.array(precursor_mz_values),
        'precursor_charge': np.array(precursor_charges, dtype=np.int8),
        'isolation_lower_mz': np.array(isolation_mz_lowers),
        'isolation_upper_mz': np.array(isolation_mz_uppers),
        'ms_level': np.array(ms_order_list, dtype=np.int8),
        'nce': np.array(ce_list, dtype=np.float32),
    }
class ThermoRawData(MSData_Base):
    """
    Loading Thermo Raw data as MSData_Base data structure.
    """
    def __init__(self, 
            centroided : bool = True,
            process_count : int = 10,
            mp_batch_size : int = 10000,
            **kwargs):
        """
        Parameters
        ----------
        centroided : bool, default = True
            if peaks will be centroided after loading, 
            by default True

        process_count : int, default = 8
            number of processes to use for loading
        
        mp_batch_size : int, default = 10000
            number of spectra to load in each batch
        """
        super().__init__(centroided)
        self.file_type = 'thermo'
        self.process_count = process_count
        self.mp_batch_size = mp_batch_size

    def _import(self,
        raw_file_path: str,
    ) -> dict:
        rawfile = pyrawfilereader.RawFileReader(raw_file_path)
        self.creation_time = rawfile.GetCreationDate()

        # create batches for multiprocessing
        first_spectrum_number = rawfile.FirstSpectrumNumber
        last_spectrum_number = rawfile.LastSpectrumNumber

        if platform.system() != 'Linux':
            batches = np.arange(first_spectrum_number, last_spectrum_number+1, self.mp_batch_size)
            batches = np.append(batches, last_spectrum_number+1)

            # use multiprocessing to load batches
            _import_batch_partial = partial(_import_batch, raw_file_path, self.centroided)
            with mp.get_context("spawn").Pool(processes = self.process_count) as pool:
                batches = list(tqdm(pool.imap(_import_batch_partial, zip(batches[:-1], batches[1:]))))

        else:
            batches = [_import_batch(raw_file_path, self.centroided, (first_spectrum_number, last_spectrum_number+1))]

        # collect peak indices
        _peak_indices = np.concatenate([batch['_peak_indices'] for batch in batches])        
        peak_indices = np.empty(rawfile.LastSpectrumNumber + 1, np.int64)
        peak_indices[0] = 0
        peak_indices[1:] = np.cumsum(_peak_indices)

        output_dict = {"peak_indices": peak_indices}

        # concatenate other arrays
        for key in batches[0].keys():
            if key == '_peak_indices':
                continue
            output_dict[key] = np.concatenate([batch[key] for batch in batches])

        rawfile.Close()

        return output_dict
    
    def import_raw(self, _path: str):
        super().import_raw(_path)
        self.save_hdf(_path+".hdf")

ms_reader_provider.register_reader('thermo', ThermoRawData)
ms_reader_provider.register_reader('thermo_raw', ThermoRawData)
