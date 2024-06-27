import multiprocessing as mp
import platform
from functools import partial

import numpy as np
from tqdm import tqdm

import alpharaw.raw_access.pythermorawfilereader as pyrawfilereader

from .ms_data_base import (
    PEAK_INTENSITY_DTYPE,
    PEAK_MZ_DTYPE,
    MSData_Base,
    ms_reader_provider,
)

#: These thermo spectrum items can be only accessed by trailer dict using RawFileReader APIs.
__trailer_extra_list__ = [
    "injection_time",
    "cv",
    "max_ion_time",
    "agc_target",
    "energy_ev",
    "injection_optics_settling_time",
    "funnel_rf_level",
    "faims_cv",
]

auxiliary_item_dtypes: dict = {
    "injection_time": np.float32,
    "cv": np.float32,
    "max_ion_time": np.float32,
    "agc_target": np.int64,
    "energy_ev": np.float32,
    "injection_optics_settling_time": np.float32,
    "funnel_rf_level": np.float32,
    "faims_cv": np.float32,
    "detector": "U",
    "activation": "U",
    "analyzer": "U",
    "detector_id": np.uint8,
    "activation_id": np.uint8,
    "analyzer_id": np.uint8,
    "scan_event_string": "U",
    "multinotch": "O",
}
"""
The auxiliary items and dtypes that can be accessed from thermo RawFileReader.
"""


class ThermoRawData(MSData_Base):
    """
    Loading Thermo Raw data as :class:`alpharaw.ms_data_base.MSData_Base` data structure.
    This class will be registered as file formats "thermo" and "thermo_raw" in
    :obj:`alpharaw.ms_data_base.ms_reader_provider` by local :func:`register_readers`.
    """

    MASS_ANALYZER_ID_TO_TYPE = dict(
        (_id, _t)
        for _id, _t in pyrawfilereader.RawFileReader.massAnalyzerType.items()
        if isinstance(_id, int)
    )
    """
    The dict of Thermo's mass analyzer ID (int) to its name (str).
    The IDs correspond to enum values (int) of `int(scan_event.MassAnalyzer)`
    in RawFileReader, which originate from
    `ThermoFisher.CommonCore.Data.FilterEnums.MassAnalyzerType`.
    """

    MASS_ANALYZER_TYPE_TO_ID = dict(
        (_t, _id)
        for _t, _id in pyrawfilereader.RawFileReader.massAnalyzerType.items()
        if isinstance(_id, int)
    )
    """
    The dict of Thermo's mass analyzer name (str) to its ID (int).
    The IDs correspond to enum values of `int(scan_event.MassAnalyzer)`
    in RawFileReader, which originate from
    `ThermoFisher.CommonCore.Data.FilterEnums.MassAnalyzerType`.
    """

    ACTIVATION_ID_TO_TYPE = dict(
        (_id, _t)
        for _id, _t in pyrawfilereader.RawFileReader.activationType.items()
        if isinstance(_id, int)
    )
    """
    The dict of Thermo's activation name (str) to its ID (int).
    The IDs correspond to enum values (int) of `int(scan_event.GetActivation(index))`
    in RawFileReader, which originate from
    `ThermoFisher.CommonCore.Data.FilterEnums.ActivationType`.
    Note that multiple activations like `ETHCD` and `ETCID` are
    not thermo's built-in activation types,
    AlphaRaw still assigns IDs to them (ETHCD=201, ETCID=202).
    We can get multiple activations from auxiliary_item `scan_event_string`.
    """

    ACTIVATION_TYPE_TO_ID = dict(
        (_t, _id)
        for _t, _id in pyrawfilereader.RawFileReader.activationType.items()
        if isinstance(_id, int)
    )
    """
    The dict of Thermo's activation ID (int) to its name (str).
    The IDs correspond to enum values (int) of `int(scan_event.GetActivation(index))`
    in RawFileReader, which originate from
    `ThermoFisher.CommonCore.Data.FilterEnums.ActivationType`.
    Note that `ETHCD` and `ETCID` are not thermo's built-in activation types,
    AlphaRaw still assigns IDs to them (ETHCD=201, ETCID=202).
    We can get multiple activations from auxiliary_item `scan_event_string`.
    """

    def __init__(
        self,
        centroided: bool = True,
        process_count: int = 10,
        mp_batch_size: int = 5000,
        save_as_hdf: bool = False,
        dda: bool = False,
        auxiliary_items: list = [],
        **kwargs,
    ):
        """
        Parameters
        ----------
        centroided : bool, optional
            If peaks will be centroided after loading. By defaults True.
        process_count : int, optional
            Number of processes to load RAW data, by default 10.
        mp_batch_size : int, optional
            Number of spectra to load in each batch, by default 5000.
        save_as_hdf : bool, optional
            Automatically save hdf after load raw data, by default False.
        dda : bool, optional
            Is DDA data, by default False.
        auxiliary_items : list, optional
            Additional spectrum items, candidates are in :data:`auxiliary_item_dtypes`.
            By default [].
        """
        super().__init__(centroided, save_as_hdf=save_as_hdf, **kwargs)
        self.file_type = "thermo"
        self.process_count = process_count
        self.mp_batch_size = mp_batch_size
        self.dda = dda
        self.auxiliary_items = auxiliary_items
        self.column_dtypes.update(auxiliary_item_dtypes)

    def _import(
        self,
        raw_file_path: str,
    ) -> dict:
        """
        Re-implementation of :func:`MSData_Base._import` to enable :func:`.MSData_Base.import_raw`.

        Parameters
        ----------
        raw_file_path : str
            File path of the raw data.

        Returns
        -------
        dict
            Spectrum information in a temporary dict format.
        """
        rawfile = pyrawfilereader.RawFileReader(raw_file_path)
        self.creation_time = rawfile.GetCreationDate()

        # create batches for multiprocessing
        first_spectrum_number = rawfile.FirstSpectrumNumber
        last_spectrum_number = rawfile.LastSpectrumNumber

        if self.process_count > 1:
            mode = "spawn" if platform.system() != "Linux" else "forkserver"

            batches = np.arange(
                first_spectrum_number, last_spectrum_number + 1, self.mp_batch_size
            )
            batches = np.append(batches, last_spectrum_number + 1)

            # use multiprocessing to load batches
            _import_batch_partial = partial(
                _import_batch,
                raw_file_path=raw_file_path,
                centroided=self.centroided,
                dda=self.dda,
                auxiliary_items=self.auxiliary_items,
            )
            with mp.get_context(mode).Pool(processes=self.process_count) as pool:
                batches = list(
                    tqdm(
                        pool.imap(_import_batch_partial, zip(batches[:-1], batches[1:]))
                    )
                )

            # collect peak indices
            _peak_indices = np.concatenate(
                [batch["_peak_indices"] for batch in batches]
            )
            peak_indices = np.empty(rawfile.LastSpectrumNumber + 1, np.int64)
            peak_indices[0] = 0
            peak_indices[1:] = np.cumsum(_peak_indices)

            output_dict = {"peak_indices": peak_indices}

            # concatenate other arrays
            for key in batches[0]:
                if key == "_peak_indices":
                    continue
                output_dict[key] = np.concatenate([batch[key] for batch in batches])
        else:
            output_dict = _import_batch(
                (first_spectrum_number, last_spectrum_number + 1),
                raw_file_path,
                self.centroided,
                dda=self.dda,
                auxiliary_items=self.auxiliary_items,
            )
            peak_indices = np.empty(rawfile.LastSpectrumNumber + 1, np.int64)
            peak_indices[0] = 0
            peak_indices[1:] = np.cumsum(output_dict["_peak_indices"])
            output_dict["peak_indices"] = peak_indices
        rawfile.Close()

        return output_dict


def _import_batch(
    start_stop_tuple: tuple,
    raw_file_path: str,
    centroided: bool,
    dda: bool,
    auxiliary_items: list,
) -> dict:
    """Collect spectra from a batch of scans.

    Parameters
    ----------
    start_stop_tuple : tuple
        tuple of start and stop scan numbers

    raw_file_path : str
        path to the raw file

    centroided : bool
        if centroided peaks should be collected

    dda : bool
        is dda data.

    auxiliary_items : list
        Candidates are in :data:`auxiliary_item_dtypes`.

    Returns
    -------
    dict
        dictionary of collected spectra

    """

    rawfile = pyrawfilereader.RawFileReader(raw_file_path)
    start, stop = start_stop_tuple
    use_trailer_extra = False

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

    auxiliary_dict = dict((item, []) for item in auxiliary_items)
    if dda:
        use_trailer_extra = True
    else:
        for item in __trailer_extra_list__:
            if item in auxiliary_dict:
                use_trailer_extra = True
                break

    for i in range(start, stop):
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

        if use_trailer_extra:
            trailer_data = rawfile.GetTrailerExtraForScanNum(i)
        scan_event = rawfile.GetScanEventForScanNum(i)

        if "injection_time" in auxiliary_dict:
            auxiliary_dict["injection_time"].append(
                float(trailer_data["Ion Injection Time (ms):"])
            )
        if "max_ion_time" in auxiliary_dict:
            auxiliary_dict["max_ion_time"].append(
                float(trailer_data["Max. Ion Time (ms):"])
            )
        if "agc_target" in auxiliary_dict:
            auxiliary_dict["agc_target"].append(float(trailer_data["AGC Target:"]))
        if "energy_ev" in auxiliary_dict:
            energy_ev = trailer_data["HCD Energy V:"]
            if not energy_ev:
                energy_ev = trailer_data["HCD Energy eV:"]
            if not energy_ev:
                energy_ev = 0
            auxiliary_dict["energy_ev"].append(float(energy_ev))
        if "injection_optics_settling_time" in auxiliary_dict:
            auxiliary_dict["injection_optics_settling_time"].append(
                float(trailer_data["Injection Optics Settling Time (ms):"])
            )
        if "funnel_rf_level" in auxiliary_dict:
            auxiliary_dict["funnel_rf_level"].append(
                float(trailer_data["Funnel RF Level:"])
            )
        if "faims_cv" in auxiliary_dict:
            auxiliary_dict["faims_cv"].append(float(trailer_data["FAIMS CV:"]))
        if "activation" in auxiliary_dict:
            auxiliary_dict["activation"].append(
                ThermoRawData.ACTIVATION_ID_TO_TYPE[int(scan_event.GetActivation(0))]
                if ms_order > 1
                else "MS1"
            )
        if "activation_id" in auxiliary_dict:
            auxiliary_dict["activation_id"].append(
                int(scan_event.GetActivation(0)) if ms_order > 1 else 255
            )
        if "analyzer" in auxiliary_dict:
            auxiliary_dict["analyzer"].append(
                ThermoRawData.MASS_ANALYZER_ID_TO_TYPE[int(scan_event.MassAnalyzer)]
            )
        if "analyzer_id" in auxiliary_dict:
            auxiliary_dict["analyzer_id"].append(int(scan_event.MassAnalyzer))
        if "scan_event_string" in auxiliary_dict:
            auxiliary_dict["scan_event_string"].append(
                rawfile.GetScanEventStringForScanNum(i)
            )
        if "multinotch" in auxiliary_dict:
            windows = []
            for _ in range(scan_event.MassCount):
                _mass = scan_event.GetMass(_)
                _width = scan_event.GetIsolationWidth(_)
                windows.append((_mass - _width / 2, _mass + _width / 2))
            auxiliary_dict["multinotch"].append(np.array(windows))

        if ms_order == 1:
            ce_list.append(0)
            if "cv" in auxiliary_dict:
                auxiliary_dict["cv"].append(0)
            precursor_mz_values.append(-1.0)
            isolation_mz_lowers.append(-1.0)
            isolation_mz_uppers.append(-1.0)
            precursor_charges.append(0)
        else:
            ce_list.append(rawfile.GetCollisionEnergyForScanNum(i))
            if "cv" in auxiliary_dict:
                n_cvs = scan_event.SourceFragmentationInfoCount
                if n_cvs > 0:
                    auxiliary_dict["cv"].append(
                        scan_event.GetSourceFragmentationInfo(0)
                    )
                else:
                    auxiliary_dict["cv"].append(0)

            isolation_center = scan_event.GetReaction(0).PrecursorMass
            isolation_width = scan_event.GetIsolationWidth(0)
            # isolation_center = rawfile.GetPrecursorMassForScanNum(i)
            # isolation_width = rawfile.GetIsolationWidthForScanNum(i)

            if dda:
                # GetMS2MonoMzAndChargeFromScanNum is slow
                mono_mz, charge = _get_mono_and_charge(trailer_data, scan_event)
                if mono_mz <= 0:
                    mono_mz = isolation_center
            else:
                mono_mz = isolation_center
                charge = 0

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

    # copys of numpy arrays are needed to move them explicitly to cpython heap
    # otherwise mono might interfere later
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
    for key, val in list(auxiliary_dict.items()):
        auxiliary_dict[key] = np.array(val, dtype=auxiliary_item_dtypes[key]).copy()
    spec_dict.update(auxiliary_dict)
    return spec_dict


def _get_mono_and_charge(trailer_data, scan_event):
    mono = float(trailer_data["Monoisotopic M/Z:"].strip())
    charge = int(trailer_data["Charge State:"].strip())
    return mono, charge


def register_readers():
    """
    Register :class:`ThermoRawData` for file formats (types):
    "thermo" and "thermo_raw" in :obj:`alpharaw.ms_data_base.ms_reader_provider`.
    """
    ms_reader_provider.register_reader("thermo", ThermoRawData)
    ms_reader_provider.register_reader("thermo_raw", ThermoRawData)
