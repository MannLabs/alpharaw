import numpy as np
from pyteomics import mzml

from .ms_data_base import (
    PEAK_INTENSITY_DTYPE,
    PEAK_MZ_DTYPE,
    MSData_Base,
    ms_reader_provider,
)

SAFE_PRECURSOR_MZ = -1.0
SAFE_ISOLATION_MZ = -1.0
DEFAULT_ISOLATION_OFFSET = 1.5


class MzMLReader(MSData_Base):
    """
    Load mzml file as `:class:`alpharaw.ms_data_base.MSData_Base` structure.
    This reader will be registered as "mzml" in
    :obj:`alphraw.ms_data_base.ms_reader_provider` by :func:`register_readers`
    """

    def __init__(self, centroided: bool = True, save_as_hdf: bool = False, **kwargs):
        """
        Parameters
        ----------
        centroided : bool, optional
            If peaks will be centroided after loading,
            by default True.

        save_as_hdf : bool, optional
            Automatically save hdf after load raw data, by default False.
        """
        super().__init__(centroided, save_as_hdf, **kwargs)

    def _import(
        self,
        mzml_file_path: str,
    ) -> dict:
        """
        Implementation of :func:`alpharaw.ms_data_base.MSData_Base._import` interface,
        which will be called by :func:`alpharaw.ms_data_base.MSData_Base.import_raw`,
        the main entry of :class:`alpharaw.ms_data_base.MSData_Base` sub-classes.

        Parameters
        ----------
        mzml_file_path : str
            Absolute or relative path of the mzml file.
            For testing purpose, this can be pyteomics `MzML` object as well.

        Returns
        -------
        dict
            Spectrum information dict.
        """
        self.file_type = "mzml"
        if isinstance(mzml_file_path, str):
            reader = mzml.read(mzml_file_path, use_index=True)
        else:
            reader = mzml_file_path
        spec_indices = np.arange(len(reader), dtype=int)

        rt_list = []
        mzs_list = []
        intens_list = []
        ms_level_list = []
        prec_mz_list = []
        charge_list = []
        _peak_indices = []
        isolation_lower_mz_list = []
        isolation_upper_mz_list = []
        nce_list = []

        for _ in spec_indices:
            spec = next(reader)

            (
                rt,
                prec_mz,
                isolation_lower_mz,
                isolation_upper_mz,
                ms_level,
                nce,
                charge,
                masses,
                intensities,
            ) = parse_mzml_entry(spec)

            nce_list.append(nce)

            sortindex = np.argsort(masses)

            masses = masses[sortindex]
            intensities = intensities[sortindex]

            rt_list.append(rt)

            # Remove zero intensities
            to_keep = intensities > 0
            masses = masses[to_keep]
            intensities = intensities[to_keep]

            _peak_indices.append(len(masses))

            mzs_list.append(masses.astype(PEAK_MZ_DTYPE))
            intens_list.append(intensities.astype(PEAK_INTENSITY_DTYPE))
            ms_level_list.append(ms_level)
            prec_mz_list.append(prec_mz)
            charge_list.append(charge)
            isolation_lower_mz_list.append(isolation_lower_mz)
            isolation_upper_mz_list.append(isolation_upper_mz)

        if isinstance(mzml_file_path, str):
            reader.close()

        peak_indices = np.empty(len(spec_indices) + 1, np.int64)
        peak_indices[0] = 0
        peak_indices[1:] = np.cumsum(_peak_indices)
        ret_dict = {
            "peak_indices": peak_indices,
            "peak_mz": np.concatenate(mzs_list),
            "peak_intensity": np.concatenate(intens_list),
            "rt": np.array(rt_list),
            "precursor_mz": np.array(prec_mz_list),
            "precursor_charge": np.array(charge_list, dtype=np.int8),
            "isolation_lower_mz": np.array(isolation_lower_mz_list),
            "isolation_upper_mz": np.array(isolation_upper_mz_list),
            "ms_level": np.array(ms_level_list, dtype=np.int8),
        }
        nce_list = np.array(nce_list, dtype=np.float32)
        if np.any(np.isnan(nce_list)):
            return ret_dict
        ret_dict["nce"] = nce_list
        return ret_dict


def _parse_nce_from_filter_string(filter_string) -> float:
    """Parse NCE from Thermo-like filter strings."""
    if not filter_string:
        return np.nan

    try:
        if "@hcd" in filter_string:
            return float(filter_string.split("@hcd")[1].split(" ")[0])

        if "@cid" in filter_string:
            return float(filter_string.split("@cid")[1].split(" ")[0])

        return np.nan
    except (TypeError, ValueError, IndexError):
        return np.nan


def _parse_charge_state(selected_ion: dict | None) -> int:
    if selected_ion is None:
        return 0

    charge_state = selected_ion.get("charge state")

    if charge_state is None:
        return 0

    try:
        return int(charge_state)
    except (TypeError, ValueError):
        return 0


def _get_first_precursor(item_dict: dict) -> dict | None:
    precursor_list = item_dict.get("precursorList")
    if not isinstance(precursor_list, dict):
        return None

    precursors = precursor_list.get("precursor")
    if not isinstance(precursors, list) or not precursors:
        return None

    precursor = precursors[0]
    if not isinstance(precursor, dict):
        return None

    return precursor


def _get_first_selected_ion(precursor: dict | None) -> dict | None:
    if precursor is None:
        return None

    selected_ion_list = precursor.get("selectedIonList")
    if not isinstance(selected_ion_list, dict):
        return None

    selected_ions = selected_ion_list.get("selectedIon")
    if not isinstance(selected_ions, list) or not selected_ions:
        return None

    selected_ion = selected_ions[0]
    if not isinstance(selected_ion, dict):
        return None

    return selected_ion


def _get_isolation_window(precursor: dict | None) -> dict | None:
    if precursor is None:
        return None

    isolation_window = precursor.get("isolationWindow")
    if not isinstance(isolation_window, dict):
        return None

    return isolation_window


def parse_mzml_entry(item_dict: dict) -> tuple:
    """
    Parse mzml entries from pyteomics extracted items.

    Parameters
    ----------
    item_dict : dict
        pyteomics extracted items

    Returns
    -------
    tuple
        items in tuple format.
    """
    rt = float(item_dict.get("scanList").get("scan")[0].get("scan start time"))
    masses = item_dict.get("m/z array")
    intensities = item_dict.get("intensity array")
    ms_level = item_dict.get("ms level")
    prec_mz = SAFE_PRECURSOR_MZ
    isolation_lower_mz = SAFE_ISOLATION_MZ
    isolation_upper_mz = SAFE_ISOLATION_MZ
    charge = 0
    nce = 0.0
    if ms_level == 2:
        precursor = _get_first_precursor(item_dict)
        selected_ion = _get_first_selected_ion(precursor)
        charge = _parse_charge_state(selected_ion)

        precursor_mz_value = (
            None if selected_ion is None else selected_ion.get("selected ion m/z")
        )
        try:
            prec_mz = float(precursor_mz_value)
        except (TypeError, ValueError):
            prec_mz = SAFE_PRECURSOR_MZ

        if prec_mz != SAFE_PRECURSOR_MZ:
            iso_window = _get_isolation_window(precursor)
            iso_lower = None if iso_window is None else iso_window.get(
                "isolation window lower offset"
            )
            iso_upper = None if iso_window is None else iso_window.get(
                "isolation window upper offset"
            )
            try:
                iso_lower = float(iso_lower)
                iso_upper = float(iso_upper)
                isolation_upper_mz = prec_mz + iso_upper
                isolation_lower_mz = prec_mz - iso_lower
            except (TypeError, ValueError):
                isolation_upper_mz = prec_mz + DEFAULT_ISOLATION_OFFSET
                isolation_lower_mz = prec_mz - DEFAULT_ISOLATION_OFFSET

        filter_string = item_dict.get("scanList").get("scan")[0].get("filter string")
        nce = _parse_nce_from_filter_string(filter_string)
    return (
        rt,
        prec_mz,
        isolation_lower_mz,
        isolation_upper_mz,
        ms_level,
        nce,
        charge,
        masses,
        intensities,
    )


def register_readers():
    """
    Register :class:`MzMLReader` for file format "mzml" in
    :obj:`alpharaw.ms_data_base.ms_reader_provider`.
    """
    ms_reader_provider.register_reader("mzml", MzMLReader)
