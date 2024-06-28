import numpy as np
from pyteomics import mzml

from .ms_data_base import (
    PEAK_INTENSITY_DTYPE,
    PEAK_MZ_DTYPE,
    MSData_Base,
    ms_reader_provider,
)


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
    prec_mz = -1.0
    isolation_lower_mz = -1.0
    isolation_upper_mz = -1.0
    charge = 0
    nce = 0.0
    if ms_level == 2:
        try:
            charge = int(
                item_dict.get("precursorList")
                .get("precursor")[0]
                .get("selectedIonList")
                .get("selectedIon")[0]
                .get("charge state")
            )
        except TypeError:
            charge = 0
        try:
            charge = int(
                item_dict.get("precursorList")
                .get("precursor")[0]
                .get("selectedIonList")
                .get("selectedIon")[0]
                .get("charge state")
            )
        except TypeError:
            charge = 0

        prec_mz = (
            item_dict.get("precursorList")
            .get("precursor")[0]
            .get("selectedIonList")
            .get("selectedIon")[0]
            .get("selected ion m/z")
        )
        try:
            iso_window = (
                item_dict.get("precursorList")
                .get("precursor")[0]
                .get("isolationWindow")
            )
            iso_lower = float(iso_window.get("isolation window lower offset"))
            iso_upper = float(iso_window.get("isolation window upper offset"))
            isolation_upper_mz = prec_mz + iso_upper
            isolation_lower_mz = prec_mz - iso_lower
        except TypeError:
            isolation_upper_mz = prec_mz + 1.5
            isolation_lower_mz = prec_mz - 1.5

        nce = np.nan
        try:
            filter_string = (
                item_dict.get("scanList").get("scan")[0].get["filter string"]
            )
            if "@hcd" in filter_string:
                nce = float(filter_string.split("@hcd")[1].split(" ")[0])
            elif "@cid" in filter_string:
                nce = float(filter_string.split("@cid")[1].split(" ")[0])
            else:
                nce = np.nan
        except Exception:
            nce = np.nan
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
