import warnings

import alpharaw.raw_access.pysciexwifffilereader as pysciexwifffilereader

from .ms_data_base import MSData_Base, ms_reader_provider


class SciexWiffData(MSData_Base):
    """
    Load Sciex Wiff data as :class:`alpharaw.ms_data_base.MSData_Base` data structure.
    This reader will be registered as "sciex", "sciex_wiff", and "sciex_raw"
    in :obj:`alpharaw.ms_data_base.ms_reader_provider` by :func:`register_readers()`.
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
        super().__init__(centroided, save_as_hdf=save_as_hdf, **kwargs)
        if self.centroided:
            self.centroided = False
            warnings.warn("Centroiding for Sciex data is not well implemented yet")
        self.centroid_ppm = 20.0
        self.ignore_empty_scans = True
        self.keep_k_peaks_per_spec = 2000
        self.sample_id = 0
        self.file_type = "sciex"

    def _import(self, _wiff_file_path: str) -> dict:
        """
        Implementation of :func:`alpharaw.ms_data_base.MSData_Base._import` interface.

        Parameters
        ----------
        _wiff_file_path : str
            Absolute or relative path of the sciex wiff file.

        Returns
        -------
        dict
            Spectrum information dict.
        """
        wiff_reader = pysciexwifffilereader.WillFileReader(_wiff_file_path)
        data_dict = wiff_reader.load_sample(
            self.sample_id,
            centroid=self.centroided,
            centroid_ppm=self.centroid_ppm,
            ignore_empty_scans=self.ignore_empty_scans,
            keep_k_peaks=self.keep_k_peaks_per_spec,
        )
        self.creation_time = (
            wiff_reader.wiffSample.Details.AcquisitionDateTime.ToString("O")
        )
        wiff_reader.close()
        return data_dict


def register_readers():
    """
    Register :class:`SciexWiffData` for file formats (types):
    "sciex", "sciex_wiff", and "sciex_raw" in :obj:`alpharaw.ms_data_base.ms_reader_provider`.
    """
    ms_reader_provider.register_reader("sciex", SciexWiffData)
    ms_reader_provider.register_reader("sciex_wiff", SciexWiffData)
    ms_reader_provider.register_reader("sciex_raw", SciexWiffData)
