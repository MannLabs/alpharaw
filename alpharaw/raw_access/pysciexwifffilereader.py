# ruff: noqa: E402  #Module level import not at top of file
import os

# require pythonnet, pip install pythonnet on Windows
import clr
import numpy as np

from alpharaw.utils.centroiding import naive_centroid

try:
    clr.AddReference("System")

    import System  # noqa: F401
    from System.Globalization import CultureInfo
    from System.Threading import Thread

    from .clr_utils import DotNetArrayToNPArray, ext_dir

    de_fr = CultureInfo("fr-FR")
    other = CultureInfo("en-US")

    Thread.CurrentThread.CurrentCulture = other
    Thread.CurrentThread.CurrentUICulture = other

    clr.AddReference(
        os.path.join(ext_dir, "sciex/Clearcore2.Data.AnalystDataProvider.dll")
    )
    clr.AddReference(os.path.join(ext_dir, "sciex/Clearcore2.Data.dll"))
    clr.AddReference(os.path.join(ext_dir, "sciex/WiffOps4Python.dll"))

    import Clearcore2  # noqa: F401
    import WiffOps4Python  # noqa: F401
    from Clearcore2.Data.AnalystDataProvider import (
        AnalystDataProviderFactory,
        AnalystWiffDataProvider,
    )
    from WiffOps4Python import WiffOps as DotNetWiffOps
except Exception:
    # allows to use the rest of the code without clr
    import traceback

    traceback.print_exc()
    print(
        "Warning: could not import dotnet-based dependencies. Do you have pythonnet and mono (Mac/Linux) installed?"
    )


class WillFileReader:
    def __init__(self, filename: str):
        self._wiffDataProvider = AnalystWiffDataProvider()
        self._wiff_file = AnalystDataProviderFactory.CreateBatch(
            filename, self._wiffDataProvider
        )
        self.sample_names = self._wiff_file.GetSampleNames()

    def close(self):
        self._wiffDataProvider.Close()

    def load_sample(
        self,
        sample_id: int,
        centroid: bool = True,
        centroid_ppm: float = 20.0,
        ignore_empty_scans: bool = True,
        keep_k_peaks: int = 2000,
    ):
        if sample_id < 0 or sample_id >= len(self.sample_names):
            raise ValueError("Incorrect sample number.")
        self.wiffSample = self._wiff_file.GetSample(sample_id)
        self.msSample = self.wiffSample.MassSpectrometerSample

        _peak_indices = []
        peak_mz_array_list = []
        peak_intensity_array_list = []
        rt_list = []
        ms_level_list = []
        precursor_mz_list = []
        precursor_charge_list = []
        ce_list = []
        isolation_lower_mz_list = []
        isolation_upper_mz_list = []

        exp_list = [
            self.msSample.GetMSExperiment(i)
            for i in range(self.msSample.ExperimentCount)
        ]

        for j in range(exp_list[0].Details.NumberOfScans):
            for i in range(self.msSample.ExperimentCount):
                exp = exp_list[i]
                massSpectrum = exp.GetMassSpectrum(j)
                massSpectrumInfo = exp.GetMassSpectrumInfo(j)
                details = exp.Details
                ms_level = massSpectrumInfo.MSLevel
                if (
                    ms_level > 1
                    and not details.IsSwath
                    and massSpectrum.NumDataPoints <= 0
                    and ignore_empty_scans
                ):
                    continue
                mz_array = DotNetArrayToNPArray(massSpectrum.GetActualXValues())
                int_array = DotNetArrayToNPArray(
                    massSpectrum.GetActualYValues()
                ).astype(np.float32)
                if centroid:
                    (mz_array, int_array) = naive_centroid(
                        mz_array, int_array, centroiding_ppm=centroid_ppm
                    )
                if len(mz_array) > keep_k_peaks:
                    idxes = np.argsort(int_array)[-keep_k_peaks:]
                    idxes = np.sort(idxes)
                    mz_array = mz_array[idxes]
                    int_array = int_array[idxes]

                peak_mz_array_list.append(mz_array)
                peak_intensity_array_list.append(int_array)

                _peak_indices.append(len(peak_mz_array_list[-1]))
                rt_list.append(exp.GetRTFromExperimentCycle(j))
                # ScanMode = massSpectrumInfo.CentroidMode ? WiffFile.ScanMode.Centroid : WiffFile.ScanMode.Profile,
                # Polarity = (details.Polarity == MSExperimentInfo.PolarityEnum.Positive) ? WiffFile.Polarity.Positive : WiffFile.Polarity.Negative,
                # low_mz = details.StartMass
                # high_mz = details.EndMass
                ms_level_list.append(ms_level)
                # ScanType = (details.IDAType == MSExperimentInfo.IDAExperimentType.Survey) ? WiffFile.ScanType.MS1 : WiffFile.ScanType.MS2,

                center_mz = -1
                isolation_window = 0
                if ms_level > 1:
                    if details.IsSwath and details.MassRangeInfo.Length > 0:
                        center_mz = DotNetWiffOps.get_center_mz(details)
                        isolation_window = DotNetWiffOps.get_isolation_window(details)
                    if isolation_window <= 0:
                        isolation_window = 3.0
                    if center_mz <= 0:
                        center_mz = massSpectrumInfo.ParentMZ
                    precursor_mz_list.append(center_mz)
                    precursor_charge_list.append(massSpectrumInfo.ParentChargeState)
                    ce_list.append(float(massSpectrumInfo.CollisionEnergy))
                    isolation_lower_mz_list.append(center_mz - isolation_window / 2)
                    isolation_upper_mz_list.append(center_mz + isolation_window / 2)
                else:
                    precursor_mz_list.append(-1.0)
                    precursor_charge_list.append(0)
                    ce_list.append(0)
                    isolation_lower_mz_list.append(-1.0)
                    isolation_upper_mz_list.append(-1.0)

        peak_indices = np.empty(len(rt_list) + 1, np.int64)
        peak_indices[0] = 0
        peak_indices[1:] = np.cumsum(_peak_indices)

        return {
            "peak_indices": peak_indices,
            "peak_mz": np.concatenate(peak_mz_array_list),
            "peak_intensity": np.concatenate(peak_intensity_array_list),
            "rt": np.array(rt_list, dtype=np.float64),
            "ms_level": np.array(ms_level_list, dtype=np.int8),
            "precursor_mz": np.array(precursor_mz_list, dtype=np.float64),
            "precursor_charge": np.array(precursor_charge_list, dtype=np.int8),
            "isolation_lower_mz": np.array(isolation_lower_mz_list),
            "isolation_upper_mz": np.array(isolation_upper_mz_list),
            "nce": np.array(ce_list, dtype=np.float32),
        }
