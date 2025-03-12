import numpy as np

from alpharaw.ms_data_base import (
    PEAK_INTENSITY_DTYPE,
    PEAK_MZ_DTYPE,
    MSData_Base,
    index_ragged_list,
    ms_reader_provider,
)


def read_until(file, until):
    lines = []
    while True:
        line = file.readline().strip()
        if line == "" or line.startswith(until):
            break
        else:
            lines.append(line)
    return lines


def find_line(lines, start):
    for line in lines:
        if line.startswith(start):
            return line
    return None


def parse_scan_from_TITLE(mgf_title):
    # raw_name.scan.scan.charge[.xx.xx]
    return int(mgf_title.split(".")[2])


class MGFReader(MSData_Base):
    """MGF Reader (MS2)"""

    def _import(self, _path: str):
        scan_mz_dict = {}
        scan_charge_dict = {}
        masses_list = []
        intens_list = []
        spec_idx_list = []
        scan_list = []
        rt_list = []
        precursor_mz_list = []
        charge_list = []
        self._has_chimeras = False

        f = open(_path) if isinstance(_path, str) else _path  # noqa: SIM115
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("BEGIN IONS"):
                lines = read_until(f, "END IONS")
                masses = []
                intens = []
                scan = None
                RT = 0
                precursor_mz = 0
                charge = 0
                for line in lines:
                    if line[0].isdigit():
                        mass, inten = (float(i) for i in line.strip().split())
                        masses.append(mass)
                        intens.append(inten)
                    elif line.startswith("SCAN="):
                        scan = int(line.split("=")[1])
                    elif line.startswith("RTINSECOND"):
                        RT = float(line.split("=")[1]) / 60
                    elif line.startswith("PEPMASS="):
                        precursor_mz = float(line.split("=")[1])
                    elif line.startswith("CHARGE="):
                        charge = int(line.split("=")[1].strip()[:-1])
                if not scan:
                    title = find_line(lines, "TITLE=")
                    scan = parse_scan_from_TITLE(title)
                if scan in scan_mz_dict:
                    scan_mz_dict[scan].append(precursor_mz)
                    scan_charge_dict[scan].append(charge)
                    self._has_chimeras = True
                    continue
                scan_mz_dict[scan] = [precursor_mz]
                scan_charge_dict[scan] = [charge]
                scan_list.append(scan)
                spec_idx_list.append(scan - 1)
                rt_list.append(RT)
                precursor_mz_list.append(precursor_mz)
                charge_list.append(charge)
                masses_list.append(np.array(masses, dtype=PEAK_MZ_DTYPE))
                intens_list.append(np.array(intens, dtype=PEAK_INTENSITY_DTYPE))
        if isinstance(_path, str):
            f.close()

        if self._has_chimeras:
            precursor_mz_list = [scan_mz_dict[scan] for scan in scan_list]
            charge_list = [scan_charge_dict[scan] for scan in scan_list]

        return {
            "peak_indices": index_ragged_list(masses_list),
            "peak_mz": np.concatenate(masses_list),
            "peak_intensity": np.concatenate(intens_list),
            "rt": np.array(rt_list),
            "precursor_mz": precursor_mz_list,
            "spec_idx": np.array(spec_idx_list, dtype=np.int64),
            "scan": np.array(scan_list, dtype=np.int64),
            "precursor_charge": charge_list,
        }

    def _set_dataframes(self, raw_data: dict):
        spec_idxes = raw_data["spec_idx"]
        spec_num = spec_idxes.max() + 1
        self.create_spectrum_df(spec_num)
        start_idxes = np.full(spec_num, -1, dtype=np.int64)
        start_idxes[spec_idxes] = raw_data["peak_indices"][:-1]
        end_idxes = np.full(spec_num, -1, dtype=np.int64)
        end_idxes[spec_idxes] = raw_data["peak_indices"][1:]
        rt_values = np.zeros(spec_num)
        rt_values[spec_idxes] = raw_data["rt"]
        if self._has_chimeras:
            precursor_mzs = [[]] * spec_num
            charges = [[]] * spec_num
            mz_vals = raw_data["precursor_mz"]
            ch_vals = raw_data["precursor_charge"]
            for i, idx in enumerate(spec_idxes):
                precursor_mzs[idx] = mz_vals[i]
                charges[idx] = ch_vals[i]
        else:
            precursor_mzs = np.zeros(spec_num)
            precursor_mzs[spec_idxes] = raw_data["precursor_mz"]
            charges = np.zeros(spec_num, np.int8)
            charges[spec_idxes] = raw_data["precursor_charge"]

        self.spectrum_df["charge"] = charges
        self.spectrum_df["precursor_mz"] = precursor_mzs

        self.set_peak_df_by_indexed_array(
            raw_data["peak_mz"], raw_data["peak_intensity"], start_idxes, end_idxes
        )
        self.add_column_in_spec_df_by_spec_idxes(
            "rt", raw_data["rt"], spec_idxes, na_value=0
        )
        self.add_column_in_spec_df_by_spec_idxes(
            "ms_level", 2, spec_idxes, dtype=np.int8, na_value=1
        )


def register_readers():
    """
    Register :class:`MGFReader` for file format "mgf" in
    :obj:`alpharaw.ms_data_base.ms_reader_provider`.
    """
    ms_reader_provider.register_reader("mgf", MGFReader)
