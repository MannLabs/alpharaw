import numpy as np

from alpharaw.ms_data_base import (
    index_ragged_list, MSData_Base,
    ms_reader_provider
)

def read_until(file, until):
    lines = []
    while True:
        line = file.readline().strip()
        if line == "": break
        elif line.startswith(until):
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
    return int(mgf_title.split('.')[2])

class MGFReader(MSData_Base):
    """MGF Reader (MS2)"""
    def _import(self, _path:str):
        if isinstance(_path, str):
            f = open(_path)
        else:
            f = _path
        scanset = set()
        masses_list = []
        intens_list = []
        spec_idx_list = []
        rt_list = []
        precursor_mz_list = []
        charge_list = []
        while True:
            line = f.readline()
            if not line: break
            if line.startswith('BEGIN IONS'):
                lines = read_until(f, 'END IONS')
                masses = []
                intens = []
                scan = None
                RT = 0
                precursor_mz = 0
                charge = 0
                for line in lines:
                    if line[0].isdigit():
                        mass,inten = [float(i) for i in line.strip().split()]
                        masses.append(mass)
                        intens.append(inten)
                    elif line.startswith('SCAN='):
                        scan = int(line.split('=')[1])
                    elif line.startswith('RTINSECOND'):
                        RT = float(line.split('=')[1])/60
                    elif line.startswith('PEPMASS='):
                        precursor_mz = float(line.split('=')[1])
                    elif line.startswith('CHARGE='):
                        charge = float(line.split('=')[1].strip()[:-1])
                if not scan:
                    title = find_line(lines, 'TITLE=')
                    scan = parse_scan_from_TITLE(title)
                if scan in scanset: continue
                scanset.add(scan)
                spec_idx_list.append(scan-1)
                rt_list.append(RT)
                precursor_mz_list.append(precursor_mz)
                charge_list.append(charge)
                masses_list.append(np.array(masses))
                intens_list.append(np.array(intens))
        if isinstance(_path, str): 
            f.close()

        precursor_mz_list = np.array(precursor_mz_list)

        return {
            'peak_indices': index_ragged_list(masses_list),
            'peak_mz': np.concatenate(masses_list),
            'peak_intensity': np.concatenate(intens_list),
            'rt': np.array(rt_list),
            'precursor_mz': precursor_mz_list,
            'isolation_mz_lower': precursor_mz_list-2,
            'isolation_mz_upper': precursor_mz_list+2,
            'spec_idx': np.array(spec_idx_list, dtype=np.int64),
            'precursor_charge': np.array(charge_list, dtype=np.int8),
        }

    def _set_dataframes(self, raw_data:dict):
        spec_idxes = raw_data['spec_idx']
        spec_num = spec_idxes.max()+1
        self.create_spectrum_df(spec_num)
        start_idxes = np.full(spec_num, -1, dtype=np.int64)
        start_idxes[spec_idxes] = raw_data['peak_indices'][:-1]
        end_idxes = np.full(spec_num, -1, dtype=np.int64)
        end_idxes[spec_idxes] = raw_data['peak_indices'][1:]
        rt_values = np.zeros(spec_num)
        rt_values[spec_idxes] = raw_data['rt']
        precursor_mzs = np.zeros(spec_num)
        precursor_mzs[spec_idxes] = raw_data['precursor_mz']
        mz_lowers = np.zeros(spec_num)
        mz_lowers[spec_idxes] = raw_data['isolation_mz_lower']
        mz_uppers = np.zeros(spec_num)
        mz_uppers[spec_idxes] = raw_data['isolation_mz_upper']
        charges = np.zeros(spec_num, np.int8)
        charges[spec_idxes] = raw_data['precursor_charge']

        self.set_peaks_by_cat_array(
            raw_data['peak_mz'],
            raw_data['peak_intensity'],
            start_idxes,end_idxes
        )
        self.add_column_in_spec_df('rt', rt_values)
        self.add_column_in_spec_df('charge', charges)
        self.spectrum_df['ms_level'] = 2
        self.set_precursor_mz(
            precursor_mzs
        )
        self.set_precursor_mz_windows(
            mz_lowers,mz_uppers
        )

ms_reader_provider.register_reader('mgf', MGFReader)
