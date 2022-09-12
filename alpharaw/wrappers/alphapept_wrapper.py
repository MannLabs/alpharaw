import numpy as np

from alpharaw.ms_data_base import MSData_Base

def get_peak_lists(starts, ends, peak_df):
    mass_list = [peak_df.mz.values[start:end].tolist() for start,end in zip(starts,ends)]
    inten_list = [peak_df.intensity.values[start:end].tolist() for start,end in zip(starts,ends)]
    return mass_list, inten_list

def extract_ms1(raw_data:MSData_Base, query_data:dict):
    spec_df = raw_data.spectrum_df.query("ms_level==1")
    scans = spec_df.spec_idx.values
    rts = spec_df.rt.values
    ms_levels = spec_df.ms_level.values

    mass_list_ms1, int_list_ms1 = get_peak_lists(
        spec_df.peak_start_idx.values, 
        spec_df.peak_end_idx.values,
        raw_data.peak_df
    )

    query_data["scan_list_ms1"] = scans
    query_data["rt_list_ms1"] = rts
    query_data["mass_list_ms1"] = np.array(mass_list_ms1, dtype=object)
    query_data["int_list_ms1"] = np.array(int_list_ms1, dtype=object)
    query_data["ms_list_ms1"] = ms_levels

def extract_ms2(raw_data:MSData_Base, query_data:dict):
    spec_df = raw_data.spectrum_df.query("ms_level==2")
    scans = spec_df.spec_idx.values
    rts = spec_df.rt.values
    ms_levels = spec_df.ms_level.values
    mono_mzs2 = spec_df.precursor_mz.values
    charges = spec_df.charge.values
    charges[charges<=0] = 2


    mass_list_ms2, int_list_ms2 = get_peak_lists(
        spec_df.peak_start_idx.values, 
        spec_df.peak_end_idx.values,
        raw_data.peak_df
    )

    query_data["scan_list_ms2"] = scans
    query_data["rt_list_ms2"] = rts
    query_data["mass_list_ms2"] = mass_list_ms2
    query_data["int_list_ms2"] = int_list_ms2
    query_data["ms_list_ms2"] = ms_levels
    query_data["prec_mass_list2"] = mass_list_ms2
    query_data["mono_mzs2"] = mono_mzs2
    query_data["charge2"] = charges

def parse_msdata_to_alphapept(raw_data:MSData_Base):
    query_data = {}
    extract_ms1(raw_data, query_data)
    extract_ms2(raw_data, query_data)

    return query_data, raw_data.creation_time