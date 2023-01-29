import numpy as np
import re
import datetime
import pathlib

from pyteomics import mzml

from .ms_data_base import MSData_Base

# def parse_mzml_item(item_dict: dict) -> tuple:
#     rt = float(item_dict.get('scanList').get('scan')[0].get('scan start time'))
#     masses = item_dict.get('m/z array')
#     intensities = item_dict.get('intensity array')
#     ms_level = item_dict.get('ms level')
#     prec_mz = -1.0
#     isolation_lower_mz = -1.0
#     isolation_upper_mz = -1.0
#     charge = 0
#     if ms_level == 2:
#         try:
#             charge = int(item_dict.get('precursorList').get('precursor')[0].get('selectedIonList').get('selectedIon')[0].get(
#                 'charge state'))
#         except TypeError:
#             charge = 0
#         try:
#             charge = int(item_dict.get('precursorList').get('precursor')[0].get('selectedIonList').get('selectedIon')[0].get(
#                 'charge state'))
#         except TypeError:
#             charge = 0

#         prec_mz = item_dict.get('precursorList').get('precursor')[0].get('selectedIonList').get('selectedIon')[0].get(
#                 'selected ion m/z')
#         try:
#             iso_lower = int(item_dict.get('precursorList').get('precursor')[0].get('isolation window lower offset').get(
#                 'charge state'))
#             iso_upper = int(item_dict.get('precursorList').get('precursor')[0].get('isolation window upper offset').get(
#                 'charge state'))
#             isolation_upper_mz = prec_mz + iso_upper
#             isolation_lower_mz = prec_mz - iso_lower
#         except TypeError:
#             isolation_upper_mz = prec_mz + 1.5
#             isolation_lower_mz = prec_mz - 1.5
        
#     return (
#         rt, prec_mz, isolation_lower_mz, isolation_upper_mz, 
#         ms_level, charge, masses, intensities
#     )

# class MSFraggerMzML_Data(MSData_Base):
#     def _import(self,
#         filename: str,
#     ):
#         reader = mzml.read(filename, use_index=True)
#         spec_indices = np.array(range(1, len(reader) + 1))

#         scan_list = []
#         rt_list = []
#         mzs_list = []
#         intens_list = []
#         ms_level_list = []
#         prec_mz_list = []
#         charge_list = []

#         self.file_type = 'unknown'
        
#         for idx, i in enumerate(spec_indices):
#             spec = next(reader)
#             if idx == 0:
#                 ext = re.findall(r"File:\".+\.(\w+)\"", spec['spectrum title'])[0]
#                 if ext.lower() == 'raw':
#                     self.file_type = "thermo"

#             scan_list.append(i)
#             rt, prec_mz, ms_level, charge, masses, intensities = parse_mzml_item(spec)
            
#             sortindex = np.argsort(masses)
            
#             masses = masses[sortindex]
#             intensities = intensities[sortindex]
            
#             rt_list.append(rt)
            
#             #Remove zero intensities
#             to_keep = intensities>0
#             masses = masses[to_keep]
#             intensities = intensities[to_keep]
            
#             mzs_list.append(masses)
#             intens_list.append(intensities)
#             ms_level_list.append(ms_level)
#             prec_mz_list.append(prec_mz)
#             charge_list.append(charge)

#         fname = pathlib.Path(filename)
#         acquisition_date_time = datetime.datetime.fromtimestamp(fname.stat().st_mtime).strftime('%Y-%m-%dT%H:%M:%S')

#         return acquisition_date_time