"""Bruker TimsTOF data handling.

This package provides the TimsTOF class for reading, slicing, and exporting
Bruker TimsTOF LC-IMS-MS/MS data from .d folders or HDF5 files.
"""

from alpharaw.bruker.filtering import filter_indices
from alpharaw.bruker.timstof import (
    TimsTOF,
    convert_slice_key_to_float_array,
    convert_slice_key_to_int_array,
    parse_keys,
)
from alpharaw.bruker.write import save_as_mgf, save_as_spectra

__all__ = [
    "TimsTOF",
    "convert_slice_key_to_float_array",
    "convert_slice_key_to_int_array",
    "filter_indices",
    "parse_keys",
    "save_as_mgf",
    "save_as_spectra",
]
