"""Bruker timsdata DLL/SO interface for native library access."""

import os
import sys
import contextlib
import logging

from alpharaw.utils.pjit import MAX_THREADS

BASE_PATH = os.path.dirname(__file__)
EXT_PATH = os.path.abspath(os.path.join(BASE_PATH, "..", "ext", "bruker"))

if sys.platform[:5] == "win32":
    BRUKER_DLL_FILE_NAME = os.path.join(
        EXT_PATH,
        "timsdata.dll"
    )
elif sys.platform[:5] == "linux":
    BRUKER_DLL_FILE_NAME = os.path.join(
        EXT_PATH,
        "timsdata.so"
    )
else:
    BRUKER_DLL_FILE_NAME = ""


def init_bruker_dll(bruker_dll_file_name: str = BRUKER_DLL_FILE_NAME):
    """Open a bruker.dll in Python.

    Five functions are defined for this dll:

        - tims_open: [c_char_p, c_uint32] -> c_uint64
        - tims_close: [c_char_p, c_uint32] -> c_uint64
        - tims_read_scans_v2: [c_uint64, c_int64, c_uint32, c_uint32, c_void_p, c_uint32] -> c_uint32
        - tims_index_to_mz: [c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32] -> None
        - tims_scannum_to_oneoverk0: Same as "tims_index_to_mz"

    Parameters
    ----------
    bruker_dll_file_name : str
        The absolute path to the timsdata.dll.
        Default is BRUKER_DLL_FILE_NAME.

    Returns
    -------
    : ctypes.cdll
        The Bruker dll library.
    """
    import ctypes
    bruker_dll = ctypes.cdll.LoadLibrary(
        os.path.realpath(bruker_dll_file_name)
    )
    bruker_dll.tims_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    bruker_dll.tims_open.restype = ctypes.c_uint64
    bruker_dll.tims_close.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_close.restype = None
    bruker_dll.tims_read_scans_v2.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32
    ]
    bruker_dll.tims_read_scans_v2.restype = ctypes.c_uint32
    bruker_dll.tims_index_to_mz.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32
    ]
    bruker_dll.tims_index_to_mz.restype = ctypes.c_uint32
    bruker_dll.tims_scannum_to_oneoverk0.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32
    ]
    bruker_dll.tims_scannum_to_oneoverk0.restype = ctypes.c_uint32
    bruker_dll.tims_set_num_threads.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_set_num_threads.restype = None
    bruker_dll.tims_set_num_threads(MAX_THREADS)
    # multiple threads is equally fast as just 1 for io?
    # bruker_dll.tims_set_num_threads(1)
    return bruker_dll


@contextlib.contextmanager
def open_bruker_d_folder(
    bruker_d_folder_name: str,
    bruker_dll_file_name=BRUKER_DLL_FILE_NAME,
) -> tuple:
    """A context manager for a bruker dll connection to a .d folder.

    Parameters
    ----------
    bruker_d_folder_name : str
        The name of a Bruker .d folder.
    bruker_dll_file_name : str, ctypes.cdll
        The path to Bruker' timsdata.dll library.
        Alternatively, the library itself can be passed as argument.
        Default is BRUKER_DLL_FILE_NAME,
        which in itself is dependent on the OS.

    Returns
    -------
    : tuple (ctypes.cdll, int).
        The opened bruker dll and identifier of the .d folder.
    """
    try:
        if isinstance(bruker_dll_file_name, str):
            bruker_dll = init_bruker_dll(bruker_dll_file_name)
        logging.info(f"Opening handle for {bruker_d_folder_name}")
        bruker_d_folder_handle = bruker_dll.tims_open(
            bruker_d_folder_name.encode('utf-8'),
            0
        )
        yield bruker_dll, bruker_d_folder_handle
    finally:
        logging.info(f"Closing handle for {bruker_d_folder_name}")
        bruker_dll.tims_close(bruker_d_folder_handle)

