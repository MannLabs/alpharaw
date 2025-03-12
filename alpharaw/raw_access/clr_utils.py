# ruff: noqa: E402  #Module level import not at top of file
import os
import warnings

import numpy as np

try:
    import clr

    clr.AddReference("System")

    import ctypes

    from System.Runtime.InteropServices import GCHandle, GCHandleType
except Exception:
    # allows to use the rest of the code without clr
    import traceback

    traceback.print_exc()
    warnings.warn(
        "Dotnet-based dependencies not installed. Do you have pythonnet and mono (Mac/Linux) installed?"
    )

# from System.Runtime.InteropServices import Marshal
# from System import IntPtr, Int64
# def DotNetArrayToNPArray(src):
#     '''
#     See https://github.com/mobiusklein/ms_deisotope/blob/90b817d4b5ae7823cfe4ad61c57119d62a6e3d9d/ms_deisotope/data_source/thermo_raw_net.py#L217
#     '''
#     if src is None:
#         return np.array([], dtype=np.float64)
#     dest = np.empty(len(src), dtype=np.float64)
#     Marshal.Copy(
#         src, 0,
#         IntPtr.__overloads__[Int64](dest.__array_interface__['data'][0]),
#         len(src))
#     return dest


def DotNetArrayToNPArray(src):
    """
    See https://mail.python.org/pipermail/pythondotnet/2014-May/001527.html
    """
    if src is None:
        return np.array([], dtype=np.float64)
    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_double * len(src)
        cbuf = bufType.from_address(src_ptr)
        dest = np.frombuffer(cbuf, dtype=cbuf._type_).copy()
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()
        return dest  # noqa: B012


ext_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ext")
