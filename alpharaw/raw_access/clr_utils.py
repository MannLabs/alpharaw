import clr
import os
import numpy as np

clr.AddReference('System')
# see https://github.com/mobiusklein/ms_deisotope/blob/90b817d4b5ae7823cfe4ad61c57119d62a6e3d9d/ms_deisotope/data_source/thermo_raw_net.py#L217
from System.Runtime.InteropServices import Marshal
from System import IntPtr, Int64

def DotNetArrayToNPArray(src):
    '''A quick and dirty implementation of the fourth technique shown in
    https://mail.python.org/pipermail/pythondotnet/2014-May/001525.html for
    copying a .NET Array[Double] to a NumPy ndarray[np.float64] via a raw
    memory copy.
    ``int_ptr_tp`` must be an integer type that can hold a pointer. On Python 2
    this is :class:`long`, and on Python 3 it is :class:`int`.
    '''
    # When the input .NET array pointer is None, return an empty array. On Py2
    # this would happen automatically, but not on Py3, and perhaps not safely on
    # all Py2 because it relies on pythonnet and the .NET runtime properly checking
    # for nulls.
    if src is None:
        return np.array([], dtype=np.float64)
    dest = np.empty(len(src), dtype=np.float64)
    Marshal.Copy(
        src, 0,
        IntPtr.__overloads__[Int64](dest.__array_interface__['data'][0]),
        len(src))
    return dest

ext_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    'ext'
)