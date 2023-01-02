import clr
import os
import numpy as np

clr.AddReference('System')
from System.Runtime.InteropServices import Marshal
from System import IntPtr, Int64

def DotNetArrayToNPArray(src):
    '''
    See https://github.com/mobiusklein/ms_deisotope/blob/90b817d4b5ae7823cfe4ad61c57119d62a6e3d9d/ms_deisotope/data_source/thermo_raw_net.py#L217
    '''
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