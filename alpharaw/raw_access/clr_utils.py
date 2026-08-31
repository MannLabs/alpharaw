# ruff: noqa: E402  #Module level import not at top of file
import atexit
import os
import warnings

import numpy as np

# Environment variable to force a specific .NET runtime backend ("coreclr",
# "netfx" or "mono"). When set, only that runtime is tried (no fallback).
DOTNET_RUNTIME_ENV = "ALPHARAW_DOTNET_RUNTIME"
# Auto-selection order when the env var is unset: prefer a .NET Framework runtime
# (Windows' built-in "netfx", then Mono) because it supports every reader.
# Fall back to the mono-free "coreclr" runtime (Thermo .NET 8 build only) when no .NET Framework runtime is installed.
# IMPORTANT: pythonnet's runtime is process-global, so this single choice applies to every
# .NET-based reader loaded in the process.
DOTNET_RUNTIME_FALLBACK_ORDER = ["netfx", "mono", "coreclr"]

# Name of the .NET runtime actually loaded; None if none could be loaded.
DOTNET_RUNTIME = None


def _load_dotnet_runtime():
    """Load a .NET runtime for pythonnet and return its name.

    Honors an explicit ``ALPHARAW_DOTNET_RUNTIME`` override; otherwise tries a
    .NET Framework runtime (netfx/Mono) first and falls back to the mono-free
    coreclr runtime. Each candidate is created via ``clr_loader`` (which raises
    when the runtime is unavailable) so ``pythonnet.load`` is called only once,
    with a runtime that is actually present.
    """
    import pythonnet
    from clr_loader import get_coreclr, get_mono, get_netfx

    factories = {"coreclr": get_coreclr, "netfx": get_netfx, "mono": get_mono}
    explicit = os.environ.get(DOTNET_RUNTIME_ENV)
    order = [explicit.lower()] if explicit else DOTNET_RUNTIME_FALLBACK_ORDER

    errors = {}
    for name in order:
        factory = factories.get(name)
        if factory is None:
            errors[name] = ValueError(f"unknown runtime '{name}'")
            continue
        try:
            runtime = factory()
        except Exception as e:
            errors[name] = e
            continue
        pythonnet.load(runtime)
        return name

    raise RuntimeError(f"No .NET runtime available (tried {order}): {errors}")


try:
    DOTNET_RUNTIME = _load_dotnet_runtime()

    import pythonnet

    # pythonnet.load(runtime) registers pythonnet.unload as an atexit hook, which fails
    # under mono and prints a spurious traceback.
    if DOTNET_RUNTIME == "mono":
        atexit.unregister(pythonnet.unload)

    import clr

    clr.AddReference("System")

    import ctypes

    from System.Reflection import Assembly
    from System.Runtime.InteropServices import GCHandle, GCHandleType
except Exception as e:
    # allows to use the rest of the code without clr; surface the underlying error
    # so runtime-selection failures (e.g. coreclr not found) are diagnosable.
    warnings.warn(
        f".NET dependencies could not be loaded. Thermo (.raw) and Sciex (.wiff) support disabled.\n{e!r}"
    )


def load_dotnet_assembly(dll_path: str):
    """Load a .NET assembly from a file path under the active runtime.

    ``Assembly.LoadFile`` resolves correctly under both the coreclr and mono
    runtimes, whereas ``clr.AddReference`` with a path is rejected by the coreclr
    assembly resolver.
    """
    return Assembly.LoadFile(os.path.abspath(dll_path))


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
