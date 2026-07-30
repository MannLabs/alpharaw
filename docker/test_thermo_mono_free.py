"""Smoke test for the mono-free Thermo reading path.

Reads a Thermo ``.raw`` file through the public ``alpharaw.thermo.ThermoRawData``
API and prints the active .NET runtime plus a few values from the parsed data.
Exits non-zero on any failure. A real file (not a heredoc) is required because
``import_raw`` uses ``multiprocessing`` spawn, which re-imports ``__main__``.
"""

import os
import platform
import sys


def main() -> int:
    raw_file = sys.argv[1] if len(sys.argv) > 1 else "nbs_tests/test_data/iRT.raw"
    raw_file = os.path.realpath(raw_file)

    print("Platform:", platform.system(), platform.machine())

    from alpharaw.raw_access import clr_utils

    print("Active .NET runtime:", clr_utils.DOTNET_RUNTIME)

    from alpharaw.thermo import ThermoRawData

    data = ThermoRawData()
    data.import_raw(raw_file)

    n_spectra = len(data.spectrum_df)
    n_peaks = len(data.peak_df)
    print(f"file: {raw_file}")
    print(f"spectra: {n_spectra}")
    print(f"peaks:   {n_peaks}")
    if n_spectra == 0 or n_peaks == 0:
        print("FAIL: no data parsed")
        return 1

    first = data.spectrum_df.iloc[0]
    print(f"scan[0] ms_level={int(first.ms_level)} rt={float(first.rt):.4f}")
    print(f"OK: read {raw_file} via '{clr_utils.DOTNET_RUNTIME}' runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
