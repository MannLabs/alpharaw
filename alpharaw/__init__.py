#!python

import warnings

warnings.filterwarnings("ignore")


def register_all_readers():
    from .legacy_msdata.mgf import register_readers as register_mgf_readers
    from .mzml import register_readers as register_mzml_readers

    register_mzml_readers()
    register_mgf_readers()

    try:
        from .sciex import register_readers as register_wiff_readers
        from .thermo import register_readers as register_raw_readers

        register_wiff_readers()
        register_raw_readers()
    except (RuntimeError, ImportError):
        print("[WARN] pythonnet is not installed")


__version__ = "0.4.6"
