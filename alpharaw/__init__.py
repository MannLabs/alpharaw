#!python


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
    except (RuntimeError, ImportError) as e:
        print(e)
        print(
            "Error importing Thermo and/or Sciex readers. Is pythonnet installed correctly?"
        )


__version__ = "0.5.0-dev0"
