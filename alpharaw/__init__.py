#!python

import warnings
warnings.filterwarnings("ignore")

def register_readers():
    from .ms_data_base import ms_reader_provider
    from .legacy_msdata import mgf
    from .mzml import MzMLReader
    from .wrappers import alphapept_wrapper
    try:
        from .sciex import SciexWiffData
        from .thermo import ThermoRawData
    except (RuntimeError, ImportError):
        print("[WARN] pythonnet is not installed")

__project__ = "alpharaw"
__version__ = "0.4.2"
__license__ = "Apache"
__description__ = "An open-source Python package to unify raw MS data access and storage."
__author__ = "Mann Labs"
__author_email__ = "jalew.zwf@qq.com"
__github__ = "https://github.com/MannLabs/alpharaw"
__keywords__ = [
    "bioinformatics",
    "software",
    "AlphaPept ecosystem",
    "mass spectrometry",
    "raw data",
    "data access",
    "data storage"
]
__python_version__ = ">=3.8"
__classifiers__ = [
    # "Development Status :: 1 - Planning",
    # "Development Status :: 2 - Pre-Alpha",
    "Development Status :: 3 - Alpha",
    # "Development Status :: 4 - Beta",
    # "Development Status :: 5 - Production/Stable",
    # "Development Status :: 6 - Mature",
    # "Development Status :: 7 - Inactive"
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "GitHub": __github__,
    # "ReadTheDocs": None,
    # "PyPi": None,
    # "Scientific paper": None,
}
__console_scripts__ = [
    "alpharaw=alpharaw.cli:run",
]
__extra_requirements__ = {
    "development": "extra_requirements/development.txt",
}
