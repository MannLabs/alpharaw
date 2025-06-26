"""
Controlled Vocabulary constants for mzML writing.

This module contains CV accession numbers and names from the PSI-MS ontology
as defined at: https://github.com/HUPO-PSI/psi-ms-CV/blob/master/psi-ms.obo

Organized by functional groups for easier maintenance and reference.
"""

# XML attribute names
CV_REF = "cvRef"
ACCESSION = "accession"
NAME = "name"
VALUE = "value"

# CV References
CV_MS = "MS"
CV_UO = "UO"
CV_PSI_MS = "PSI-MS"

# Namespace URIs
NS_URI_MZML = "http://psi.hupo.org/ms/mzml"
NS_URI_XSI = "http://www.w3.org/2001/XMLSchema-instance"

# Schema locations
SCHEMA_LOCATION = "http://psi.hupo.org/ms/mzml http://psi.hupo.org/ms/mzml"

# CV URIs
CV_URI_MS = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
CV_URI_UO = "http://ontologies.berkeleybop.org/uo.obo"
CV_URI_PSI_MS = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"

# CV Names
CV_NAME_MS = "Proteomics Standards Initiative Mass Spectrometry Ontology"
CV_NAME_UO = "Unit Ontology"
CV_NAME_PSI_MS = "PSI-MS Controlled Vocabulary"

# Spectrum types
ACCESSION_CENTROIDED = "MS:1000127"
NAME_CENTROIDED = "centroid spectrum"

ACCESSION_PROFILE = "MS:1000128"
NAME_PROFILE = "profile spectrum"

# MS levels
ACCESSION_MS_LEVEL = "MS:1000511"
NAME_MS_LEVEL = "ms level"

# Charge state
ACCESSION_CHARGE_STATE = "MS:1000041"
NAME_CHARGE_STATE = "charge state"

# File content types
ACCESSION_MS1_SPECTRUM = "MS:1000579"
NAME_MS1_SPECTRUM = "MS1 spectrum"

ACCESSION_MSN_SPECTRUM = "MS:1000580"
NAME_MSN_SPECTRUM = "MSn spectrum"

# Binary data precision
ACCESSION_32BIT_FLOAT = "MS:1000521"
NAME_32BIT_FLOAT = "32-bit float"

ACCESSION_64BIT_FLOAT = "MS:1000523"
NAME_64BIT_FLOAT = "64-bit float"

# Compression
ACCESSION_NO_COMPRESSION = "MS:1000576"
NAME_NO_COMPRESSION = "no compression"

ACCESSION_ZLIB_COMPRESSION = "MS:1000574"
NAME_ZLIB_COMPRESSION = "zlib compression"

# Array types
ACCESSION_MZ_ARRAY = "MS:1000514"
NAME_MZ_ARRAY = "m/z array"

ACCESSION_INTENSITY_ARRAY = "MS:1000515"
NAME_INTENSITY_ARRAY = "intensity array"

# Units
ACCESSION_MZ_UNIT = "MS:1000040"
NAME_MZ_UNIT = "m/z"

ACCESSION_DETECTOR_COUNTS = "MS:1000131"
NAME_DETECTOR_COUNTS = "number of detector counts"

ACCESSION_SECOND = "UO:0000010"
NAME_SECOND = "second"

ACCESSION_ELECTRONVOLT = "UO:0000266"
NAME_ELECTRONVOLT = "electronvolt"

# Time parameters
ACCESSION_SCAN_START_TIME = "MS:1000016"
NAME_SCAN_START_TIME = "scan start time"

# Scan combination
ACCESSION_NO_COMBINATION = "MS:1000795"
NAME_NO_COMBINATION = "no combination"

# Precursor information
ACCESSION_ISOLATION_TARGET_MZ = "MS:1000827"
NAME_ISOLATION_TARGET_MZ = "isolation window target m/z"

ACCESSION_ISOLATION_LOWER_OFFSET = "MS:1000828"
NAME_ISOLATION_LOWER_OFFSET = "isolation window lower offset"

ACCESSION_ISOLATION_UPPER_OFFSET = "MS:1000829"
NAME_ISOLATION_UPPER_OFFSET = "isolation window upper offset"

ACCESSION_SELECTED_ION_MZ = "MS:1000744"
NAME_SELECTED_ION_MZ = "selected ion m/z"

# Collision energy
ACCESSION_COLLISION_ENERGY = "MS:1000045"
NAME_COLLISION_ENERGY = "collision energy"

# Activation methods
ACCESSION_HCD = "MS:1000422"
NAME_HCD = "beam-type collision-induced dissociation"

ACCESSION_CID = "MS:1000133"
NAME_CID = "collision-induced dissociation"

ACCESSION_ETD = "MS:1000598"
NAME_ETD = "electron transfer dissociation"

ACCESSION_ECD = "MS:1000250"
NAME_ECD = "electron capture dissociation"

ACCESSION_PHOTODISSOCIATION = "MS:1000435"
NAME_PHOTODISSOCIATION = "photodissociation"

ACCESSION_DISSOCIATION_METHOD = "MS:1000044"
NAME_DISSOCIATION_METHOD = "dissociation method"

# Software
ACCESSION_ANALYSIS_SOFTWARE = "MS:1001456"
NAME_ANALYSIS_SOFTWARE = "analysis software"

# File formats
ACCESSION_THERMO_RAW = "MS:1000563"
NAME_THERMO_RAW = "Thermo RAW format"

# Data processing
ACCESSION_FILE_FORMAT_CONVERSION = "MS:1000530"
NAME_FILE_FORMAT_CONVERSION = "file format conversion"

# Instrument components
ACCESSION_INSTRUMENT_MODEL = "MS:1000031"
NAME_INSTRUMENT_MODEL = "instrument model"

ACCESSION_ESI = "MS:1000073"
NAME_ESI = "electrospray ionization"

ACCESSION_ELECTRON_MULTIPLIER = "MS:1000253"
NAME_ELECTRON_MULTIPLIER = "electron multiplier"

# Analyzer types
ACCESSION_ORBITRAP = "MS:1000484"
NAME_ORBITRAP = "orbitrap"

ACCESSION_TOF = "MS:1000084"
NAME_TOF = "time-of-flight"

ACCESSION_ION_TRAP = "MS:1000264"
NAME_ION_TRAP = "ion trap"

ACCESSION_FTICR = "MS:1000079"
NAME_FTICR = "fourier transform ion cyclotron resonance mass spectrometer"

ACCESSION_QUADRUPOLE = "MS:1000081"
NAME_QUADRUPOLE = "quadrupole"

# Activation method mapping dictionary
ACTIVATION_METHODS = {
    "HCD": (ACCESSION_HCD, NAME_HCD),
    "CID": (ACCESSION_CID, NAME_CID),
    "ETD": (ACCESSION_ETD, NAME_ETD),
    "ECD": (ACCESSION_ECD, NAME_ECD),
    "EAD": (ACCESSION_ECD, NAME_ECD),  # Using ECD as proxy
    "EXD": (ACCESSION_ECD, NAME_ECD),  # Using ECD as proxy
    "UVPD": (ACCESSION_PHOTODISSOCIATION, NAME_PHOTODISSOCIATION),
    "ETHCD": (ACCESSION_HCD, NAME_HCD),  # Using HCD as proxy
    "ETCID": (ACCESSION_CID, NAME_CID),  # Using CID as proxy
    "EXCID": (ACCESSION_CID, NAME_CID),  # Using CID as proxy
    "NETD": (ACCESSION_ETD, NAME_ETD),  # Using ETD as proxy
}

# Analyzer type mapping dictionary
ANALYZER_TYPES = {
    "orbitrap": (ACCESSION_ORBITRAP, NAME_ORBITRAP),
    "tof": (ACCESSION_TOF, NAME_TOF),
    "it": (ACCESSION_ION_TRAP, NAME_ION_TRAP),
    "ft": (ACCESSION_FTICR, NAME_FTICR),
    "quadrupole": (ACCESSION_QUADRUPOLE, NAME_QUADRUPOLE),
}
