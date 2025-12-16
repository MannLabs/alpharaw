"""Controlled Vocabulary constants for mzML writing.

This module contains CV accession numbers, static constants, and the processor
for extracting official labels from the PSI-MS OWL file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from owlready2 import get_ontology


class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(self, name, value):
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]


class CV(metaclass=ConstantsClass):
    """String constants for CV-related attributes."""

    REF = "cvRef"
    ACCESSION = "accession"
    NAME = "name"
    VALUE = "value"
    MS = "MS"
    UO = "UO"
    PSI_MS = "PSI-MS"
    NAME_MS = "Proteomics Standards Initiative Mass Spectrometry Ontology"
    NAME_UO = "Unit Ontology"
    NAME_PSI_MS = "PSI-MS Controlled Vocabulary"


class XML(metaclass=ConstantsClass):
    """String constants for XML-related attributes."""

    NS_URI_MZML = "http://psi.hupo.org/ms/mzml"
    NS_URI_XSI = "http://www.w3.org/2001/XMLSchema-instance"
    SCHEMA_LOCATION = "http://psi.hupo.org/ms/mzml http://psi.hupo.org/ms/mzml"
    URI_MS = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
    URI_UO = "http://ontologies.berkeleybop.org/uo.obo"
    URI_PSI_MS = (
        "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
    )


class CVTerms(metaclass=ConstantsClass):
    """String constants for CV terms."""

    CENTROIDED = "CENTROIDED"
    PROFILE = "PROFILE"
    MS_LEVEL = "MS_LEVEL"
    CHARGE_STATE = "CHARGE_STATE"
    MS1_SPECTRUM = "MS1_SPECTRUM"
    MSN_SPECTRUM = "MSN_SPECTRUM"
    FLOAT_32BIT = "32BIT_FLOAT"
    FLOAT_64BIT = "64BIT_FLOAT"
    NO_COMPRESSION = "NO_COMPRESSION"
    ZLIB_COMPRESSION = "ZLIB_COMPRESSION"
    MZ_ARRAY = "MZ_ARRAY"
    INTENSITY_ARRAY = "INTENSITY_ARRAY"
    MZ_UNIT = "MZ_UNIT"
    DETECTOR_COUNTS = "DETECTOR_COUNTS"
    SECOND = "SECOND"
    ELECTRONVOLT = "ELECTRONVOLT"
    SCAN_START_TIME = "SCAN_START_TIME"
    NO_COMBINATION = "NO_COMBINATION"
    ISOLATION_TARGET_MZ = "ISOLATION_TARGET_MZ"
    ISOLATION_LOWER_OFFSET = "ISOLATION_LOWER_OFFSET"
    ISOLATION_UPPER_OFFSET = "ISOLATION_UPPER_OFFSET"
    SELECTED_ION_MZ = "SELECTED_ION_MZ"
    COLLISION_ENERGY = "COLLISION_ENERGY"
    HCD = "HCD"
    CID = "CID"
    ETD = "ETD"
    ECD = "ECD"
    PHOTODISSOCIATION = "PHOTODISSOCIATION"
    DISSOCIATION_METHOD = "DISSOCIATION_METHOD"
    ANALYSIS_SOFTWARE = "ANALYSIS_SOFTWARE"
    THERMO_RAW = "THERMO_RAW"
    FILE_FORMAT_CONVERSION = "FILE_FORMAT_CONVERSION"
    INSTRUMENT_MODEL = "INSTRUMENT_MODEL"
    ESI = "ESI"
    ELECTRON_MULTIPLIER = "ELECTRON_MULTIPLIER"
    ORBITRAP = "ORBITRAP"
    TOF = "TOF"
    ION_TRAP = "ION_TRAP"
    FTICR = "FTICR"
    QUADRUPOLE = "QUADRUPOLE"


# ============================================================================
# CV TERM MAPPINGS (variable names to MS IDs)
# ============================================================================

CV_TERM_MAPPING = {
    # Spectrum types
    CVTerms.CENTROIDED: "MS:1000127",
    CVTerms.PROFILE: "MS:1000128",
    # MS levels
    CVTerms.MS_LEVEL: "MS:1000511",
    # Charge state
    CVTerms.CHARGE_STATE: "MS:1000041",
    # File content types
    CVTerms.MS1_SPECTRUM: "MS:1000579",
    CVTerms.MSN_SPECTRUM: "MS:1000580",
    # Binary data precision
    CVTerms.FLOAT_32BIT: "MS:1000521",
    CVTerms.FLOAT_64BIT: "MS:1000523",
    # Compression
    CVTerms.NO_COMPRESSION: "MS:1000576",
    CVTerms.ZLIB_COMPRESSION: "MS:1000574",
    # Array types
    CVTerms.MZ_ARRAY: "MS:1000514",
    CVTerms.INTENSITY_ARRAY: "MS:1000515",
    # Units
    CVTerms.MZ_UNIT: "MS:1000040",
    CVTerms.DETECTOR_COUNTS: "MS:1000131",
    CVTerms.SECOND: "UO:0000010",
    CVTerms.ELECTRONVOLT: "UO:0000266",
    # Time parameters
    CVTerms.SCAN_START_TIME: "MS:1000016",
    # Scan combination
    CVTerms.NO_COMBINATION: "MS:1000795",
    # Precursor information
    CVTerms.ISOLATION_TARGET_MZ: "MS:1000827",
    CVTerms.ISOLATION_LOWER_OFFSET: "MS:1000828",
    CVTerms.ISOLATION_UPPER_OFFSET: "MS:1000829",
    CVTerms.SELECTED_ION_MZ: "MS:1000744",
    # Collision energy
    CVTerms.COLLISION_ENERGY: "MS:1000045",
    # Activation methods
    CVTerms.HCD: "MS:1000422",
    CVTerms.CID: "MS:1000133",
    CVTerms.ETD: "MS:1000598",
    CVTerms.ECD: "MS:1000250",
    CVTerms.PHOTODISSOCIATION: "MS:1000435",
    CVTerms.DISSOCIATION_METHOD: "MS:1000044",
    # Software
    CVTerms.ANALYSIS_SOFTWARE: "MS:1001456",
    # File formats
    CVTerms.THERMO_RAW: "MS:1000563",
    # Data processing
    CVTerms.FILE_FORMAT_CONVERSION: "MS:1000530",
    # Instrument components
    CVTerms.INSTRUMENT_MODEL: "MS:1000031",
    CVTerms.ESI: "MS:1000073",
    CVTerms.ELECTRON_MULTIPLIER: "MS:1000253",
    # Analyzer types
    CVTerms.ORBITRAP: "MS:1000484",
    CVTerms.TOF: "MS:1000084",
    CVTerms.ION_TRAP: "MS:1000264",
    CVTerms.FTICR: "MS:1000079",
    CVTerms.QUADRUPOLE: "MS:1000081",
}


# ============================================================================
# ACTIVATION METHODS MAPPING (method names to variable names)
# ============================================================================

ACTIVATION_METHODS_MAPPING = {
    # Direct mappings
    "HCD": CVTerms.HCD,
    "CID": CVTerms.CID,
    "ETD": CVTerms.ETD,
    "ECD": CVTerms.ECD,
    "UVPD": CVTerms.PHOTODISSOCIATION,
    # Proxy mappings (use same CV term as base method)
    "EAD": CVTerms.ECD,
    "EXD": CVTerms.ECD,
    "ETHCD": CVTerms.HCD,
    "ETCID": CVTerms.CID,
    "EXCID": CVTerms.CID,
    "NETD": CVTerms.ETD,
}


# ============================================================================
# ANALYZER TYPES MAPPING (analyzer names to variable names)
# ============================================================================

ANALYZER_TYPES_MAPPING = {
    "orbitrap": CVTerms.ORBITRAP,
    "tof": CVTerms.TOF,
    "it": CVTerms.ION_TRAP,
    "ft": CVTerms.FTICR,
    "quadrupole": CVTerms.QUADRUPOLE,
}


# ============================================================================
# CV KEY GENERATORS
# ============================================================================


ACCESSION_PREFIX = "ACCESSION_"
NAME_PREFIX = "NAME_"


def get_accession_key(name: str) -> str:
    """Get the accession key for a given CV term name."""
    return f"{ACCESSION_PREFIX}{name}"


def get_name_key(name: str) -> str:
    """Get the name key for a given CV term name."""
    return f"{NAME_PREFIX}{name}"


# ============================================================================
# CV TERM PROCESSOR CLASS
# ============================================================================


class CVTermProcessor:
    """Processor for extracting CV terms from PSI-MS OWL files.

    This class reads the official PSI-MS OWL file using owlready2 and provides
    methods to look up CV terms and generate constants dictionaries using the
    predefined mappings above.
    """

    def __init__(self, owl_file_path: str | Path) -> None:
        """Initialize the CV processor with an OWL file.

        Parameters
        ----------
        owl_file_path : str | Path
            Path to the PSI-MS OWL file

        """
        self.owl_file_path = Path(owl_file_path)
        if not self.owl_file_path.exists():
            msg = f"OWL file not found: {self.owl_file_path}"
            raise FileNotFoundError(msg)

        try:
            # Load the ontology
            self.onto = get_ontology(f"file://{self.owl_file_path.absolute()}").load()
        except Exception as e:
            msg = f"Failed to load OWL file: {e}"
            raise Exception(msg) from e

    def get_cv_term_info(self, ms_id: str) -> dict[str, str] | None:
        """Get CV term information by MS ID.

        Parameters
        ----------
        ms_id : str
            The MS ID (e.g., "MS:1000127" or "MS_1000127")

        Returns
        -------
        dict[str, str] | None
            Dictionary with 'id' and 'label' keys, or None if not found

        """
        # Convert MS:1000127 format to MS_1000127 for OWL search
        search_id = ms_id.replace(":", "_")

        # Search for the term in the ontology
        result = self.onto.search_one(iri=f"*{search_id}")

        if result is None:
            return None

        # Extract label
        label = "Unknown term"
        if hasattr(result, "label") and result.label:
            label = result.label[0]

        return {
            "id": ms_id.replace("_", ":"),  # Convert back to MS:1000127 format
            "label": label,
        }

    def process_cv_mapping(
        self, cv_mapping: dict[str, str]
    ) -> dict[str, dict[str, str]]:
        """Process a mapping of variable names to MS IDs and return full CV information.

        Parameters
        ----------
        cv_mapping : dict[str, str]
            Dictionary mapping variable names to MS IDs

        Returns
        -------
        dict[str, dict[str, str]]
            Dictionary with variable names as keys and CV term info as values

        """
        result = {}

        for var_name, ms_id in cv_mapping.items():
            cv_term = self.get_cv_term_info(ms_id)
            if cv_term:
                result[var_name] = cv_term
            else:
                # Fallback for missing terms
                result[var_name] = {
                    "id": ms_id.replace("_", ":"),
                    "label": f"Unknown term {ms_id}",
                }

        return result

    def generate_cv_constants(self) -> dict[str, Any]:
        """Generate a complete CV constants dictionary using the predefined mappings.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all CV constants with official labels

        """
        # Process the predefined CV mapping to get full term information
        cv_terms = self.process_cv_mapping(CV_TERM_MAPPING)

        # Start with static constants
        constants = {
            CV.REF: CV.REF,
            CV.ACCESSION: CV.ACCESSION,
            CV.NAME: CV.NAME,
            CV.VALUE: CV.VALUE,
            CV.MS: CV.MS,
            CV.UO: CV.UO,
            CV.PSI_MS: CV.PSI_MS,
            XML.NS_URI_MZML: XML.NS_URI_MZML,
            XML.NS_URI_XSI: XML.NS_URI_XSI,
            XML.SCHEMA_LOCATION: XML.SCHEMA_LOCATION,
            XML.URI_MS: XML.URI_MS,
            XML.URI_UO: XML.URI_UO,
            XML.URI_PSI_MS: XML.URI_PSI_MS,
            CV.NAME_MS: CV.NAME_MS,
            CV.NAME_UO: CV.NAME_UO,
            CV.NAME_PSI_MS: CV.NAME_PSI_MS,
        }

        # Add accession and name constants for each CV term
        for var_name, term_info in cv_terms.items():
            constants[get_accession_key(var_name)] = term_info["id"]
            constants[get_name_key(var_name)] = term_info["label"]

        # Create activation methods mapping using predefined mapping
        activation_methods = {}
        for method_name, var_name in ACTIVATION_METHODS_MAPPING.items():
            if var_name in cv_terms:
                activation_methods[method_name] = (
                    cv_terms[var_name]["id"],
                    cv_terms[var_name]["label"],
                )

        constants["ACTIVATION_METHODS"] = activation_methods

        # Create analyzer types mapping using predefined mapping
        analyzer_types = {}
        for analyzer_name, var_name in ANALYZER_TYPES_MAPPING.items():
            if var_name in cv_terms:
                analyzer_types[analyzer_name] = (
                    cv_terms[var_name]["id"],
                    cv_terms[var_name]["label"],
                )

        constants["ANALYZER_TYPES"] = analyzer_types

        return constants


def create_cv_constants_from_owl(owl_file_path: str | Path) -> dict[str, Any]:
    """Create CV constants dictionary from OWL file using predefined mappings.

    Parameters
    ----------
    owl_file_path : str | Path
        Path to the PSI-MS OWL file

    Returns
    -------
    dict[str, Any]
        Dictionary containing all CV constants with official labels

    """
    processor = CVTermProcessor(owl_file_path)
    return processor.generate_cv_constants()
