"""Controlled Vocabulary constants for mzML writing.

This module contains CV accession numbers, static constants, and the processor
for extracting official labels from the PSI-MS OWL file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from owlready2 import get_ontology

# ============================================================================
# STATIC CONSTANTS (XML attributes, namespaces, etc.)
# ============================================================================

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


# ============================================================================
# CV TERM MAPPINGS (variable names to MS IDs)
# ============================================================================

CV_TERM_MAPPING = {
    # Spectrum types
    "CENTROIDED": "MS:1000127",
    "PROFILE": "MS:1000128",
    # MS levels
    "MS_LEVEL": "MS:1000511",
    # Charge state
    "CHARGE_STATE": "MS:1000041",
    # File content types
    "MS1_SPECTRUM": "MS:1000579",
    "MSN_SPECTRUM": "MS:1000580",
    # Binary data precision
    "32BIT_FLOAT": "MS:1000521",
    "64BIT_FLOAT": "MS:1000523",
    # Compression
    "NO_COMPRESSION": "MS:1000576",
    "ZLIB_COMPRESSION": "MS:1000574",
    # Array types
    "MZ_ARRAY": "MS:1000514",
    "INTENSITY_ARRAY": "MS:1000515",
    # Units
    "MZ_UNIT": "MS:1000040",
    "DETECTOR_COUNTS": "MS:1000131",
    "SECOND": "UO:0000010",
    "ELECTRONVOLT": "UO:0000266",
    # Time parameters
    "SCAN_START_TIME": "MS:1000016",
    # Scan combination
    "NO_COMBINATION": "MS:1000795",
    # Precursor information
    "ISOLATION_TARGET_MZ": "MS:1000827",
    "ISOLATION_LOWER_OFFSET": "MS:1000828",
    "ISOLATION_UPPER_OFFSET": "MS:1000829",
    "SELECTED_ION_MZ": "MS:1000744",
    # Collision energy
    "COLLISION_ENERGY": "MS:1000045",
    # Activation methods
    "HCD": "MS:1000422",
    "CID": "MS:1000133",
    "ETD": "MS:1000598",
    "ECD": "MS:1000250",
    "PHOTODISSOCIATION": "MS:1000435",
    "DISSOCIATION_METHOD": "MS:1000044",
    # Software
    "ANALYSIS_SOFTWARE": "MS:1001456",
    # File formats
    "THERMO_RAW": "MS:1000563",
    # Data processing
    "FILE_FORMAT_CONVERSION": "MS:1000530",
    # Instrument components
    "INSTRUMENT_MODEL": "MS:1000031",
    "ESI": "MS:1000073",
    "ELECTRON_MULTIPLIER": "MS:1000253",
    # Analyzer types
    "ORBITRAP": "MS:1000484",
    "TOF": "MS:1000084",
    "ION_TRAP": "MS:1000264",
    "FTICR": "MS:1000079",
    "QUADRUPOLE": "MS:1000081",
}


# ============================================================================
# ACTIVATION METHODS MAPPING (method names to variable names)
# ============================================================================

ACTIVATION_METHODS_MAPPING = {
    # Direct mappings
    "HCD": "HCD",
    "CID": "CID",
    "ETD": "ETD",
    "ECD": "ECD",
    "UVPD": "PHOTODISSOCIATION",
    # Proxy mappings (use same CV term as base method)
    "EAD": "ECD",
    "EXD": "ECD",
    "ETHCD": "HCD",
    "ETCID": "CID",
    "EXCID": "CID",
    "NETD": "ETD",
}


# ============================================================================
# ANALYZER TYPES MAPPING (analyzer names to variable names)
# ============================================================================

ANALYZER_TYPES_MAPPING = {
    "orbitrap": "ORBITRAP",
    "tof": "TOF",
    "it": "ION_TRAP",
    "ft": "FTICR",
    "quadrupole": "QUADRUPOLE",
}


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
            "CV_REF": CV_REF,
            "ACCESSION": ACCESSION,
            "NAME": NAME,
            "VALUE": VALUE,
            "CV_MS": CV_MS,
            "CV_UO": CV_UO,
            "CV_PSI_MS": CV_PSI_MS,
            "NS_URI_MZML": NS_URI_MZML,
            "NS_URI_XSI": NS_URI_XSI,
            "SCHEMA_LOCATION": SCHEMA_LOCATION,
            "CV_URI_MS": CV_URI_MS,
            "CV_URI_UO": CV_URI_UO,
            "CV_URI_PSI_MS": CV_URI_PSI_MS,
            "CV_NAME_MS": CV_NAME_MS,
            "CV_NAME_UO": CV_NAME_UO,
            "CV_NAME_PSI_MS": CV_NAME_PSI_MS,
        }

        # Add accession and name constants for each CV term
        for var_name, term_info in cv_terms.items():
            constants[f"ACCESSION_{var_name}"] = term_info["id"]
            constants[f"NAME_{var_name}"] = term_info["label"]

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
