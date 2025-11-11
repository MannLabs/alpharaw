import base64
import os
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from alpharaw.mzml_io.cv_constants import (
    CV,
    XML,
    CVTerms,
    create_cv_constants_from_owl,
    get_accession_key,
    get_name_key,
)


class MzMLWriter:
    """Class for converting MSData_Base objects to mzML format.

    This class provides functionality to write mass spectrometry data in the
    standard mzML format, compliant with the PSI-MS controlled vocabulary.

    Attributes
    ----------
    root : Optional[ET.Element]
        Root element of the XML tree (set after write() is called)

    """

    # XML attribute constants
    ATTR_ID = "id"
    ATTR_FULL_NAME = "fullName"
    ATTR_URI = "URI"

    def __init__(
        self,
        ms_data,
        output_path: str,
        binary_precision: int = 32,  # Changed from 64 to 32
        compression: Optional[str] = None,
    ) -> None:
        """Initialize the writer with an MSData_Base object.

        Parameters
        ----------
        ms_data : MSData_Base
            The mass spectrometry data object to convert. Must have attributes:
            - spectrum_df: DataFrame with spectrum metadata
            - centroided: bool indicating if data is centroided
            - get_peaks(): method to retrieve m/z and intensity arrays
        output_path : str
            Path where the mzML file will be saved
        binary_precision : int, optional
            Binary encoding precision (32 or 64 bit), by default 32.
            32-bit precision matches the internal float32 data precision.
        compression : Optional[str], optional
            Compression method (None, 'zlib'), by default None

        """
        if binary_precision not in [32, 64]:
            raise ValueError("binary_precision must be 32 or 64")
        if compression not in [None, "zlib"]:
            raise ValueError("compression must be None or 'zlib'")

        self._ms_data = ms_data
        self._output_path = output_path
        self._binary_precision = binary_precision
        self._compression = compression

        # Initialize CV constants
        self._initialize_cv_constants()

        self._ns_uri = self._cv[XML.NS_URI_MZML]
        self._ns_prefix = "{" + self._ns_uri + "}"
        self.root: Optional[ET.Element] = None

    @property
    def ns_uri(self) -> str:
        """Get the namespace URI for XML parsing."""
        return self._ns_uri

    def _initialize_cv_constants(self) -> None:
        """Initialize CV constants from OWL file."""
        # Get path to OWL file
        current_dir = Path(__file__).parent
        owl_file_path = current_dir.parent / "resources" / "psi-ms.owl"

        # Load CV constants from OWL file
        self._cv = create_cv_constants_from_owl(owl_file_path)

    def write(self) -> None:
        """Main method to write the mzML file.

        Creates the complete mzML XML structure and writes it to the specified
        output path with proper formatting and encoding.

        Raises
        ------
        IOError
            If the output file cannot be written

        """
        # Register namespace for proper output
        ET.register_namespace("", self._ns_uri)

        # Create the mzML content tree
        self.root = self._create_mzml_content()

        # Apply indentation
        ET.indent(self.root)

        tree = ET.ElementTree(self.root)
        tree.write(self._output_path, encoding="utf-8", xml_declaration=True)

    def _create_mzml_content(self) -> ET.Element:
        """Create the mzML content tree.

        Returns
        -------
        ET.Element
            Root element of the mzML XML tree with all required subelements

        """
        # Create root element for mzML
        root = ET.Element(self._ns_prefix + "mzML")
        root.set("version", "1.1.0")
        root.set("xmlns:xsi", self._cv[XML.NS_URI_XSI])
        root.set("xsi:schemaLocation", self._cv[XML.SCHEMA_LOCATION])

        # Add required elements
        self._add_cv_list(root)
        self._add_file_description(root)
        self._add_software_list(root)
        self._add_instrument_configuration_list(root)
        self._add_data_processing_list(root)
        self._add_run(root)

        return root

    def _add_cv_list(self, parent_elem: ET.Element) -> None:
        """Add controlled vocabulary list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the CV list to

        """
        cv_list = ET.SubElement(parent_elem, self._ns_prefix + "cvList")
        cv_list.set("count", "3")

        # MS CV
        cv = ET.SubElement(cv_list, self._ns_prefix + "cv")
        cv.set(self.ATTR_ID, self._cv[CV.MS])
        cv.set(self.ATTR_FULL_NAME, self._cv[CV.NAME_MS])
        cv.set(self.ATTR_URI, self._cv[XML.URI_MS])

        # UO CV
        cv = ET.SubElement(cv_list, self._ns_prefix + "cv")
        cv.set(self.ATTR_ID, self._cv[CV.UO])
        cv.set(self.ATTR_FULL_NAME, self._cv[CV.NAME_UO])
        cv.set(self.ATTR_URI, self._cv[XML.URI_UO])

        # PSI-MS CV
        cv = ET.SubElement(cv_list, self._ns_prefix + "cv")
        cv.set(self.ATTR_ID, self._cv[CV.PSI_MS])
        cv.set(self.ATTR_FULL_NAME, self._cv[CV.NAME_PSI_MS])
        cv.set(self.ATTR_URI, self._cv[XML.URI_PSI_MS])

    def _add_file_description(self, parent_elem: ET.Element) -> None:
        """Add file description section to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the file description to

        """
        file_description = ET.SubElement(
            parent_elem, self._ns_prefix + "fileDescription"
        )

        # File content
        file_content = ET.SubElement(file_description, self._ns_prefix + "fileContent")

        # Add CV params for file content
        self._add_cv_param(
            file_content,
            self._cv[CV.MS],
            self._cv[get_accession_key(CVTerms.MS1_SPECTRUM)],
            self._cv[get_name_key(CVTerms.MS1_SPECTRUM)],
        )

        if (self._ms_data.spectrum_df["ms_level"] == 2).any():
            self._add_cv_param(
                file_content,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.MSN_SPECTRUM)],
                self._cv[get_name_key(CVTerms.MSN_SPECTRUM)],
            )

        # Source file list
        if hasattr(self._ms_data, "raw_file_path") and self._ms_data.raw_file_path:
            source_file_list = ET.SubElement(
                file_description, self._ns_prefix + "sourceFileList"
            )
            source_file_list.set("count", "1")

            source_file = ET.SubElement(
                source_file_list, self._ns_prefix + "sourceFile"
            )
            source_file.set(self.ATTR_ID, "RAW1")
            source_file.set("name", os.path.basename(self._ms_data.raw_file_path))
            source_file.set("location", os.path.dirname(self._ms_data.raw_file_path))

            # Add CV params for source file if available
            if (
                hasattr(self._ms_data, "file_type")
                and self._ms_data.file_type == "thermo"
                and get_accession_key(CVTerms.THERMO_RAW) in self._cv
            ):
                self._add_cv_param(
                    source_file,
                    self._cv[CV.MS],
                    self._cv[get_accession_key(CVTerms.THERMO_RAW)],
                    self._cv[get_name_key(CVTerms.THERMO_RAW)],
                )

    def _add_software_list(self, parent_elem: ET.Element) -> None:
        """Add software list section to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the software list to

        """
        software_list = ET.SubElement(parent_elem, self._ns_prefix + "softwareList")
        software_list.set("count", "1")

        software = ET.SubElement(software_list, self._ns_prefix + "software")
        software.set(self.ATTR_ID, "alpharaw")
        software.set("version", "0.4.7.dev0")

        # Use analysis software CV term if available
        if get_accession_key(CVTerms.ANALYSIS_SOFTWARE) in self._cv:
            self._add_cv_param(
                software,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ANALYSIS_SOFTWARE)],
                self._cv[get_name_key(CVTerms.ANALYSIS_SOFTWARE)],
                "alpharaw",
            )

    def _add_instrument_configuration_list(self, parent_elem: ET.Element) -> None:
        """Add instrument configuration list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the instrument configuration to

        """
        instrument_list = ET.SubElement(
            parent_elem, self._ns_prefix + "instrumentConfigurationList"
        )
        instrument_list.set("count", "1")

        instrument = ET.SubElement(
            instrument_list, self._ns_prefix + "instrumentConfiguration"
        )
        instrument.set(self.ATTR_ID, "IC1")

        # Use instrument model CV term if available
        if get_accession_key(CVTerms.INSTRUMENT_MODEL) in self._cv:
            self._add_cv_param(
                instrument,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.INSTRUMENT_MODEL)],
                self._cv[get_name_key(CVTerms.INSTRUMENT_MODEL)],
            )

        # Add component list
        component_list = ET.SubElement(instrument, self._ns_prefix + "componentList")
        component_list.set("count", "3")

        # Source
        source = ET.SubElement(component_list, self._ns_prefix + "source")
        source.set("order", "1")
        if get_accession_key(CVTerms.ESI) in self._cv:
            self._add_cv_param(
                source,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ESI)],
                self._cv[get_name_key(CVTerms.ESI)],
            )

        # Analyzer
        analyzer = ET.SubElement(component_list, self._ns_prefix + "analyzer")
        analyzer.set("order", "2")

        if (
            hasattr(self._ms_data, "auxiliary_items")
            and "analyzer" in self._ms_data.auxiliary_items
            and "analyzer" in self._ms_data.spectrum_df.columns
        ):
            analyzer_type = (
                self._ms_data.spectrum_df["analyzer"].iloc[0]
                if not self._ms_data.spectrum_df["analyzer"].empty
                else "orbitrap"
            )

            if analyzer_type.lower() in self._cv.get("ANALYZER_TYPES", {}):
                accession, name = self._cv["ANALYZER_TYPES"][analyzer_type.lower()]
                self._add_cv_param(analyzer, self._cv[CV.MS], accession, name)
            elif get_accession_key(CVTerms.ORBITRAP) in self._cv:
                self._add_cv_param(
                    analyzer,
                    self._cv[CV.MS],
                    self._cv[get_accession_key(CVTerms.ORBITRAP)],
                    self._cv[get_name_key(CVTerms.ORBITRAP)],
                )
        elif get_accession_key(CVTerms.ORBITRAP) in self._cv:
            self._add_cv_param(
                analyzer,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ORBITRAP)],
                self._cv[get_name_key(CVTerms.ORBITRAP)],
            )

        # Detector
        detector = ET.SubElement(component_list, self._ns_prefix + "detector")
        detector.set("order", "3")
        if get_accession_key(CVTerms.ELECTRON_MULTIPLIER) in self._cv:
            self._add_cv_param(
                detector,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ELECTRON_MULTIPLIER)],
                self._cv[get_name_key(CVTerms.ELECTRON_MULTIPLIER)],
            )

    def _add_data_processing_list(self, parent_elem: ET.Element) -> None:
        """Add data processing list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the data processing list to

        """
        data_processing_list = ET.SubElement(
            parent_elem, self._ns_prefix + "dataProcessingList"
        )
        data_processing_list.set("count", "1")

        data_processing = ET.SubElement(
            data_processing_list, self._ns_prefix + "dataProcessing"
        )
        data_processing.set(self.ATTR_ID, "alpharaw_processing")

        processing_method = ET.SubElement(
            data_processing, self._ns_prefix + "processingMethod"
        )
        processing_method.set("order", "1")
        processing_method.set("softwareRef", "alpharaw")

        if get_accession_key(CVTerms.FILE_FORMAT_CONVERSION) in self._cv:
            self._add_cv_param(
                processing_method,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.FILE_FORMAT_CONVERSION)],
                self._cv[get_name_key(CVTerms.FILE_FORMAT_CONVERSION)],
            )

    def _add_run(self, parent_elem: ET.Element) -> None:
        """Add run section with spectrum list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the run section to

        """
        run = ET.SubElement(parent_elem, self._ns_prefix + "run")
        run.set(self.ATTR_ID, "run1")
        run.set("defaultInstrumentConfigurationRef", "IC1")

        # Add spectrum list
        spectrum_list = ET.SubElement(run, self._ns_prefix + "spectrumList")
        spectrum_count = len(self._ms_data.spectrum_df)
        spectrum_list.set("count", str(spectrum_count))
        spectrum_list.set("defaultDataProcessingRef", "alpharaw_processing")

        print(f"Writing {spectrum_count} spectra...")

        # Add spectra to spectrum_list
        for i in tqdm(range(spectrum_count), desc="Processing spectra"):
            row = self._ms_data.spectrum_df.iloc[i]
            self._add_spectrum(spectrum_list, i, row)

    def _add_spectrum(self, spectrum_list: ET.Element, i: int, row: pd.Series) -> None:
        """Add a single spectrum to the spectrum list.

        Parameters
        ----------
        spectrum_list : ET.Element
            Parent spectrum list element
        i : int
            Spectrum index
        row : pd.Series
            Spectrum metadata from the spectrum dataframe

        """
        spectrum = ET.SubElement(spectrum_list, self._ns_prefix + "spectrum")

        # Required attributes
        spectrum.set("index", str(i))
        spectrum.set(self.ATTR_ID, f"scan={i}")

        # Get peak data
        try:
            mz_array, intensity_array = self._ms_data.get_peaks(i)
        except Exception:
            # If there's an error, create empty arrays
            mz_array = np.array([])
            intensity_array = np.array([])

        spectrum.set("defaultArrayLength", str(len(mz_array)))

        # MS level - ensure it's an integer
        ms_level = int(row.get("ms_level", 1))
        self._add_cv_param(
            spectrum,
            self._cv[CV.MS],
            self._cv[get_accession_key(CVTerms.MS_LEVEL)],
            self._cv[get_name_key(CVTerms.MS_LEVEL)],
            ms_level,
        )

        # Add centroid/profile indication
        if self._ms_data.centroided:
            self._add_cv_param(
                spectrum,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.CENTROIDED)],
                self._cv[get_name_key(CVTerms.CENTROIDED)],
            )
        else:
            self._add_cv_param(
                spectrum,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.PROFILE)],
                self._cv[get_name_key(CVTerms.PROFILE)],
            )

        # Add scan list
        scan_list = ET.SubElement(spectrum, self._ns_prefix + "scanList")
        scan_list.set("count", "1")

        if get_accession_key(CVTerms.NO_COMBINATION) in self._cv:
            self._add_cv_param(
                scan_list,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.NO_COMBINATION)],
                self._cv[get_name_key(CVTerms.NO_COMBINATION)],
            )

        scan = ET.SubElement(scan_list, self._ns_prefix + "scan")

        # Add retention time
        rt_seconds = row["rt"] * 60  # Convert to seconds
        if (
            get_accession_key(CVTerms.SCAN_START_TIME) in self._cv
            and get_accession_key(CVTerms.SECOND) in self._cv
        ):
            self._add_cv_param(
                scan,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.SCAN_START_TIME)],
                self._cv[get_name_key(CVTerms.SCAN_START_TIME)],
                str(rt_seconds),
                self._cv[CV.UO],
                self._cv[get_accession_key(CVTerms.SECOND)],
                self._cv[get_name_key(CVTerms.SECOND)],
            )

        # Add precursor information for MS2+ spectra
        if ms_level > 1:
            self._add_precursor_info(spectrum, row)

        # Add binary data arrays
        binary_list = ET.SubElement(spectrum, self._ns_prefix + "binaryDataArrayList")
        binary_list.set("count", "2")  # m/z and intensity

        # m/z array
        self._add_binary_data_array(
            binary_list,
            mz_array,
            self._cv[get_accession_key(CVTerms.MZ_ARRAY)],
            self._cv[get_name_key(CVTerms.MZ_ARRAY)],
        )

        # Intensity array
        self._add_binary_data_array(
            binary_list,
            intensity_array,
            self._cv[get_accession_key(CVTerms.INTENSITY_ARRAY)],
            self._cv[get_name_key(CVTerms.INTENSITY_ARRAY)],
        )

    def _add_precursor_info(self, spectrum: ET.Element, row: pd.Series) -> None:
        """Add precursor information for MS2+ spectra.

        Parameters
        ----------
        spectrum : ET.Element
            Spectrum element to add precursor information to
        row : pd.Series
            Spectrum metadata containing precursor information

        """
        precursor_list = ET.SubElement(spectrum, self._ns_prefix + "precursorList")
        precursor_list.set("count", "1")

        precursor = ET.SubElement(precursor_list, self._ns_prefix + "precursor")

        # Isolation window
        isolation_window = ET.SubElement(precursor, self._ns_prefix + "isolationWindow")

        precursor_mz = row.get("precursor_mz", 0)

        # Add isolation window parameters if CV terms are available
        if get_accession_key(CVTerms.ISOLATION_TARGET_MZ) in self._cv:
            self._add_cv_param(
                isolation_window,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ISOLATION_TARGET_MZ)],
                self._cv[get_name_key(CVTerms.ISOLATION_TARGET_MZ)],
                str(precursor_mz),
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.MZ_UNIT)],
                self._cv[get_name_key(CVTerms.MZ_UNIT)],
            )

        lower_offset = row.get("precursor_mz", 0) - row.get("isolation_lower_mz", 0)
        upper_offset = row.get("isolation_upper_mz", 0) - row.get("precursor_mz", 0)

        if get_accession_key(CVTerms.ISOLATION_LOWER_OFFSET) in self._cv and lower_offset > 0:
            self._add_cv_param(
                isolation_window,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ISOLATION_LOWER_OFFSET)],
                self._cv[get_name_key(CVTerms.ISOLATION_LOWER_OFFSET)],
                str(lower_offset),
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.MZ_UNIT)],
                self._cv[get_name_key(CVTerms.MZ_UNIT)],
            )

        if get_accession_key(CVTerms.ISOLATION_UPPER_OFFSET) in self._cv and upper_offset > 0:
            self._add_cv_param(
                isolation_window,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ISOLATION_UPPER_OFFSET)],
                self._cv[get_name_key(CVTerms.ISOLATION_UPPER_OFFSET)],
                str(upper_offset),
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.MZ_UNIT)],
                self._cv[get_name_key(CVTerms.MZ_UNIT)],
            )

        # Selected ion list
        selected_ion_list = ET.SubElement(
            precursor, self._ns_prefix + "selectedIonList"
        )
        selected_ion_list.set("count", "1")

        selected_ion = ET.SubElement(selected_ion_list, self._ns_prefix + "selectedIon")

        if get_accession_key(CVTerms.SELECTED_ION_MZ) in self._cv:
            self._add_cv_param(
                selected_ion,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.SELECTED_ION_MZ)],
                self._cv[get_name_key(CVTerms.SELECTED_ION_MZ)],
                str(precursor_mz),
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.MZ_UNIT)],
                self._cv[get_name_key(CVTerms.MZ_UNIT)],
            )

        # Ensure charge state is an integer
        if "precursor_charge" in row and row["precursor_charge"] > 0:
            charge = int(row["precursor_charge"])
            if get_accession_key(CVTerms.CHARGE_STATE) in self._cv:
                self._add_cv_param(
                    selected_ion,
                    self._cv[CV.MS],
                    self._cv[get_accession_key(CVTerms.CHARGE_STATE)],
                    self._cv[get_name_key(CVTerms.CHARGE_STATE)],
                    charge,
                )

        # Activation
        activation = ET.SubElement(precursor, self._ns_prefix + "activation")

        # Default to HCD if not specified
        activation_method = row.get("activation", "HCD")

        if activation_method in self._cv.get("ACTIVATION_METHODS", {}):
            accession, name = self._cv["ACTIVATION_METHODS"][activation_method]
            self._add_cv_param(activation, self._cv[CV.MS], accession, name)
        elif get_accession_key(CVTerms.DISSOCIATION_METHOD) in self._cv:
            # Fallback for unknown activation methods
            self._add_cv_param(
                activation,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.DISSOCIATION_METHOD)],
                self._cv[get_name_key(CVTerms.DISSOCIATION_METHOD)],
                activation_method,
            )

        # Add collision energy if available
        if (
            "nce" in row
            and row["nce"] > 0
            and get_accession_key(CVTerms.COLLISION_ENERGY) in self._cv
        ):
            self._add_cv_param(
                activation,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.COLLISION_ENERGY)],
                self._cv[get_name_key(CVTerms.COLLISION_ENERGY)],
                str(row["nce"]),
                self._cv[CV.UO],
                self._cv[get_accession_key(CVTerms.ELECTRONVOLT)],
                self._cv[get_name_key(CVTerms.ELECTRONVOLT)],
            )

    def _add_binary_data_array(
        self,
        parent: ET.Element,
        data: np.ndarray,
        array_type_acc: str,
        array_type_name: str,
    ) -> None:
        """Add binary data array element for m/z or intensity data.

        Parameters
        ----------
        parent : ET.Element
            Parent element to add the binary data array to
        data : np.ndarray
            Numeric data to encode
        array_type_acc : str
            CV accession for the array type (e.g., 'MS:1000514' for m/z array)
        array_type_name : str
            CV name for the array type (e.g., 'm/z array')

        """
        binary_array = ET.SubElement(parent, self._ns_prefix + "binaryDataArray")

        # Encode data
        encoded_data = ""
        if len(data) > 0:
            # Format string: '<' for little-endian, 'f' for 32-bit float, 'd' for 64-bit float
            format_char = "f" if self._binary_precision == 32 else "d"
            buffer = struct.pack(f"<{len(data)}{format_char}", *data)

            # Apply compression if requested
            if self._compression == "zlib":
                buffer = zlib.compress(buffer)

            encoded_data = base64.b64encode(buffer).decode("ascii")

        # Set required attributes
        binary_array.set("encodedLength", str(len(encoded_data)))
        binary_array.set("arrayLength", str(len(data)))

        # Data type and compression
        if self._binary_precision == 32:
            self._add_cv_param(
                binary_array,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.FLOAT_32BIT)],
                self._cv[get_name_key(CVTerms.FLOAT_32BIT)],
            )
        else:
            self._add_cv_param(
                binary_array,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.FLOAT_64BIT)],
                self._cv[get_name_key(CVTerms.FLOAT_64BIT)],
            )

        # Compression
        if self._compression == "zlib":
            self._add_cv_param(
                binary_array,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.ZLIB_COMPRESSION)],
                self._cv[get_name_key(CVTerms.ZLIB_COMPRESSION)],
            )
        else:
            self._add_cv_param(
                binary_array,
                self._cv[CV.MS],
                self._cv[get_accession_key(CVTerms.NO_COMPRESSION)],
                self._cv[get_name_key(CVTerms.NO_COMPRESSION)],
            )

        self._add_cv_param(
            binary_array, self._cv[CV.MS], array_type_acc, array_type_name
        )

        # Add binary element
        binary = ET.SubElement(binary_array, self._ns_prefix + "binary")
        binary.text = encoded_data

    def _add_cv_param(
        self,
        parent: ET.Element,
        cv_ref: str,
        accession: str,
        name: str,
        value: Union[str, float] = "",
        unit_cv_ref: Optional[str] = None,
        unit_accession: Optional[str] = None,
        unit_name: Optional[str] = None,
    ) -> None:
        """Helper method to add a CV (Controlled Vocabulary) parameter to an XML element.

        CV parameters use standardized terms from PSI-MS ontology to provide semantic
        meaning to mzML data elements. Per HUPO-PSI mzML specifications, all cvParam
        elements must have a value attribute. If the ontology term does not have an
        associated value, it must be set to an empty string.

        Parameters
        ----------
        parent : ET.Element
            Parent XML element to add the CV parameter to
        cv_ref : str
            CV reference (e.g., 'MS', 'UO')
        accession : str
            CV accession number (e.g., 'MS:1000511')
        name : str
            CV parameter name (e.g., 'ms level')
        value : Union[str, int, float], optional
            Parameter value, by default "". Empty string is used when the CV term
            has no associated value (e.g., for binary CV terms like 'centroid spectrum').
        unit_cv_ref : Optional[str], optional
            Unit CV reference, by default None
        unit_accession : Optional[str], optional
            Unit CV accession, by default None
        unit_name : Optional[str], optional
            Unit CV name, by default None

        """
        cv_param = ET.SubElement(parent, self._ns_prefix + "cvParam")
        cv_param.set(self._cv[CV.REF], cv_ref)
        cv_param.set(self._cv[CV.ACCESSION], accession)
        cv_param.set(self._cv[CV.NAME], name)

        # Format value appropriately based on the parameter type
        if value != "" and value is not None:
            # Convert to string and handle special cases
            str_value = str(value)

            # If this is an ms level or charge state, ensure it's an integer
            if accession in [
                self._cv.get(get_accession_key(CVTerms.MS_LEVEL)),
                self._cv.get(get_accession_key(CVTerms.CHARGE_STATE)),
            ]:
                try:
                    int_value = int(float(str_value))
                    str_value = str(int_value)
                except (ValueError, TypeError):
                    str_value = str_value

            cv_param.set(self._cv[CV.VALUE], str_value)
        else:
            cv_param.set(self._cv[CV.VALUE], "")

        if unit_cv_ref:
            cv_param.set("unitCvRef", unit_cv_ref)
            cv_param.set("unitAccession", unit_accession)
            cv_param.set("unitName", unit_name)
