import base64
import os
import struct
import xml.etree.ElementTree as ET
import zlib
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from alpharaw.mzml_io import cv_constants as CV


class MzMLWriter:
    """
    Class for converting MSData_Base objects to mzML format.

    This class provides functionality to write mass spectrometry data in the
    standard mzML format, compliant with the PSI-MS controlled vocabulary.

    Attributes
    ----------
    ms_data : MSData_Base
        The mass spectrometry data object to convert
    output_path : str
        Path where the mzML file will be saved
    binary_precision : int
        Binary encoding precision (32 or 64 bit)
    compression : Optional[str]
        Compression method (None or 'zlib')
    ns_uri : str
        Namespace URI for mzML format
    ns_prefix : str
        Namespace prefix for XML elements
    root : Optional[ET.Element]
        Root element of the XML tree (set after write() is called)
    """

    def __init__(
        self,
        ms_data,
        output_path: str,
        binary_precision: int = 64,
        compression: Optional[str] = None,
    ) -> None:
        """
        Initialize the writer with an MSData_Base object.

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
            Binary encoding precision (32 or 64 bit), by default 64
        compression : Optional[str], optional
            Compression method (None, 'zlib'), by default None

        Raises
        ------
        ValueError
            If binary_precision is not 32 or 64
            If compression is not None or 'zlib'
        """
        if binary_precision not in [32, 64]:
            raise ValueError("binary_precision must be 32 or 64")
        if compression not in [None, "zlib"]:
            raise ValueError("compression must be None or 'zlib'")

        self.ms_data = ms_data
        self.output_path = output_path
        self.binary_precision = binary_precision
        self.compression = compression
        self.ns_uri = CV.NS_URI_MZML
        self.ns_prefix = "{" + self.ns_uri + "}"
        self.root: Optional[ET.Element] = None

    def write(self) -> None:
        """
        Main method to write the mzML file.

        Creates the complete mzML XML structure and writes it to the specified
        output path with proper formatting and encoding.

        Raises
        ------
        IOError
            If the output file cannot be written
        """
        # Register namespace for proper output
        ET.register_namespace("", self.ns_uri)

        # Create the mzML content tree
        self.root = self._create_mzml_content()

        # Apply indentation
        ET.indent(self.root)

        tree = ET.ElementTree(self.root)
        tree.write(self.output_path, encoding="utf-8", xml_declaration=True)

    def _create_mzml_content(self) -> ET.Element:
        """
        Create the mzML content tree.

        Returns
        -------
        ET.Element
            Root element of the mzML XML tree with all required subelements
        """
        # Create root element for mzML
        root = ET.Element(self.ns_prefix + "mzML")
        root.set("version", "1.1.0")
        root.set("xmlns:xsi", CV.NS_URI_XSI)
        root.set("xsi:schemaLocation", CV.SCHEMA_LOCATION)

        # Add required elements
        self._add_cv_list(root)
        self._add_file_description(root)
        self._add_software_list(root)
        self._add_instrument_configuration_list(root)
        self._add_data_processing_list(root)
        self._add_run(root)

        return root

    def _add_cv_list(self, parent_elem: ET.Element) -> None:
        """
        Add controlled vocabulary list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the CV list to
        """
        cv_list = ET.SubElement(parent_elem, self.ns_prefix + "cvList")
        cv_list.set("count", "3")

        # MS CV
        cv = ET.SubElement(cv_list, self.ns_prefix + "cv")
        cv.set("id", CV.CV_MS)
        cv.set("fullName", CV.CV_NAME_MS)
        cv.set("URI", CV.CV_URI_MS)

        # UO CV
        cv = ET.SubElement(cv_list, self.ns_prefix + "cv")
        cv.set("id", CV.CV_UO)
        cv.set("fullName", CV.CV_NAME_UO)
        cv.set("URI", CV.CV_URI_UO)

        # PSI-MS CV
        cv = ET.SubElement(cv_list, self.ns_prefix + "cv")
        cv.set("id", CV.CV_PSI_MS)
        cv.set("fullName", CV.CV_NAME_PSI_MS)
        cv.set("URI", CV.CV_URI_PSI_MS)

    def _add_file_description(self, parent_elem: ET.Element) -> None:
        """
        Add file description section to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the file description to
        """
        file_description = ET.SubElement(
            parent_elem, self.ns_prefix + "fileDescription"
        )

        # File content
        file_content = ET.SubElement(file_description, self.ns_prefix + "fileContent")

        # Add CV params for file content
        self._add_cv_param(
            file_content, CV.CV_MS, CV.ACCESSION_MS1_SPECTRUM, CV.NAME_MS1_SPECTRUM, ""
        )

        if (self.ms_data.spectrum_df["ms_level"] == 2).any():
            self._add_cv_param(
                file_content,
                CV.CV_MS,
                CV.ACCESSION_MSN_SPECTRUM,
                CV.NAME_MSN_SPECTRUM,
                "",
            )

        # Source file list
        if hasattr(self.ms_data, "raw_file_path") and self.ms_data.raw_file_path:
            source_file_list = ET.SubElement(
                file_description, self.ns_prefix + "sourceFileList"
            )
            source_file_list.set("count", "1")

            source_file = ET.SubElement(source_file_list, self.ns_prefix + "sourceFile")
            source_file.set("id", "RAW1")
            source_file.set("name", os.path.basename(self.ms_data.raw_file_path))
            source_file.set("location", os.path.dirname(self.ms_data.raw_file_path))

            # Add CV params for source file
            if (
                hasattr(self.ms_data, "file_type")
                and self.ms_data.file_type == "thermo"
            ):
                self._add_cv_param(
                    source_file,
                    CV.CV_MS,
                    CV.ACCESSION_THERMO_RAW,
                    CV.NAME_THERMO_RAW,
                    "",
                )

    def _add_software_list(self, parent_elem: ET.Element) -> None:
        """
        Add software list section to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the software list to
        """
        software_list = ET.SubElement(parent_elem, self.ns_prefix + "softwareList")
        software_list.set("count", "1")

        software = ET.SubElement(software_list, self.ns_prefix + "software")
        software.set("id", "alpharaw")
        software.set("version", "0.4.7.dev0")

        self._add_cv_param(
            software,
            CV.CV_MS,
            CV.ACCESSION_ANALYSIS_SOFTWARE,
            CV.NAME_ANALYSIS_SOFTWARE,
            "alpharaw",
        )

    def _add_instrument_configuration_list(self, parent_elem: ET.Element) -> None:
        """
        Add instrument configuration list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the instrument configuration to
        """
        instrument_list = ET.SubElement(
            parent_elem, self.ns_prefix + "instrumentConfigurationList"
        )
        instrument_list.set("count", "1")

        instrument = ET.SubElement(
            instrument_list, self.ns_prefix + "instrumentConfiguration"
        )
        instrument.set("id", "IC1")

        self._add_cv_param(
            instrument,
            CV.CV_MS,
            CV.ACCESSION_INSTRUMENT_MODEL,
            CV.NAME_INSTRUMENT_MODEL,
            "",
        )

        # Add component list
        component_list = ET.SubElement(instrument, self.ns_prefix + "componentList")
        component_list.set("count", "3")

        # Source
        source = ET.SubElement(component_list, self.ns_prefix + "source")
        source.set("order", "1")
        self._add_cv_param(source, CV.CV_MS, CV.ACCESSION_ESI, CV.NAME_ESI, "")

        # Analyzer
        analyzer = ET.SubElement(component_list, self.ns_prefix + "analyzer")
        analyzer.set("order", "2")

        if (
            hasattr(self.ms_data, "auxiliary_items")
            and "analyzer" in self.ms_data.auxiliary_items
            and "analyzer" in self.ms_data.spectrum_df.columns
        ):
            analyzer_type = (
                self.ms_data.spectrum_df["analyzer"].iloc[0]
                if not self.ms_data.spectrum_df["analyzer"].empty
                else "orbitrap"
            )

            if analyzer_type.lower() in CV.ANALYZER_TYPES:
                accession, name = CV.ANALYZER_TYPES[analyzer_type.lower()]
                self._add_cv_param(analyzer, CV.CV_MS, accession, name, "")
            else:
                self._add_cv_param(
                    analyzer, CV.CV_MS, CV.ACCESSION_ORBITRAP, CV.NAME_ORBITRAP, ""
                )
        else:
            self._add_cv_param(
                analyzer, CV.CV_MS, CV.ACCESSION_ORBITRAP, CV.NAME_ORBITRAP, ""
            )

        # Detector
        detector = ET.SubElement(component_list, self.ns_prefix + "detector")
        detector.set("order", "3")
        self._add_cv_param(
            detector,
            CV.CV_MS,
            CV.ACCESSION_ELECTRON_MULTIPLIER,
            CV.NAME_ELECTRON_MULTIPLIER,
            "",
        )

    def _add_data_processing_list(self, parent_elem: ET.Element) -> None:
        """
        Add data processing list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the data processing list to
        """
        data_processing_list = ET.SubElement(
            parent_elem, self.ns_prefix + "dataProcessingList"
        )
        data_processing_list.set("count", "1")

        data_processing = ET.SubElement(
            data_processing_list, self.ns_prefix + "dataProcessing"
        )
        data_processing.set("id", "alpharaw_processing")

        processing_method = ET.SubElement(
            data_processing, self.ns_prefix + "processingMethod"
        )
        processing_method.set("order", "1")
        processing_method.set("softwareRef", "alpharaw")

        self._add_cv_param(
            processing_method,
            CV.CV_MS,
            CV.ACCESSION_FILE_FORMAT_CONVERSION,
            CV.NAME_FILE_FORMAT_CONVERSION,
            "",
        )

    def _add_run(self, parent_elem: ET.Element) -> None:
        """
        Add run section with spectrum list to the mzML document.

        Parameters
        ----------
        parent_elem : ET.Element
            Parent XML element to add the run section to
        """
        run = ET.SubElement(parent_elem, self.ns_prefix + "run")
        run.set("id", "run1")
        run.set("defaultInstrumentConfigurationRef", "IC1")

        # Add spectrum list
        spectrum_list = ET.SubElement(run, self.ns_prefix + "spectrumList")
        spectrum_count = len(self.ms_data.spectrum_df)
        spectrum_list.set("count", str(spectrum_count))
        spectrum_list.set("defaultDataProcessingRef", "alpharaw_processing")

        print(f"Writing {spectrum_count} spectra...")

        # Add spectra to spectrum_list
        for i in tqdm(range(spectrum_count), desc="Processing spectra"):
            try:
                row = self.ms_data.spectrum_df.iloc[i]
                self._add_spectrum(spectrum_list, i, row)
            except Exception as e:
                print(f"Error adding spectrum {i}: {e}")
                # Add empty spectrum as fallback
                spectrum = ET.SubElement(spectrum_list, self.ns_prefix + "spectrum")
                spectrum.set("index", str(i))
                spectrum.set("id", f"scan={i}")
                spectrum.set("defaultArrayLength", "0")

    def _add_spectrum(self, spectrum_list: ET.Element, i: int, row: pd.Series) -> None:
        """
        Add a single spectrum to the spectrum list.

        Parameters
        ----------
        spectrum_list : ET.Element
            Parent spectrum list element
        i : int
            Spectrum index
        row : pd.Series
            Spectrum metadata from the spectrum dataframe
        """
        spectrum = ET.SubElement(spectrum_list, self.ns_prefix + "spectrum")

        # Required attributes
        spectrum.set("index", str(i))
        spectrum.set("id", f"scan={i}")

        # Get peak data
        try:
            mz_array, intensity_array = self.ms_data.get_peaks(i)
        except Exception:
            # If there's an error, create empty arrays
            mz_array = np.array([])
            intensity_array = np.array([])

        spectrum.set("defaultArrayLength", str(len(mz_array)))

        # MS level - ensure it's an integer
        ms_level = int(row.get("ms_level", 1))
        self._add_cv_param(
            spectrum, CV.CV_MS, CV.ACCESSION_MS_LEVEL, CV.NAME_MS_LEVEL, ms_level
        )

        # Add centroid/profile indication
        if self.ms_data.centroided:
            self._add_cv_param(
                spectrum, CV.CV_MS, CV.ACCESSION_CENTROIDED, CV.NAME_CENTROIDED, ""
            )
        else:
            self._add_cv_param(
                spectrum, CV.CV_MS, CV.ACCESSION_PROFILE, CV.NAME_PROFILE, ""
            )

        # Add scan list
        scan_list = ET.SubElement(spectrum, self.ns_prefix + "scanList")
        scan_list.set("count", "1")

        self._add_cv_param(
            scan_list, CV.CV_MS, CV.ACCESSION_NO_COMBINATION, CV.NAME_NO_COMBINATION, ""
        )

        scan = ET.SubElement(scan_list, self.ns_prefix + "scan")

        # Add retention time
        rt_seconds = row["rt"] * 60  # Convert to seconds
        self._add_cv_param(
            scan,
            CV.CV_MS,
            CV.ACCESSION_SCAN_START_TIME,
            CV.NAME_SCAN_START_TIME,
            str(rt_seconds),
            CV.CV_UO,
            CV.ACCESSION_SECOND,
            CV.NAME_SECOND,
        )

        # Add precursor information for MS2+ spectra
        if ms_level > 1:
            self._add_precursor_info(spectrum, row)

        # Add binary data arrays
        binary_list = ET.SubElement(spectrum, self.ns_prefix + "binaryDataArrayList")
        binary_list.set("count", "2")  # m/z and intensity

        # m/z array
        self._add_binary_data_array(
            binary_list,
            mz_array,
            CV.ACCESSION_MZ_ARRAY,
            CV.NAME_MZ_ARRAY,
            CV.ACCESSION_MZ_UNIT,
            CV.NAME_MZ_UNIT,
        )

        # Intensity array
        self._add_binary_data_array(
            binary_list,
            intensity_array,
            CV.ACCESSION_INTENSITY_ARRAY,
            CV.NAME_INTENSITY_ARRAY,
            CV.ACCESSION_DETECTOR_COUNTS,
            CV.NAME_DETECTOR_COUNTS,
        )

    def _add_precursor_info(self, spectrum: ET.Element, row: pd.Series) -> None:
        """
        Add precursor information for MS2+ spectra.

        Parameters
        ----------
        spectrum : ET.Element
            Spectrum element to add precursor information to
        row : pd.Series
            Spectrum metadata containing precursor information
        """
        precursor_list = ET.SubElement(spectrum, self.ns_prefix + "precursorList")
        precursor_list.set("count", "1")

        precursor = ET.SubElement(precursor_list, self.ns_prefix + "precursor")

        # Isolation window
        isolation_window = ET.SubElement(precursor, self.ns_prefix + "isolationWindow")

        precursor_mz = row.get("precursor_mz", 0)
        isolation_lower = row.get("isolation_lower_mz", precursor_mz - 1.5)
        isolation_upper = row.get("isolation_upper_mz", precursor_mz + 1.5)

        self._add_cv_param(
            isolation_window,
            CV.CV_MS,
            CV.ACCESSION_ISOLATION_TARGET_MZ,
            CV.NAME_ISOLATION_TARGET_MZ,
            str(precursor_mz),
            CV.CV_MS,
            CV.ACCESSION_MZ_UNIT,
            CV.NAME_MZ_UNIT,
        )
        self._add_cv_param(
            isolation_window,
            CV.CV_MS,
            CV.ACCESSION_ISOLATION_LOWER_OFFSET,
            CV.NAME_ISOLATION_LOWER_OFFSET,
            str(precursor_mz - isolation_lower),
            CV.CV_MS,
            CV.ACCESSION_MZ_UNIT,
            CV.NAME_MZ_UNIT,
        )
        self._add_cv_param(
            isolation_window,
            CV.CV_MS,
            CV.ACCESSION_ISOLATION_UPPER_OFFSET,
            CV.NAME_ISOLATION_UPPER_OFFSET,
            str(isolation_upper - precursor_mz),
            CV.CV_MS,
            CV.ACCESSION_MZ_UNIT,
            CV.NAME_MZ_UNIT,
        )

        # Selected ion list
        selected_ion_list = ET.SubElement(precursor, self.ns_prefix + "selectedIonList")
        selected_ion_list.set("count", "1")

        selected_ion = ET.SubElement(selected_ion_list, self.ns_prefix + "selectedIon")
        self._add_cv_param(
            selected_ion,
            CV.CV_MS,
            CV.ACCESSION_SELECTED_ION_MZ,
            CV.NAME_SELECTED_ION_MZ,
            str(precursor_mz),
            CV.CV_MS,
            CV.ACCESSION_MZ_UNIT,
            CV.NAME_MZ_UNIT,
        )

        # Ensure charge state is an integer
        if "precursor_charge" in row and row["precursor_charge"] > 0:
            charge = int(row["precursor_charge"])
            self._add_cv_param(
                selected_ion,
                CV.CV_MS,
                CV.ACCESSION_CHARGE_STATE,
                CV.NAME_CHARGE_STATE,
                charge,
            )

        # Activation
        activation = ET.SubElement(precursor, self.ns_prefix + "activation")

        # Default to HCD if not specified
        activation_method = row.get("activation", "HCD")

        if activation_method in CV.ACTIVATION_METHODS:
            accession, name = CV.ACTIVATION_METHODS[activation_method]
            self._add_cv_param(activation, CV.CV_MS, accession, name, "")
        else:
            # Fallback for unknown activation methods
            self._add_cv_param(
                activation,
                CV.CV_MS,
                CV.ACCESSION_DISSOCIATION_METHOD,
                CV.NAME_DISSOCIATION_METHOD,
                activation_method,
            )

        # Add collision energy if available
        if "nce" in row and row["nce"] > 0:
            self._add_cv_param(
                activation,
                CV.CV_MS,
                CV.ACCESSION_COLLISION_ENERGY,
                CV.NAME_COLLISION_ENERGY,
                str(row["nce"]),
                CV.CV_UO,
                CV.ACCESSION_ELECTRONVOLT,
                CV.NAME_ELECTRONVOLT,
            )

    def _add_binary_data_array(
        self,
        parent: ET.Element,
        data: np.ndarray,
        array_type_acc: str,
        array_type_name: str,
        unit_acc: Optional[str] = None,
        unit_name: Optional[str] = None,
    ) -> None:
        """
        Add binary data array element for m/z or intensity data.

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
        unit_acc : Optional[str], optional
            CV accession for the unit, by default None
        unit_name : Optional[str], optional
            CV name for the unit, by default None
        """
        binary_array = ET.SubElement(parent, self.ns_prefix + "binaryDataArray")

        # Encode data
        encoded_data = ""
        if len(data) > 0:
            # Format string: '<' for little-endian, 'f' for 32-bit float, 'd' for 64-bit float
            format_char = "f" if self.binary_precision == 32 else "d"
            buffer = struct.pack(f"<{len(data)}{format_char}", *data)

            # Apply compression if requested
            if self.compression == "zlib":
                buffer = zlib.compress(buffer)

            encoded_data = base64.b64encode(buffer).decode("ascii")

        # Set required attributes
        binary_array.set("encodedLength", str(len(encoded_data)))
        binary_array.set("arrayLength", str(len(data)))

        # Data type and compression
        if self.binary_precision == 32:
            self._add_cv_param(
                binary_array,
                CV.CV_MS,
                CV.ACCESSION_32BIT_FLOAT,
                CV.NAME_32BIT_FLOAT,
                "",
            )
        else:
            self._add_cv_param(
                binary_array,
                CV.CV_MS,
                CV.ACCESSION_64BIT_FLOAT,
                CV.NAME_64BIT_FLOAT,
                "",
            )

        # Compression
        if self.compression == "zlib":
            self._add_cv_param(
                binary_array,
                CV.CV_MS,
                CV.ACCESSION_ZLIB_COMPRESSION,
                CV.NAME_ZLIB_COMPRESSION,
                "",
            )
        else:
            self._add_cv_param(
                binary_array,
                CV.CV_MS,
                CV.ACCESSION_NO_COMPRESSION,
                CV.NAME_NO_COMPRESSION,
                "",
            )

        self._add_cv_param(binary_array, CV.CV_MS, array_type_acc, array_type_name, "")

        # Add binary element
        binary = ET.SubElement(binary_array, self.ns_prefix + "binary")
        binary.text = encoded_data

    def _add_cv_param(
        self,
        parent: ET.Element,
        cv_ref: str,
        accession: str,
        name: str,
        value: Union[str, int, float] = "",
        unit_cv_ref: Optional[str] = None,
        unit_accession: Optional[str] = None,
        unit_name: Optional[str] = None,
    ) -> None:
        """
        Helper method to add a CV parameter to an XML element.

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
            Parameter value, by default ""
        unit_cv_ref : Optional[str], optional
            Unit CV reference, by default None
        unit_accession : Optional[str], optional
            Unit CV accession, by default None
        unit_name : Optional[str], optional
            Unit CV name, by default None
        """
        cv_param = ET.SubElement(parent, self.ns_prefix + "cvParam")
        cv_param.set(CV.CV_REF, cv_ref)
        cv_param.set(CV.ACCESSION, accession)
        cv_param.set(CV.NAME, name)

        # Format value appropriately based on the parameter type
        if value != "" and value is not None:
            # Convert to string and handle special cases
            str_value = str(value)

            # If this is an ms level or charge state, ensure it's an integer
            if accession in [CV.ACCESSION_MS_LEVEL, CV.ACCESSION_CHARGE_STATE]:
                try:
                    int_value = int(float(str_value))
                    str_value = str(int_value)
                except (ValueError, TypeError):
                    str_value = str_value

            cv_param.set(CV.VALUE, str_value)
        else:
            cv_param.set(CV.VALUE, "")

        if unit_cv_ref:
            cv_param.set("unitCvRef", unit_cv_ref)
            cv_param.set("unitAccession", unit_accession)
            cv_param.set("unitName", unit_name)
