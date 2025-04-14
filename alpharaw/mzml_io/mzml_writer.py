import xml.etree.ElementTree as ET
import base64
import struct
import os
import zlib
import hashlib
import multiprocessing as mp
import platform
from functools import partial
from tqdm import tqdm
import numpy as np

def _create_spectrum_xml(i, row, ms_data, binary_precision, compression, ns_prefix):
    """Create XML string for a single spectrum"""
    # Build spectrum using ElementTree
    spectrum = ET.Element(ns_prefix + "spectrum")
    
    # Required attributes
    spectrum.set("index", str(i))
    spectrum.set("id", f"scan={i}")
    
    # Get peak data using position i
    try:
        mz_array, intensity_array = ms_data.get_peaks(i)
    except Exception:
        # If there's an error, create empty arrays
        mz_array = np.array([])
        intensity_array = np.array([])
        
    spectrum.set("defaultArrayLength", str(len(mz_array)))
    
    # MS level
    ms_level = row.get('ms_level', 1)
    cv_param = ET.SubElement(spectrum, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000511")
    cv_param.set("name", "ms level")
    cv_param.set("value", str(ms_level))

    # Add centroid/profile indication
    cv_param = ET.SubElement(spectrum, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000127" if ms_data.centroided else "MS:1000128")
    cv_param.set("name", "centroid spectrum" if ms_data.centroided else "profile spectrum")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # Add scan list
    scan_list = ET.SubElement(spectrum, ns_prefix + "scanList")
    scan_list.set("count", "1")
    
    scan = ET.SubElement(scan_list, ns_prefix + "scan")
    
    # Add retention time
    rt_seconds = row['rt'] * 60  # Convert to seconds
    cv_param = ET.SubElement(scan, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000016")
    cv_param.set("name", "scan start time")
    cv_param.set("value", str(rt_seconds))
    cv_param.set("unitCvRef", "UO")
    cv_param.set("unitAccession", "UO:0000010")
    cv_param.set("unitName", "second")
    
    # Add precursor information for MS2+ spectra
    if ms_level > 1:
        precursor_list = ET.SubElement(spectrum, ns_prefix + "precursorList")
        precursor_list.set("count", "1")
        
        precursor = ET.SubElement(precursor_list, ns_prefix + "precursor")
        
        # Isolation window
        isolation_window = ET.SubElement(precursor, ns_prefix + "isolationWindow")
        
        precursor_mz = row.get('precursor_mz', 0)
        isolation_lower = row.get('isolation_lower_mz', precursor_mz - 1.5)
        isolation_upper = row.get('isolation_upper_mz', precursor_mz + 1.5)
        
        # Target m/z
        cv_param = ET.SubElement(isolation_window, ns_prefix + "cvParam")
        cv_param.set("cvRef", "MS")
        cv_param.set("accession", "MS:1000827")
        cv_param.set("name", "isolation window target m/z")
        cv_param.set("value", str(precursor_mz))
        cv_param.set("unitCvRef", "MS")
        cv_param.set("unitAccession", "MS:1000040")
        cv_param.set("unitName", "m/z")
        
        # Lower offset
        cv_param = ET.SubElement(isolation_window, ns_prefix + "cvParam")
        cv_param.set("cvRef", "MS")
        cv_param.set("accession", "MS:1000828")
        cv_param.set("name", "isolation window lower offset")
        cv_param.set("value", str(precursor_mz - isolation_lower))
        cv_param.set("unitCvRef", "MS")
        cv_param.set("unitAccession", "MS:1000040")
        cv_param.set("unitName", "m/z")
        
        # Upper offset
        cv_param = ET.SubElement(isolation_window, ns_prefix + "cvParam")
        cv_param.set("cvRef", "MS")
        cv_param.set("accession", "MS:1000829")
        cv_param.set("name", "isolation window upper offset")
        cv_param.set("value", str(isolation_upper - precursor_mz))
        cv_param.set("unitCvRef", "MS")
        cv_param.set("unitAccession", "MS:1000040")
        cv_param.set("unitName", "m/z")
        
        # Selected ion list
        selected_ion_list = ET.SubElement(precursor, ns_prefix + "selectedIonList")
        selected_ion_list.set("count", "1")
        
        selected_ion = ET.SubElement(selected_ion_list, ns_prefix + "selectedIon")
        
        cv_param = ET.SubElement(selected_ion, ns_prefix + "cvParam")
        cv_param.set("cvRef", "MS")
        cv_param.set("accession", "MS:1000744")
        cv_param.set("name", "selected ion m/z")
        cv_param.set("value", str(precursor_mz))
        cv_param.set("unitCvRef", "MS")
        cv_param.set("unitAccession", "MS:1000040")
        cv_param.set("unitName", "m/z")
        
        if 'precursor_charge' in row and row['precursor_charge'] > 0:
            cv_param = ET.SubElement(selected_ion, ns_prefix + "cvParam")
            cv_param.set("cvRef", "MS")
            cv_param.set("accession", "MS:1000041")
            cv_param.set("name", "charge state")
            cv_param.set("value", str(row['precursor_charge']))
        
        # Activation
        activation = ET.SubElement(precursor, ns_prefix + "activation")
        
        # Default to HCD if not specified
        activation_method = "HCD"
        if 'activation' in row:
            activation_method = row['activation']
        
        # Activation type
        activation_map = {
            "HCD": ("MS:1000422", "beam-type collision-induced dissociation"),
            "CID": ("MS:1000133", "collision-induced dissociation"),
            "ETD": ("MS:1000128", "electron transfer dissociation"),
            "ECD": ("MS:1000127", "electron capture dissociation"),
            "EAD": ("MS:1000129", "electron activated dissociation"),
            "EXD": ("MS:1000130", "electron induced dissociation"),
            "UVPD": ("MS:1000126", "ultraviolet photodissociation"),
            "ETHCD": ("MS:1000423", "electron transfer/higher-energy collision dissociation"),
            "ETCID": ("MS:1000424", "electron transfer/collision-induced dissociation"),
            "EXCID": ("MS:1000425", "electron capture/collision-induced dissociation"),
            "NETD": ("MS:1000426", "negative electron transfer dissociation")
        }
        
        # Add activation CV param
        if activation_method in activation_map:
            accession, name = activation_map[activation_method]
            cv_param = ET.SubElement(activation, ns_prefix + "cvParam")
            cv_param.set("cvRef", "MS")
            cv_param.set("accession", accession)
            cv_param.set("name", name)
            cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
        
        # Add collision energy if available
        if 'nce' in row and row['nce'] > 0:
            cv_param = ET.SubElement(activation, ns_prefix + "cvParam")
            cv_param.set("cvRef", "MS")
            cv_param.set("accession", "MS:1000045")
            cv_param.set("name", "collision energy")
            cv_param.set("value", str(row['nce']))
            cv_param.set("unitCvRef", "UO")
            cv_param.set("unitAccession", "UO:0000266")
            cv_param.set("unitName", "electronvolt")
    
    # Add binary data arrays
    binary_list = ET.SubElement(spectrum, ns_prefix + "binaryDataArrayList")
    binary_list.set("count", "2")  # m/z and intensity
    
    # Encode data
    format_char = 'f' if binary_precision == 32 else 'd'
    
    # Add m/z array
    binary_array = ET.SubElement(binary_list, ns_prefix + "binaryDataArray")
    
    if len(mz_array) > 0:
        buffer = struct.pack(f"<{len(mz_array)}{format_char}", *mz_array)
        if compression == 'zlib':
            buffer = zlib.compress(buffer)
        encoded_mz = base64.b64encode(buffer).decode('ascii')
    else:
        encoded_mz = ""
        
    binary_array.set("encodedLength", str(len(encoded_mz)))
    binary_array.set("arrayLength", str(len(mz_array)))
    
    # Binary precision
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000521" if binary_precision == 32 else "MS:1000523")
    cv_param.set("name", "32-bit float" if binary_precision == 32 else "64-bit float")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # Compression
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000574" if compression == 'zlib' else "MS:1000576")
    cv_param.set("name", "zlib compression" if compression == 'zlib' else "no compression")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # Array type
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000514")
    cv_param.set("name", "m/z array")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # With units
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000514")
    cv_param.set("name", "m/z array")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    cv_param.set("unitCvRef", "MS")
    cv_param.set("unitAccession", "MS:1000040")
    cv_param.set("unitName", "m/z")
    
    # Binary data
    binary = ET.SubElement(binary_array, ns_prefix + "binary")
    binary.text = encoded_mz
    
    # Add intensity array
    binary_array = ET.SubElement(binary_list, ns_prefix + "binaryDataArray")
    
    if len(intensity_array) > 0:
        buffer = struct.pack(f"<{len(intensity_array)}{format_char}", *intensity_array)
        if compression == 'zlib':
            buffer = zlib.compress(buffer)
        encoded_intensity = base64.b64encode(buffer).decode('ascii')
    else:
        encoded_intensity = ""
        
    binary_array.set("encodedLength", str(len(encoded_intensity)))
    binary_array.set("arrayLength", str(len(intensity_array)))
    
    # Binary precision
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000521" if binary_precision == 32 else "MS:1000523")
    cv_param.set("name", "32-bit float" if binary_precision == 32 else "64-bit float")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # Compression
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000574" if compression == 'zlib' else "MS:1000576")
    cv_param.set("name", "zlib compression" if compression == 'zlib' else "no compression")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # Array type
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000515")
    cv_param.set("name", "intensity array")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    
    # With units
    cv_param = ET.SubElement(binary_array, ns_prefix + "cvParam")
    cv_param.set("cvRef", "MS")
    cv_param.set("accession", "MS:1000515")
    cv_param.set("name", "intensity array")
    cv_param.set("value", "")  # ADD EMPTY VALUE ATTRIBUTE HERE
    cv_param.set("unitCvRef", "MS")
    cv_param.set("unitAccession", "MS:1000131")
    cv_param.set("unitName", "number of detector counts")
    
    # Binary data
    binary = ET.SubElement(binary_array, ns_prefix + "binary")
    binary.text = encoded_intensity
    
    # Convert to string
    return ET.tostring(spectrum, encoding="utf-8")

def _process_spectrum_range(range_tuple, ms_data, binary_precision, compression, ns_prefix):
    """Process a range of spectra defined by start and end indices"""
    start_idx, end_idx = range_tuple
    spectrum_strings = []
    
    for i in range(int(start_idx), int(end_idx)):
        try:
            row = ms_data.spectrum_df.iloc[i]
            spectrum_xml = _create_spectrum_xml(i, row, ms_data, binary_precision, compression, ns_prefix)
            spectrum_strings.append((i, spectrum_xml))
        except Exception as e:
            print(f"Error processing spectrum {i}: {e}")
    
    return spectrum_strings

def _process_offset_batch_mmap(args):
    """Process a batch of spectrum IDs using memory mapping"""
    file_path, start_idx, end_idx = args
    results = {}
    
    # Open the file using memory mapping (shared memory view)
    with open(file_path, 'rb') as f:
        import mmap
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i in range(int(start_idx), int(end_idx)):
                spectrum_id = f'scan={i}'
                # Look for the opening tag with this ID
                search_tag = f'id="{spectrum_id}"'.encode('utf-8')
                # Find the position of this tag in the file
                pos = mm.find(search_tag)
                if pos != -1:
                    # Find the start of the spectrum tag (<spectrum)
                    tag_start = -1
                    search_pos = pos
                    while search_pos > 0:
                        search_pos -= 1
                        mm.seek(search_pos)
                        if mm.read(9) == b'<spectrum':
                            tag_start = search_pos
                            break
                    
                    if tag_start != -1:
                        results[spectrum_id] = tag_start
    
    return results

class MzMLWriter:
    """
    Class for converting MSData_Base objects to mzML format.
    """
    def __init__(self, ms_data, output_path, binary_precision=64, compression=None, indexed=False, process_count=1, batch_size=5000):
        """
        Initialize the writer with an MSData_Base object.
        
        Parameters
        ----------
        ms_data : MSData_Base
            The mass spectrometry data object to convert
        output_path : str
            Path where the mzML file will be saved
        binary_precision : int, optional
            Binary encoding precision (32 or 64 bit), by default 64
        compression : str, optional
            Compression method (None, 'zlib'), by default None
        indexed : bool, optional
            Whether to create an indexed mzML file, by default False
        process_count : int, optional
            Number of processes to use for parallel processing, by default 1
        batch_size : int, optional
            Size of each batch for parallel processing, by default 5000
        """
        self.ms_data = ms_data
        self.output_path = output_path
        self.binary_precision = binary_precision
        self.compression = compression
        self.indexed = indexed
        self.process_count = process_count
        self.batch_size = batch_size
        self.ns_uri = "http://psi.hupo.org/ms/mzml"
        self.ns_prefix = "{" + self.ns_uri + "}"        

    def write(self):
        """
        Main method to write the mzML file
        """
        # Register namespace for proper output
        ET.register_namespace("", self.ns_uri)
        
        if self.indexed:
            # For indexed mzML, we need to create the mzML content first
            self.mzml_root = self._create_mzml_content()
            
            # Create the indexedmzML root
            self.root = ET.Element(self.ns_prefix + "indexedmzML", {
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.2_idx.xsd"
            })
            
            # Add the mzML content to indexedmzML
            self.root.append(self.mzml_root)
            
            # Write the content to a temporary file to get element offsets
            # Apply indentation before writing
            ET.indent(self.mzml_root)
            temp_path = self.output_path + ".temp"
            tree = ET.ElementTree(self.mzml_root)
            tree.write(temp_path, encoding="utf-8", xml_declaration=True)
            
            # Get offsets for elements by parsing the temp file
            print("Calculating byte offset between spectra...")
            offsets = self._calculate_offsets(temp_path)
            
            # Add index elements
            print("Adding indices...")
            self._add_index_elements(offsets)
            
            # Apply indentation to the full indexed document
            ET.indent(self.root)
            
            # Write final file
            print("Writing final file")
            tree = ET.ElementTree(self.root)
            tree.write(self.output_path, encoding="utf-8", xml_declaration=True)
            
            # Remove temp file
            os.remove(temp_path)
        else:
            # For regular mzML, just create and write content directly
            self.root = self._create_mzml_content()
            
            # Apply indentation
            ET.indent(self.root)
            
            tree = ET.ElementTree(self.root)
            tree.write(self.output_path, encoding="utf-8", xml_declaration=True)
    
    def _create_mzml_content(self):
        """Create the mzML content tree"""
        # Create root element for mzML
        root = ET.Element(self.ns_prefix + "mzML")
        root.set("version", "1.1.0")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:schemaLocation", "http://psi.hupo.org/ms/mzml http://psi.hupo.org/ms/mzml")
        
        # Add required elements
        self._add_cv_list(root)
        self._add_file_description(root)
        self._add_software_list(root)
        self._add_instrument_configuration_list(root)
        self._add_data_processing_list(root)
        self._add_run(root)
        
        return root
    
    def _calculate_offsets(self, temp_file_path):
        """Calculate byte offsets for each spectrum using memory-mapped file"""
        # Create batch ranges for parallelization
        spectrum_count = len(self.ms_data.spectrum_df)
        batch_size = 5000  # Adjust based on your needs
        boundaries = np.arange(0, spectrum_count, batch_size)
        boundaries = np.append(boundaries, spectrum_count)
        
        # Prepare arguments for the worker function - pass file path instead of content
        batch_args = [(temp_file_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # Use multiprocessing
        offsets = {}
        mode = "spawn" if platform.system() != "Linux" else "fork"
        with mp.get_context(mode).Pool(processes=self.process_count) as pool:
            batch_results = list(tqdm(
                pool.imap(_process_offset_batch_mmap, batch_args),
                total=len(batch_args),
                desc="Calculating offsets"
            ))
        
        # Combine results
        for batch_result in batch_results:
            offsets.update(batch_result)
    
        return offsets
    
    def _add_index_elements(self, offsets):
        """Add index list and related elements to the indexedmzML document"""
        # Create index list with count="2" (both spectrum and chromatogram indices required)
        index_list = ET.SubElement(self.root, self.ns_prefix + "indexList", {"count": "2"})
        
        # Spectrum index
        spectrum_index = ET.SubElement(index_list, self.ns_prefix + "index", {"name": "spectrum"})
        
        # Sort scan IDs numerically
        sorted_spectrum_ids = sorted(offsets.keys(), key=lambda x: int(x.split('=')[1]))
        
        # Add offset for each spectrum with idRef attribute (not id)
        for spectrum_id in sorted_spectrum_ids:
            ET.SubElement(spectrum_index, self.ns_prefix + "offset", {
                "idRef": spectrum_id
            }).text = str(offsets[spectrum_id])
        
        # Add empty chromatogram index (required by schema)
        ET.SubElement(index_list, self.ns_prefix + "index", {"name": "chromatogram"}).text = ""
        
        # Add indexListOffset (position where the indexList starts)
        with open(self.output_path + ".temp", 'rb') as f:
            content = f.read()
        index_list_pos = len(content)
        ET.SubElement(self.root, self.ns_prefix + "indexListOffset").text = str(index_list_pos)
        
        # Add file checksum (SHA-1)
        sha1 = hashlib.sha1(content).hexdigest()
        ET.SubElement(self.root, self.ns_prefix + "fileChecksum").text = sha1
        
    def _add_cv_list(self, parent_elem):
        """Add controlled vocabulary list"""
        cv_list = ET.SubElement(parent_elem, self.ns_prefix + "cvList")
        cv_list.set("count", "3")
        
        # MS CV
        cv = ET.SubElement(cv_list, self.ns_prefix + "cv")
        cv.set("id", "MS")
        cv.set("fullName", "Proteomics Standards Initiative Mass Spectrometry Ontology")
        cv.set("URI", "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo")
        
        # UO CV
        cv = ET.SubElement(cv_list, self.ns_prefix + "cv")
        cv.set("id", "UO")
        cv.set("fullName", "Unit Ontology")
        cv.set("URI", "http://ontologies.berkeleybop.org/uo.obo")
        
        # PSI-MS CV
        cv = ET.SubElement(cv_list, self.ns_prefix + "cv")
        cv.set("id", "PSI-MS")
        cv.set("fullName", "PSI-MS Controlled Vocabulary")
        cv.set("URI", "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo")
        
    def _add_file_description(self, parent_elem):
        """Add file description section"""
        file_description = ET.SubElement(parent_elem, self.ns_prefix + "fileDescription")
        
        # File content
        file_content = ET.SubElement(file_description, self.ns_prefix + "fileContent")
        
        # Add CV params for file content
        self._add_cv_param(file_content, "MS", "MS:1000579", "MS1 spectrum", "")
        
        if (self.ms_data.spectrum_df['ms_level'] == 2).any():
            self._add_cv_param(file_content, "MS", "MS:1000580", "MSn spectrum", "")
        
        # Source file list
        if hasattr(self.ms_data, 'raw_file_path') and self.ms_data.raw_file_path:
            source_file_list = ET.SubElement(file_description, self.ns_prefix + "sourceFileList")
            source_file_list.set("count", "1")
            
            source_file = ET.SubElement(source_file_list, self.ns_prefix + "sourceFile")
            source_file.set("id", "RAW1")
            source_file.set("name", os.path.basename(self.ms_data.raw_file_path))
            source_file.set("location", os.path.dirname(self.ms_data.raw_file_path))
            
            # Add CV params for source file
            if self.ms_data.file_type == "thermo":
                self._add_cv_param(source_file, "MS", "MS:1000563", self.ms_data.file_type, "")
                
    def _add_software_list(self, parent_elem):
        """Add software list section"""
        software_list = ET.SubElement(parent_elem, self.ns_prefix + "softwareList")
        software_list.set("count", "1")
        
        software = ET.SubElement(software_list, self.ns_prefix + "software")
        software.set("id", "alpharaw")
        software.set("version", "1.0")  # Use actual version when available
        
        self._add_cv_param(software, "MS", "MS:1000799", "custom software tool", "alpharaw")
        
    def _add_instrument_configuration_list(self, parent_elem):
        """Add instrument configuration list"""
        instrument_list = ET.SubElement(parent_elem, self.ns_prefix + "instrumentConfigurationList")
        instrument_list.set("count", "1")
        
        instrument = ET.SubElement(instrument_list, self.ns_prefix + "instrumentConfiguration")
        instrument.set("id", "IC1")
        
        self._add_cv_param(instrument, "MS", "MS:1000031", "instrument model", "")
        
        # Add component list
        component_list = ET.SubElement(instrument, self.ns_prefix + "componentList")
        component_list.set("count", "3")
        
        # Source
        source = ET.SubElement(component_list, self.ns_prefix + "source")
        source.set("order", "1")
        self._add_cv_param(source, "MS", "MS:1000073", "electrospray ionization", "")
        
        # Analyzer
        analyzer = ET.SubElement(component_list, self.ns_prefix + "analyzer")
        analyzer.set("order", "2")

        if hasattr(self.ms_data, 'auxiliary_items') and "analyzer" in self.ms_data.auxiliary_items and "analyzer" in self.ms_data.spectrum_df.columns:
            analyzer_type = self.ms_data.spectrum_df["analyzer"].iloc[0] if not self.ms_data.spectrum_df["analyzer"].empty else "orbitrap"
            self._add_cv_param(analyzer, "MS", "MS:1000484", analyzer_type, "")
        else:
            self._add_cv_param(analyzer, "MS", "MS:1000484", "orbitrap", "")
        
        # Detector
        detector = ET.SubElement(component_list, self.ns_prefix + "detector")
        detector.set("order", "3")
        self._add_cv_param(detector, "MS", "MS:1000253", "electron multiplier", "")
        
    def _add_data_processing_list(self, parent_elem):
        """Add data processing list"""
        data_processing_list = ET.SubElement(parent_elem, self.ns_prefix + "dataProcessingList")
        data_processing_list.set("count", "1")
        
        data_processing = ET.SubElement(data_processing_list, self.ns_prefix + "dataProcessing")
        data_processing.set("id", "alpharaw_processing")
        
        processing_method = ET.SubElement(data_processing, self.ns_prefix + "processingMethod")
        processing_method.set("order", "1")
        processing_method.set("softwareRef", "alpharaw")
        
        self._add_cv_param(processing_method, "MS", "MS:1000544", "Conversion to mzML", "")
        
    def _add_run(self, parent_elem):
        """Add run section with spectrum list using multiprocessing"""
        run = ET.SubElement(parent_elem, self.ns_prefix + "run")
        run.set("id", "run1")
        run.set("defaultInstrumentConfigurationRef", "IC1")
        
        # Add spectrum list
        spectrum_list = ET.SubElement(run, self.ns_prefix + "spectrumList")
        spectrum_count = len(self.ms_data.spectrum_df)
        spectrum_list.set("count", str(spectrum_count))
        spectrum_list.set("defaultDataProcessingRef", "alpharaw_processing")
                        
        # Create batch ranges using your colleagues' approach
        first_spectrum_number = 0
        last_spectrum_number = spectrum_count - 1
        
        # Create batch boundaries
        boundaries = np.arange(first_spectrum_number, last_spectrum_number + 1, self.batch_size)
        boundaries = np.append(boundaries, last_spectrum_number + 1)
        
        # Create start-end pairs for each batch
        batch_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        
        # Process batches in parallel
        process_batch_partial = partial(
            _process_spectrum_range,  # New function defined below
            ms_data=self.ms_data,
            binary_precision=self.binary_precision,
            compression=self.compression,
            ns_prefix=self.ns_prefix
        )
        
        spectra_by_index = {}
        
        print(f"Processing {spectrum_count} spectra on {min(self.process_count,mp.cpu_count())} cores using {len(batch_pairs)} batches...")
        
        # Use multiprocessing pool
        mode = "spawn" if platform.system() != "Linux" else "fork"  # Try fork instead of forkserver for better performance
        with mp.get_context(mode).Pool(processes=self.process_count) as pool:
            for batch_results in tqdm(pool.imap(process_batch_partial, batch_pairs), 
                                    total=len(batch_pairs), 
                                    desc="Processing spectra"):
                for idx, spectrum_xml in batch_results:
                    spectra_by_index[idx] = spectrum_xml
        
        print("Adding spectra to mzML file...")
        
        # Add spectra to spectrum_list in correct order
        for i in range(spectrum_count):
            if i in spectra_by_index:
                # Add pre-rendered XML string
                spectrum_element = ET.fromstring(spectra_by_index[i])
                spectrum_list.append(spectrum_element)
            else:
                # Fallback for any missing spectra
                try:
                    row = self.ms_data.spectrum_df.iloc[i]
                    self._add_spectrum(spectrum_list, i, row)
                except Exception as e:
                    print(f"Error adding spectrum {i}: {e}")
                    # Add empty spectrum
                    spectrum = ET.SubElement(spectrum_list, self.ns_prefix + "spectrum")
                    spectrum.set("index", str(i))
                    spectrum.set("id", f"scan={i}")
                    spectrum.set("defaultArrayLength", "0")
        
    def _add_spectrum(self, spectrum_list, i, row):
        """Add a single spectrum (fallback method)"""
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
        
        # MS level
        ms_level = row.get('ms_level', 1)
        self._add_cv_param(spectrum, "MS", "MS:1000511", "ms level", str(ms_level))

        # Add centroid/profile indication
        if self.ms_data.centroided:
            self._add_cv_param(spectrum, "MS", "MS:1000127", "centroid spectrum", "")
        else:
            self._add_cv_param(spectrum, "MS", "MS:1000128", "profile spectrum", "")
        
        # Add scan list
        scan_list = ET.SubElement(spectrum, self.ns_prefix + "scanList")
        scan_list.set("count", "1")
        
        scan = ET.SubElement(scan_list, self.ns_prefix + "scan")
        
        # Add retention time
        rt_seconds = row['rt'] * 60  # Convert to seconds
        self._add_cv_param(scan, "MS", "MS:1000016", "scan start time", str(rt_seconds), 
                          "UO", "UO:0000010", "second")
        
        # Add precursor information for MS2+ spectra
        if ms_level > 1:
            precursor_list = ET.SubElement(spectrum, self.ns_prefix + "precursorList")
            precursor_list.set("count", "1")
            
            precursor = ET.SubElement(precursor_list, self.ns_prefix + "precursor")
            
            # Isolation window
            isolation_window = ET.SubElement(precursor, self.ns_prefix + "isolationWindow")
            
            precursor_mz = row.get('precursor_mz', 0)
            isolation_lower = row.get('isolation_lower_mz', precursor_mz - 1.5)
            isolation_upper = row.get('isolation_upper_mz', precursor_mz + 1.5)
            
            self._add_cv_param(isolation_window, "MS", "MS:1000827", "isolation window target m/z", 
                              str(precursor_mz), "MS", "MS:1000040", "m/z")
            self._add_cv_param(isolation_window, "MS", "MS:1000828", "isolation window lower offset", 
                              str(precursor_mz - isolation_lower), "MS", "MS:1000040", "m/z")
            self._add_cv_param(isolation_window, "MS", "MS:1000829", "isolation window upper offset", 
                              str(isolation_upper - precursor_mz), "MS", "MS:1000040", "m/z")
            
            # Selected ion list
            selected_ion_list = ET.SubElement(precursor, self.ns_prefix + "selectedIonList")
            selected_ion_list.set("count", "1")
            
            selected_ion = ET.SubElement(selected_ion_list, self.ns_prefix + "selectedIon")
            self._add_cv_param(selected_ion, "MS", "MS:1000744", "selected ion m/z", 
                              str(precursor_mz), "MS", "MS:1000040", "m/z")
            
            if 'precursor_charge' in row and row['precursor_charge'] > 0:
                self._add_cv_param(selected_ion, "MS", "MS:1000041", "charge state", 
                                  str(row['precursor_charge']))
            
            # Activation
            activation = ET.SubElement(precursor, self.ns_prefix + "activation")
            
            # Default to HCD if not specified
            activation_method = "HCD"
            if 'activation' in row:
                activation_method = row['activation']
                
            # Add appropriate activation parameter based on method
            activation_map = {
                "HCD": ("MS:1000422", "beam-type collision-induced dissociation"),
                "CID": ("MS:1000133", "collision-induced dissociation"),
                "ETD": ("MS:1000128", "electron transfer dissociation"),
                "ECD": ("MS:1000127", "electron capture dissociation"),
                "EAD": ("MS:1000129", "electron activated dissociation"),
                "EXD": ("MS:1000130", "electron induced dissociation"),
                "UVPD": ("MS:1000126", "ultraviolet photodissociation"),
                "ETHCD": ("MS:1000423", "electron transfer/higher-energy collision dissociation"),
                "ETCID": ("MS:1000424", "electron transfer/collision-induced dissociation"),
                "EXCID": ("MS:1000425", "electron capture/collision-induced dissociation"),
                "NETD": ("MS:1000426", "negative electron transfer dissociation")
            }
            
            if activation_method in activation_map:
                accession, name = activation_map[activation_method]
                self._add_cv_param(activation, "MS", accession, name, "")
            
            # Add collision energy if available
            if 'nce' in row and row['nce'] > 0:
                self._add_cv_param(activation, "MS", "MS:1000045", "collision energy", 
                                  str(row['nce']), "UO", "UO:0000266", "electronvolt")
        
        # Add binary data arrays
        binary_list = ET.SubElement(spectrum, self.ns_prefix + "binaryDataArrayList")
        binary_list.set("count", "2")  # m/z and intensity
        
        # m/z array
        self._add_binary_data_array(binary_list, mz_array, "MS:1000514", "m/z array", 
                                   "MS:1000040", "m/z")
        
        # Intensity array
        self._add_binary_data_array(binary_list, intensity_array, "MS:1000515", "intensity array", 
                                   "MS:1000131", "number of detector counts")
        
    def _add_binary_data_array(self, parent, data, array_type_acc, array_type_name, unit_acc=None, unit_name=None):
        """Add binary data array element"""
        binary_array = ET.SubElement(parent, self.ns_prefix + "binaryDataArray")
        
        # Encode data
        encoded_data = ""
        if len(data) > 0:
            # Format string: '<' for little-endian, 'f' for 32-bit float, 'd' for 64-bit float
            format_char = 'f' if self.binary_precision == 32 else 'd'
            buffer = struct.pack(f"<{len(data)}{format_char}", *data)
            
            # Apply compression if requested
            if self.compression == 'zlib':
                buffer = zlib.compress(buffer)
            
            encoded_data = base64.b64encode(buffer).decode('ascii')
        
        # Set required attributes
        binary_array.set("encodedLength", str(len(encoded_data)))
        binary_array.set("arrayLength", str(len(data)))
        
        # Data type and compression
        if self.binary_precision == 32:
            self._add_cv_param(binary_array, "MS", "MS:1000521", "32-bit float", "")
        else:
            self._add_cv_param(binary_array, "MS", "MS:1000523", "64-bit float", "")
        
        # Compression
        if self.compression == 'zlib':
            self._add_cv_param(binary_array, "MS", "MS:1000574", "zlib compression", "")
        else:
            self._add_cv_param(binary_array, "MS", "MS:1000576", "no compression", "")
        
        self._add_cv_param(binary_array, "MS", array_type_acc, array_type_name, "")
        
        if unit_acc and unit_name:
            self._add_cv_param(binary_array, "MS", array_type_acc, array_type_name, "", 
                              "MS", unit_acc, unit_name)
        
        # Add binary element
        binary = ET.SubElement(binary_array, self.ns_prefix + "binary")
        binary.text = encoded_data
        
    def _add_cv_param(self, parent, cv_ref, accession, name, value="", unit_cv_ref=None, 
                    unit_accession=None, unit_name=None):
        """Helper method to add a CV parameter"""
        cv_param = ET.SubElement(parent, self.ns_prefix + "cvParam")
        cv_param.set("cvRef", cv_ref)
        cv_param.set("accession", accession)
        cv_param.set("name", name)
        
        # Always include a value attribute, even if it's empty
        cv_param.set("value", value if value is not None else "")
            
        if unit_cv_ref:
            cv_param.set("unitCvRef", unit_cv_ref)
            cv_param.set("unitAccession", unit_accession)
            cv_param.set("unitName", unit_name)
