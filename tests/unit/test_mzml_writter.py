import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from pathlib import Path

from alpharaw.mzml_io.mzml_writer import MzMLWriter

# Create a simple mock MSData object for testing
class MockMSData:
    def __init__(self):
        import pandas as pd
        
        # Create a simple spectrum dataframe with minimal required columns
        self.spectrum_df = pd.DataFrame({
            'spec_idx': [0, 1, 2],
            'rt': [1.0, 2.0, 3.0],  # retention time in minutes
            'ms_level': [1, 2, 1],
            'precursor_mz': [-1.0, 400.0, -1.0],
            'precursor_charge': [0, 2, 0],
            'isolation_lower_mz': [-1.0, 398.5, -1.0],
            'isolation_upper_mz': [-1.0, 401.5, -1.0],
            'nce': [0.0, 25.0, 0.0],
            'peak_start_idx': [0, 3, 6],
            'peak_stop_idx': [3, 6, 9]
        })
        
        # Create a simple peak dataframe with mz and intensity values
        self.peak_df = pd.DataFrame({
            'mz': np.array([100.0, 200.0, 300.0, 401.0, 402.0, 403.0, 500.0, 600.0, 700.0], dtype=np.float32),
            'intensity': np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0], dtype=np.float32)
        })
        
        # Additional required attributes
        self.centroided = True
        self.file_type = "test"
        self.raw_file_path = "test_file.raw"
        self.creation_time = "2025-04-14T12:00:00"
    
    def get_peaks(self, spec_idx):
        """Return mz and intensity arrays for the given spectrum index"""
        start, end = self.spectrum_df[['peak_start_idx', 'peak_stop_idx']].values[spec_idx, :]
        return (
            self.peak_df.mz.values[start:end],
            self.peak_df.intensity.values[start:end]
        )


class TestMzMLWriter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, creating a mock MSData object and a temporary directory"""
        self.ms_data = MockMSData()
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()
    
    def test_basic_mzml_creation(self):
        """Test the basic creation of a non-indexed mzML file"""
        output_path = os.path.join(self.temp_dir.name, "test_output.mzML")
        
        # Create and write a non-indexed mzML file
        writer = MzMLWriter(self.ms_data, output_path, indexed=False)
        writer.write()
        
        # Check that the file was created
        assert os.path.exists(output_path)
        
        # Try to parse the file to make sure it's valid XML
        try:
            tree = ET.parse(output_path)
            root = tree.getroot()
        except Exception as e:
            pytest.fail(f"Failed to parse the generated mzML file: {e}")
        
        # Check basic structure of the mzML file
        ns = {'mzml': writer.ns_uri}
        
        # Check that we have the right root element
        assert root.tag.endswith("mzML")
        
        # Check spectrum count
        spectrum_list = root.findall(".//mzml:spectrumList", ns)
        assert len(spectrum_list) == 1
        assert spectrum_list[0].get("count") == "3"  # We created 3 spectra
        
        # Check that spectra have the right structure
        spectra = root.findall(".//mzml:spectrum", ns)
        assert len(spectra) == 3
        
        # Check that all CV parameters have a value attribute
        cv_params = root.findall(".//mzml:cvParam", ns)
        for param in cv_params:
            assert "value" in param.attrib
    
    def test_indexed_mzml_creation(self):
        """Test the creation of an indexed mzML file"""
        output_path = os.path.join(self.temp_dir.name, "test_output_indexed.mzML")
        
        # Create and write an indexed mzML file
        writer = MzMLWriter(self.ms_data, output_path, indexed=True)
        writer.write()
        
        # Check that the file was created
        assert os.path.exists(output_path)
        
        # Try to parse the file to make sure it's valid XML
        try:
            tree = ET.parse(output_path)
            root = tree.getroot()
        except Exception as e:
            pytest.fail(f"Failed to parse the generated indexed mzML file: {e}")
        
        # Check basic structure of the indexed mzML file
        ns = {'mzml': writer.ns_uri}
        
        # Check that we have the right root element
        assert root.tag.endswith("indexedmzML")
        
        # Check that the mzML element is present
        mzml = root.find(".//mzml:mzML", ns)
        assert mzml is not None
        
        # Check that the indexList element is present
        index_list = root.find(".//mzml:indexList", ns)
        assert index_list is not None
        assert index_list.get("count") == "2"  # One for spectra, one for chromatograms
        
        # Check that the indexListOffset element is present
        index_list_offset = root.find(".//mzml:indexListOffset", ns)
        assert index_list_offset is not None
        assert index_list_offset.text is not None  # Should contain an offset value
        
        # Check that the fileChecksum element is present
        file_checksum = root.find(".//mzml:fileChecksum", ns)
        assert file_checksum is not None
        assert file_checksum.text is not None  # Should contain a SHA-1 checksum
    
    def test_spectrum_content(self):
        """Test that spectra in the mzML file have the expected content"""
        output_path = os.path.join(self.temp_dir.name, "test_spectrum_content.mzML")
        
        # Create and write a non-indexed mzML file
        writer = MzMLWriter(self.ms_data, output_path, indexed=False)
        writer.write()
        
        # Parse the file
        tree = ET.parse(output_path)
        root = tree.getroot()
        ns = {'mzml': writer.ns_uri}
        
        # Get all spectra
        spectra = root.findall(".//mzml:spectrum", ns)
        
        # Check first spectrum (MS1)
        spec1 = spectra[0]
        assert spec1.get("index") == "0"
        assert spec1.get("id") == "scan=0"
        
        # Check MS level
        ms_level = spec1.find(".//mzml:cvParam[@name='ms level']", ns)
        assert ms_level is not None
        assert ms_level.get("value") == "1.0"
        
        # Check that it has a scan with retention time
        scan_time = spec1.find(".//mzml:scan/mzml:cvParam[@name='scan start time']", ns)
        assert scan_time is not None
        assert float(scan_time.get("value")) == pytest.approx(60.0)  # 1.0 min converted to seconds
        
        # Check second spectrum (MS2)
        spec2 = spectra[1]
        assert spec2.get("index") == "1"
        
        # Check MS level
        ms_level = spec2.find(".//mzml:cvParam[@name='ms level']", ns)
        assert ms_level is not None
        assert ms_level.get("value") == "2.0"
        
        # Check precursor information
        precursor = spec2.find(".//mzml:precursorList/mzml:precursor", ns)
        assert precursor is not None
        
        # Check isolation window
        isolation_window = precursor.find(".//mzml:isolationWindow", ns)
        assert isolation_window is not None
        
        # Check target m/z
        target_mz = isolation_window.find(".//mzml:cvParam[@name='isolation window target m/z']", ns)
        assert target_mz is not None
        assert float(target_mz.get("value")) == pytest.approx(400.0)
        
        # Check that there are binary data arrays
        binary_arrays = spec1.findall(".//mzml:binaryDataArray", ns)
        assert len(binary_arrays) == 2  # m/z and intensity
        
        # Check that binary arrays have data
        binary_data = binary_arrays[0].find(".//mzml:binary", ns)
        assert binary_data is not None
        assert binary_data.text is not None  # Should contain base64-encoded data


if __name__ == "__main__":
    unittest.main()