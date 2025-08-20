import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from alpharaw.mzml_io.mzml_writer import MzMLWriter


# Create a simple mock MSData object for testing
class MockMSData:
    def __init__(self):
        import pandas as pd

        # Create a simple spectrum dataframe with minimal required columns
        self.spectrum_df = pd.DataFrame(
            {
                "spec_idx": [0, 1, 2],
                "rt": [1.0, 2.0, 3.0],  # retention time in minutes
                "ms_level": [1, 2, 1],
                "precursor_mz": [-1.0, 400.0, -1.0],
                "precursor_charge": [0, 2, 0],
                "isolation_lower_mz": [-1.0, 398.5, -1.0],
                "isolation_upper_mz": [-1.0, 401.5, -1.0],
                "nce": [0.0, 25.0, 0.0],
                "activation": ["", "HCD", ""],
                "peak_start_idx": [0, 3, 6],
                "peak_stop_idx": [3, 6, 9],
            }
        )

        # Create a simple peak dataframe with mz and intensity values
        self.peak_df = pd.DataFrame(
            {
                "mz": np.array(
                    [100.0, 200.0, 300.0, 401.0, 402.0, 403.0, 500.0, 600.0, 700.0],
                    dtype=np.float32,
                ),
                "intensity": np.array(
                    [
                        1000.0,
                        2000.0,
                        3000.0,
                        4000.0,
                        5000.0,
                        6000.0,
                        7000.0,
                        8000.0,
                        9000.0,
                    ],
                    dtype=np.float32,
                ),
            }
        )

        # Additional required attributes
        self.centroided = True
        self.file_type = "thermo"
        self.raw_file_path = "/path/to/test_file.raw"
        self.creation_time = "2025-04-14T12:00:00"

    def get_peaks(self, spec_idx):
        """Return mz and intensity arrays for the given spectrum index"""
        start, end = self.spectrum_df[["peak_start_idx", "peak_stop_idx"]].values[
            spec_idx, :
        ]
        return (
            self.peak_df.mz.values[start:end],
            self.peak_df.intensity.values[start:end],
        )


@pytest.fixture
def ms_data():
    """Fixture to provide mock MS data for tests"""
    return MockMSData()


@pytest.fixture
def temp_dir():
    """Fixture to provide a temporary directory for test outputs"""
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir.name
    temp_dir.cleanup()


def test_basic_mzml_creation(ms_data, temp_dir):
    """Test the basic creation of an mzML file"""
    output_path = os.path.join(temp_dir, "test_output.mzML")

    # Create and write an mzML file
    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    # Check that the file was created
    assert os.path.exists(output_path)

    # Parse the file to make sure it's valid XML
    tree = ET.parse(output_path)
    root = tree.getroot()

    # Check basic structure of the mzML file
    ns = {"mzml": writer.ns_uri}

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


def test_binary_precision_32_bit(ms_data, temp_dir):
    """Test 32-bit binary precision (default)"""
    output_path = os.path.join(temp_dir, "test_32bit.mzML")

    # Create writer with default (32-bit) precision
    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    # Parse and check precision CV param
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    precision_params = root.findall(".//mzml:cvParam[@accession='MS:1000521']", ns)
    assert len(precision_params) > 0  # Should have 32-bit float params
    assert precision_params[0].get("name") == "32-bit float"


def test_binary_precision_64_bit(ms_data, temp_dir):
    """Test 64-bit binary precision (explicit)"""
    output_path = os.path.join(temp_dir, "test_64bit.mzML")

    # Create writer with explicit 64-bit precision
    writer = MzMLWriter(ms_data, output_path, binary_precision=64)
    writer.write()

    # Parse and check precision CV param
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    precision_params = root.findall(".//mzml:cvParam[@accession='MS:1000523']", ns)
    assert len(precision_params) > 0  # Should have 64-bit float params
    assert precision_params[0].get("name") == "64-bit float"


def test_compression_none(ms_data, temp_dir):
    """Test no compression (default)"""
    output_path = os.path.join(temp_dir, "test_no_compression.mzML")

    writer = MzMLWriter(ms_data, output_path, compression=None)
    writer.write()

    # Parse and check compression CV param
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    compression_params = root.findall(".//mzml:cvParam[@accession='MS:1000576']", ns)
    assert len(compression_params) > 0  # Should have no compression params
    assert compression_params[0].get("name") == "no compression"


def test_compression_zlib(ms_data, temp_dir):
    """Test zlib compression"""
    output_path = os.path.join(temp_dir, "test_zlib_compression.mzML")

    writer = MzMLWriter(ms_data, output_path, compression="zlib")
    writer.write()

    # Parse and check compression CV param
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    compression_params = root.findall(".//mzml:cvParam[@accession='MS:1000574']", ns)
    assert len(compression_params) > 0  # Should have zlib compression params
    assert compression_params[0].get("name") == "zlib compression"


def test_spectrum_content(ms_data, temp_dir):
    """Test that spectra in the mzML file have the expected content"""
    output_path = os.path.join(temp_dir, "test_spectrum_content.mzML")

    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    # Parse the file
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    # Get all spectra
    spectra = root.findall(".//mzml:spectrum", ns)

    # Check first spectrum (MS1)
    spec1 = spectra[0]
    assert spec1.get("index") == "0"
    assert spec1.get("id") == "scan=0"

    # Check MS level - should be integer, not float
    ms_level = spec1.find(".//mzml:cvParam[@name='ms level']", ns)
    assert ms_level is not None
    assert ms_level.get("value") == "1"  # Should be "1", not "1.0"

    # Check that it has a scan with retention time
    scan_time = spec1.find(".//mzml:scan/mzml:cvParam[@name='scan start time']", ns)
    assert scan_time is not None
    assert float(scan_time.get("value")) == pytest.approx(
        60.0
    )  # 1.0 min converted to seconds

    # Check second spectrum (MS2)
    spec2 = spectra[1]
    assert spec2.get("index") == "1"

    # Check MS level
    ms_level = spec2.find(".//mzml:cvParam[@name='ms level']", ns)
    assert ms_level is not None
    assert ms_level.get("value") == "2"  # Should be "2", not "2.0"

    # Check precursor information
    precursor = spec2.find(".//mzml:precursorList/mzml:precursor", ns)
    assert precursor is not None

    # Check isolation window
    isolation_window = precursor.find(".//mzml:isolationWindow", ns)
    assert isolation_window is not None

    # Check target m/z
    target_mz = isolation_window.find(
        ".//mzml:cvParam[@name='isolation window target m/z']", ns
    )
    assert target_mz is not None
    assert float(target_mz.get("value")) == pytest.approx(400.0)

    # Check activation method
    activation = spec2.find(
        ".//mzml:activation/mzml:cvParam[@accession='MS:1000422']", ns
    )
    assert activation is not None
    assert activation.get("name") == "beam-type collision-induced dissociation"

    # Check that there are binary data arrays
    binary_arrays = spec1.findall(".//mzml:binaryDataArray", ns)
    assert len(binary_arrays) == 2  # m/z and intensity

    # Check that binary arrays have data
    binary_data = binary_arrays[0].find(".//mzml:binary", ns)
    assert binary_data is not None
    assert binary_data.text is not None  # Should contain base64-encoded data


def test_centroided_flag(ms_data, temp_dir):
    """Test that centroided flag is properly set"""
    output_path = os.path.join(temp_dir, "test_centroided.mzML")

    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    # Check for centroid spectrum CV param
    centroid_params = root.findall(".//mzml:cvParam[@accession='MS:1000127']", ns)
    assert len(centroid_params) > 0
    assert centroid_params[0].get("name") == "centroid spectrum"


def test_cv_list_structure(ms_data, temp_dir):
    """Test that CV list has the correct structure"""
    output_path = os.path.join(temp_dir, "test_cv_list.mzML")

    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    # Check CV list
    cv_list = root.find(".//mzml:cvList", ns)
    assert cv_list is not None
    assert cv_list.get("count") == "3"

    # Check individual CVs
    cvs = cv_list.findall(".//mzml:cv", ns)
    assert len(cvs) == 3

    cv_ids = [cv.get("id") for cv in cvs]
    assert "MS" in cv_ids
    assert "UO" in cv_ids
    assert "PSI-MS" in cv_ids


def test_software_info(ms_data, temp_dir):
    """Test that software information is correctly set"""
    output_path = os.path.join(temp_dir, "test_software.mzML")

    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    # Check software list
    software = root.find(".//mzml:software[@id='alpharaw']", ns)
    assert software is not None

    # Check software CV param - should be "analysis software" not "custom unreleased software tool"
    software_param = software.find(".//mzml:cvParam[@accession='MS:1001456']", ns)
    assert software_param is not None
    assert software_param.get("name") == "analysis software"
    assert software_param.get("value") == "alpharaw"


def test_empty_spectrum_handling(temp_dir):
    """Test handling of spectra with no peaks"""
    # Create mock data with empty spectrum
    empty_ms_data = MockMSData()
    empty_ms_data.spectrum_df.loc[0, "peak_start_idx"] = 0
    empty_ms_data.spectrum_df.loc[0, "peak_stop_idx"] = 0  # Empty spectrum

    output_path = os.path.join(temp_dir, "test_empty_spectrum.mzML")

    writer = MzMLWriter(empty_ms_data, output_path)
    writer.write()

    # Should not raise an exception
    assert os.path.exists(output_path)

    # Parse and check that spectrum exists with defaultArrayLength="0"
    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    first_spectrum = root.find(".//mzml:spectrum[@index='0']", ns)
    assert first_spectrum is not None
    assert first_spectrum.get("defaultArrayLength") == "0"


def test_integer_values_formatting(ms_data, temp_dir):
    """Test that integer values (MS level, charge state) are properly formatted"""
    output_path = os.path.join(temp_dir, "test_integers.mzML")

    writer = MzMLWriter(ms_data, output_path)
    writer.write()

    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"mzml": writer.ns_uri}

    # Check MS level values are integers
    ms_level_params = root.findall(".//mzml:cvParam[@accession='MS:1000511']", ns)
    for param in ms_level_params:
        value = param.get("value")
        assert value.isdigit(), f"MS level should be integer format, got {value}"

    # Check charge state values are integers
    charge_params = root.findall(".//mzml:cvParam[@accession='MS:1000041']", ns)
    for param in charge_params:
        value = param.get("value")
        assert value.isdigit(), f"Charge state should be integer format, got {value}"
