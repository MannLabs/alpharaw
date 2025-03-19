import numpy as np
import pandas as pd

from alpharaw.ms_data_base import MSData_Base


# Define a custom subclass for testing
class CustomMSData(MSData_Base):
    """A custom subclass of MSData_Base for testing inheritance behavior"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_attribute = "custom_value"
        self.another_attribute = 42


def test_remove_unused_peaks_basic():
    """Test the basic functionality of remove_unused_peaks"""
    # Create a simple MSData_Base instance
    ms_data = MSData_Base()

    # Create spectrum_df with 3 spectra
    ms_data.spectrum_df = pd.DataFrame(
        {
            "spec_idx": [0, 1, 2],
            "rt": [1.0, 2.0, 3.0],
            "ms_level": [1, 2, 2],
            "peak_start_idx": [0, 5, 10],
            "peak_stop_idx": [5, 10, 15],
        }
    )

    # Create peak_df with 15 peaks (5 per spectrum)
    ms_data.peak_df = pd.DataFrame(
        {
            "mz": np.arange(15, dtype=np.float32) + 100,
            "intensity": np.arange(15, dtype=np.float32) * 1000,
        }
    )

    # Test in-place modification
    result = ms_data.remove_unused_peaks(in_place=True)

    # Verify that the method returns None for in-place modification
    assert result is None

    # Verify that the peak indices remain the same since no spectra were removed
    assert len(ms_data.peak_df) == 15
    assert np.array_equal(ms_data.spectrum_df["peak_start_idx"].values, [0, 5, 10])
    assert np.array_equal(ms_data.spectrum_df["peak_stop_idx"].values, [5, 10, 15])


def test_remove_unused_peaks_with_filtering():
    """Test remove_unused_peaks when spectra have been removed"""
    # Create a simple MSData_Base instance
    ms_data = MSData_Base()

    # Create spectrum_df with 3 spectra initially
    ms_data.spectrum_df = pd.DataFrame(
        {
            "spec_idx": [0, 1, 2],
            "rt": [1.0, 2.0, 3.0],
            "ms_level": [1, 2, 2],
            "peak_start_idx": [0, 5, 10],
            "peak_stop_idx": [5, 10, 15],
        }
    )

    # Create peak_df with 15 peaks (5 per spectrum)
    ms_data.peak_df = pd.DataFrame(
        {
            "mz": np.arange(15, dtype=np.float32) + 100,
            "intensity": np.arange(15, dtype=np.float32) * 1000,
        }
    )

    # Filter out the middle spectrum (index 1)
    ms_data.spectrum_df = ms_data.spectrum_df.loc[[0, 2]].reset_index(drop=True)
    ms_data.spectrum_df["spec_idx"] = ms_data.spectrum_df.index

    # Now call remove_unused_peaks with in_place=False to get a new instance
    new_ms_data = ms_data.remove_unused_peaks(in_place=False)

    # Verify that the original instance is unchanged
    assert len(ms_data.peak_df) == 15
    assert np.array_equal(ms_data.spectrum_df["peak_start_idx"].values, [0, 10])
    assert np.array_equal(ms_data.spectrum_df["peak_stop_idx"].values, [5, 15])

    # Verify that the new instance has the correct peak data
    assert len(new_ms_data.peak_df) == 10  # 5 peaks for spec 0 and 5 for spec 2
    assert np.array_equal(new_ms_data.spectrum_df["peak_start_idx"].values, [0, 5])
    assert np.array_equal(new_ms_data.spectrum_df["peak_stop_idx"].values, [5, 10])

    # Verify that the peak values are correct
    expected_mzs = np.concatenate(
        [
            np.arange(5, dtype=np.float32) + 100,  # First 5 peaks from spec 0
            np.arange(5, dtype=np.float32) + 110,  # Last 5 peaks from spec 2
        ]
    )
    expected_intensities = np.concatenate(
        [
            np.arange(5, dtype=np.float32) * 1000,  # First 5 peaks from spec 0
            np.arange(10, 15, dtype=np.float32) * 1000,  # Last 5 peaks from spec 2
        ]
    )

    assert np.array_equal(new_ms_data.peak_df["mz"].values, expected_mzs)
    assert np.array_equal(new_ms_data.peak_df["intensity"].values, expected_intensities)


def test_remove_unused_peaks_with_extra_columns():
    """Test remove_unused_peaks preserves additional columns in peak_df"""
    # Create a simple MSData_Base instance
    ms_data = MSData_Base()

    # Create spectrum_df with 3 spectra
    ms_data.spectrum_df = pd.DataFrame(
        {
            "spec_idx": [0, 1, 2],
            "rt": [1.0, 2.0, 3.0],
            "ms_level": [1, 2, 2],
            "peak_start_idx": [0, 5, 10],
            "peak_stop_idx": [5, 10, 15],
            "extra_spec_col": ["a", "b", "c"],  # Extra column in spectrum_df
        }
    )

    # Create peak_df with 15 peaks and an extra column
    ms_data.peak_df = pd.DataFrame(
        {
            "mz": np.arange(15, dtype=np.float32) + 100,
            "intensity": np.arange(15, dtype=np.float32) * 1000,
            "extra_peak_col": np.arange(15, dtype=np.float32)
            * 0.1,  # Extra column in peak_df
            "extra_str_col": [f"peak_{i}" for i in range(15)],  # String column
        }
    )

    # Filter out the middle spectrum (index 1)
    ms_data.spectrum_df = ms_data.spectrum_df.loc[[0, 2]].reset_index(drop=True)
    ms_data.spectrum_df["spec_idx"] = ms_data.spectrum_df.index

    # Call remove_unused_peaks with in_place=True
    ms_data.remove_unused_peaks(in_place=True)

    # Verify that the peak data is correct
    assert len(ms_data.peak_df) == 10  # 5 peaks for spec 0 and 5 for spec 2
    assert np.array_equal(ms_data.spectrum_df["peak_start_idx"].values, [0, 5])
    assert np.array_equal(ms_data.spectrum_df["peak_stop_idx"].values, [5, 10])

    # Verify that extra columns are preserved
    assert "extra_peak_col" in ms_data.peak_df.columns
    assert "extra_str_col" in ms_data.peak_df.columns
    assert "extra_spec_col" in ms_data.spectrum_df.columns

    # Verify that the extra column values are correctly mapped
    expected_extra_peak_col = np.concatenate(
        [
            np.arange(5, dtype=np.float32) * 0.1,  # First 5 peaks from spec 0
            np.arange(10, 15, dtype=np.float32) * 0.1,  # Last 5 peaks from spec 2
        ]
    )
    expected_extra_str_col = [f"peak_{i}" for i in list(range(5)) + list(range(10, 15))]

    assert np.array_equal(
        ms_data.peak_df["extra_peak_col"].values, expected_extra_peak_col
    )
    assert ms_data.peak_df["extra_str_col"].tolist() == expected_extra_str_col
    assert ms_data.spectrum_df["extra_spec_col"].tolist() == ["a", "c"]


def test_remove_unused_peaks_with_subclass():
    """Test remove_unused_peaks preserves class type and custom attributes"""
    # Create a custom subclass instance
    ms_data = CustomMSData()

    # Create spectrum_df with 3 spectra
    ms_data.spectrum_df = pd.DataFrame(
        {
            "spec_idx": [0, 1, 2],
            "rt": [1.0, 2.0, 3.0],
            "ms_level": [1, 2, 2],
            "peak_start_idx": [0, 5, 10],
            "peak_stop_idx": [5, 10, 15],
        }
    )

    # Create peak_df with 15 peaks
    ms_data.peak_df = pd.DataFrame(
        {
            "mz": np.arange(15, dtype=np.float32) + 100,
            "intensity": np.arange(15, dtype=np.float32) * 1000,
        }
    )

    # Filter out the middle spectrum (index 1)
    ms_data.spectrum_df = ms_data.spectrum_df.loc[[0, 2]].reset_index(drop=True)
    ms_data.spectrum_df["spec_idx"] = ms_data.spectrum_df.index

    # Call remove_unused_peaks with in_place=False to get a new instance
    new_ms_data = ms_data.remove_unused_peaks(in_place=False)

    # Verify that the new instance is of the same class
    assert isinstance(new_ms_data, CustomMSData)
    assert type(new_ms_data) == type(ms_data)

    # Verify that custom attributes are preserved
    assert hasattr(new_ms_data, "custom_attribute")
    assert hasattr(new_ms_data, "another_attribute")
    assert new_ms_data.custom_attribute == "custom_value"
    assert new_ms_data.another_attribute == 42

    # Verify that the peak data is correct
    assert len(new_ms_data.peak_df) == 10  # 5 peaks for spec 0 and 5 for spec 2
    assert np.array_equal(new_ms_data.spectrum_df["peak_start_idx"].values, [0, 5])
    assert np.array_equal(new_ms_data.spectrum_df["peak_stop_idx"].values, [5, 10])
