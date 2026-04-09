"""
Test suite for centroiding algorithms.

This module provides:
1. Synthetic profile data generation with known ground truth
2. Metrics for evaluating centroiding quality (mass accuracy, intensity, isotope spacing)
3. Real-world validation using extracted profile data from SCIEX WIFF files
"""

import numpy as np
import pytest
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from alpharaw.utils.centroiding import centroid_local_maxima, naive_centroid


# =============================================================================
# Test Data Structures
# =============================================================================


@dataclass
class SyntheticPeak:
    """Represents a synthetic peak with known ground truth."""

    mz: float  # True m/z position
    intensity: float  # True total intensity (area)
    sigma: float = 0.002  # Peak width - default matches SCIEX WIFF data (~9 ppm FWHM)


# Realistic profile parameters based on SCIEX WIFF data analysis
SCIEX_PROFILE_PARAMS = {
    "sigma": 0.002,  # ~9 ppm FWHM at m/z 500
    "mz_spacing": 0.005,  # ~10 ppm spacing between profile points
    "centroiding_ppm": 20.0,  # Default PPM for SCIEX data
}

# Wider peaks (for testing algorithm limits)
WIDE_PEAK_PARAMS = {
    "sigma": 0.01,  # ~47 ppm FWHM at m/z 500
    "mz_spacing": 0.004,
    "centroiding_ppm": 100.0,  # Need larger PPM for wider peaks
}


@dataclass
class CentroidingResult:
    """Result from a centroiding algorithm."""

    mz: np.ndarray
    intensity: np.ndarray


@dataclass
class CentroidingMetrics:
    """Metrics for evaluating centroiding quality."""

    mass_accuracy_ppm: float  # Mean absolute mass error in ppm
    mass_accuracy_da: float  # Mean absolute mass error in Da
    intensity_accuracy_percent: float  # Mean intensity error as percentage
    n_peaks_detected: int  # Number of peaks found
    n_peaks_expected: int  # Number of peaks in ground truth
    peak_detection_rate: float  # Fraction of true peaks detected
    false_positive_rate: float  # Fraction of detected peaks that are false positives
    isotope_spacing_error_ppm: Optional[float] = None  # Error in C13 isotope spacing


# =============================================================================
# Synthetic Data Generation
# =============================================================================


def generate_gaussian_profile(
    peak: SyntheticPeak,
    mz_spacing: float = 0.004,  # Typical profile mode spacing
    n_sigma: float = 4.0,  # How many sigma to extend the peak
    min_intensity_fraction: float = 0.001,  # Filter out points below this fraction of max
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Gaussian profile for a single peak.

    Parameters
    ----------
    peak : SyntheticPeak
        The peak parameters (mz, intensity, sigma)
    mz_spacing : float
        Spacing between consecutive m/z points
    n_sigma : float
        Number of standard deviations to extend the peak
    min_intensity_fraction : float
        Filter out points with intensity below this fraction of peak max

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (mz_array, intensity_array)
    """
    half_width = n_sigma * peak.sigma
    n_points = int(2 * half_width / mz_spacing) + 1

    mz_array = np.linspace(peak.mz - half_width, peak.mz + half_width, n_points)

    # Gaussian: I = I0 * exp(-(mz - mz0)^2 / (2 * sigma^2))
    # Scale so that the sum approximates the total intensity
    intensity_array = np.exp(-((mz_array - peak.mz) ** 2) / (2 * peak.sigma**2))
    # Normalize so sum equals total intensity
    intensity_array = intensity_array * peak.intensity / intensity_array.sum()

    # Filter out very low intensity points at the tails
    threshold = intensity_array.max() * min_intensity_fraction
    mask = intensity_array >= threshold
    mz_array = mz_array[mask]
    intensity_array = intensity_array[mask]

    return mz_array.astype(np.float64), intensity_array.astype(np.float64)


def generate_synthetic_spectrum(
    peaks: List[SyntheticPeak],
    mz_spacing: float = 0.004,
    noise_level: float = 0.0,
    baseline: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic profile spectrum from a list of peaks.

    Parameters
    ----------
    peaks : List[SyntheticPeak]
        List of peaks to include in the spectrum
    mz_spacing : float
        Spacing between consecutive m/z points
    noise_level : float
        Standard deviation of Gaussian noise to add (as fraction of max intensity)
    baseline : float
        Constant baseline to add
    seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (mz_array, intensity_array) sorted by m/z
    """
    if seed is not None:
        np.random.seed(seed)

    all_mz = []
    all_int = []

    for peak in peaks:
        mz, intensity = generate_gaussian_profile(peak, mz_spacing)
        all_mz.extend(mz)
        all_int.extend(intensity)

    # Sort by m/z
    mz_array = np.array(all_mz)
    int_array = np.array(all_int)
    sort_idx = np.argsort(mz_array)
    mz_array = mz_array[sort_idx]
    int_array = int_array[sort_idx]

    # Add noise
    if noise_level > 0:
        max_int = int_array.max() if len(int_array) > 0 else 1.0
        noise = np.random.normal(0, noise_level * max_int, len(int_array))
        int_array = np.maximum(int_array + noise, 0)

    # Add baseline
    if baseline > 0:
        int_array = int_array + baseline

    return mz_array, int_array


def generate_isotope_pattern(
    mono_mz: float,
    charge: int = 1,
    mono_intensity: float = 10000.0,
    n_isotopes: int = 4,
    sigma: float = 0.01,
    isotope_ratios: Optional[List[float]] = None,
) -> List[SyntheticPeak]:
    """
    Generate an isotope pattern (M0, M+1, M+2, ...).

    Parameters
    ----------
    mono_mz : float
        Monoisotopic m/z
    charge : int
        Charge state
    mono_intensity : float
        Intensity of the monoisotopic peak
    n_isotopes : int
        Number of isotope peaks to generate
    sigma : float
        Peak width (standard deviation)
    isotope_ratios : Optional[List[float]]
        Relative intensities of isotopes. If None, uses a simple decay pattern.

    Returns
    -------
    List[SyntheticPeak]
        List of peaks representing the isotope pattern
    """
    C13_SPACING = 1.003355  # Da per charge unit

    if isotope_ratios is None:
        # Simple exponential decay pattern (approximation)
        isotope_ratios = [1.0, 0.5, 0.2, 0.08, 0.03][:n_isotopes]

    peaks = []
    for i, ratio in enumerate(isotope_ratios[:n_isotopes]):
        mz = mono_mz + (i * C13_SPACING / charge)
        intensity = mono_intensity * ratio
        peaks.append(SyntheticPeak(mz=mz, intensity=intensity, sigma=sigma))

    return peaks


# =============================================================================
# Evaluation Metrics
# =============================================================================


def match_peaks(
    detected_mz: np.ndarray,
    detected_int: np.ndarray,
    true_peaks: List[SyntheticPeak],
    tolerance_ppm: float = 50.0,
) -> Dict:
    """
    Match detected peaks to ground truth peaks.

    Returns
    -------
    Dict with keys:
        - 'matched': List of (true_peak, detected_mz, detected_int) tuples
        - 'unmatched_true': List of SyntheticPeak not detected
        - 'unmatched_detected': List of (mz, int) tuples that are false positives
    """
    matched = []
    unmatched_true = []
    detected_used = set()

    for true_peak in true_peaks:
        best_match_idx = None
        best_match_error = float("inf")

        for i, (det_mz, det_int) in enumerate(zip(detected_mz, detected_int)):
            if i in detected_used:
                continue

            error_ppm = abs(det_mz - true_peak.mz) / true_peak.mz * 1e6
            if error_ppm < tolerance_ppm and error_ppm < best_match_error:
                best_match_idx = i
                best_match_error = error_ppm

        if best_match_idx is not None:
            matched.append(
                (true_peak, detected_mz[best_match_idx], detected_int[best_match_idx])
            )
            detected_used.add(best_match_idx)
        else:
            unmatched_true.append(true_peak)

    unmatched_detected = [
        (detected_mz[i], detected_int[i])
        for i in range(len(detected_mz))
        if i not in detected_used
    ]

    return {
        "matched": matched,
        "unmatched_true": unmatched_true,
        "unmatched_detected": unmatched_detected,
    }


def calculate_metrics(
    detected_mz: np.ndarray,
    detected_int: np.ndarray,
    true_peaks: List[SyntheticPeak],
    tolerance_ppm: float = 50.0,
) -> CentroidingMetrics:
    """
    Calculate comprehensive metrics for centroiding quality.

    Parameters
    ----------
    detected_mz : np.ndarray
        Detected m/z values from centroiding
    detected_int : np.ndarray
        Detected intensities from centroiding
    true_peaks : List[SyntheticPeak]
        Ground truth peaks
    tolerance_ppm : float
        Tolerance for matching peaks

    Returns
    -------
    CentroidingMetrics
        Comprehensive metrics
    """
    match_result = match_peaks(detected_mz, detected_int, true_peaks, tolerance_ppm)

    n_matched = len(match_result["matched"])
    n_true = len(true_peaks)
    n_detected = len(detected_mz)

    # Mass accuracy
    if n_matched > 0:
        mass_errors_ppm = []
        mass_errors_da = []
        intensity_errors = []

        for true_peak, det_mz, det_int in match_result["matched"]:
            error_ppm = (det_mz - true_peak.mz) / true_peak.mz * 1e6
            error_da = det_mz - true_peak.mz
            int_error = abs(det_int - true_peak.intensity) / true_peak.intensity * 100

            mass_errors_ppm.append(abs(error_ppm))
            mass_errors_da.append(abs(error_da))
            intensity_errors.append(int_error)

        mass_accuracy_ppm = np.mean(mass_errors_ppm)
        mass_accuracy_da = np.mean(mass_errors_da)
        intensity_accuracy = np.mean(intensity_errors)
    else:
        mass_accuracy_ppm = float("inf")
        mass_accuracy_da = float("inf")
        intensity_accuracy = float("inf")

    # Detection rates
    peak_detection_rate = n_matched / n_true if n_true > 0 else 0.0
    false_positive_rate = (
        len(match_result["unmatched_detected"]) / n_detected if n_detected > 0 else 0.0
    )

    return CentroidingMetrics(
        mass_accuracy_ppm=mass_accuracy_ppm,
        mass_accuracy_da=mass_accuracy_da,
        intensity_accuracy_percent=intensity_accuracy,
        n_peaks_detected=n_detected,
        n_peaks_expected=n_true,
        peak_detection_rate=peak_detection_rate,
        false_positive_rate=false_positive_rate,
    )


def calculate_isotope_spacing_error(
    detected_mz: np.ndarray,
    expected_spacing: float = 1.003355,
    tolerance_da: float = 0.1,
) -> Optional[float]:
    """
    Calculate error in isotope spacing detection.

    Finds pairs of peaks that are approximately 1 Da apart and calculates
    how well they match the expected C13 isotope spacing.

    Returns
    -------
    Optional[float]
        Mean isotope spacing error in ppm, or None if no isotope pairs found
    """
    if len(detected_mz) < 2:
        return None

    spacing_errors = []
    for i in range(len(detected_mz) - 1):
        for j in range(i + 1, len(detected_mz)):
            spacing = detected_mz[j] - detected_mz[i]
            # Check if this looks like an isotope pair
            if abs(spacing - expected_spacing) < tolerance_da:
                error_ppm = (spacing - expected_spacing) / detected_mz[i] * 1e6
                spacing_errors.append(abs(error_ppm))
            if spacing > expected_spacing + tolerance_da:
                break

    return np.mean(spacing_errors) if spacing_errors else None


# =============================================================================
# Tests for Synthetic Data
# =============================================================================


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation utilities."""

    def test_generate_single_gaussian(self):
        """Test that a single Gaussian peak is generated correctly."""
        peak = SyntheticPeak(mz=500.0, intensity=10000.0, sigma=0.01)
        mz, intensity = generate_gaussian_profile(peak)

        # Check that peak is centered
        max_idx = np.argmax(intensity)
        assert abs(mz[max_idx] - 500.0) < 0.001

        # Check that total intensity is approximately correct
        # Allow 1% tolerance since tail filtering removes a small fraction
        assert abs(intensity.sum() - 10000.0) / 10000.0 < 0.01

        # Check that it's symmetric
        n = len(mz)
        left_half = intensity[: n // 2]
        right_half = intensity[n // 2 + 1 :][::-1]
        min_len = min(len(left_half), len(right_half))
        np.testing.assert_array_almost_equal(
            left_half[:min_len], right_half[:min_len], decimal=5
        )

    def test_generate_isotope_pattern(self):
        """Test that isotope pattern has correct spacing."""
        peaks = generate_isotope_pattern(mono_mz=500.0, charge=1, n_isotopes=3)

        assert len(peaks) == 3

        # Check spacing
        spacing_01 = peaks[1].mz - peaks[0].mz
        spacing_12 = peaks[2].mz - peaks[1].mz

        assert abs(spacing_01 - 1.003355) < 0.0001
        assert abs(spacing_12 - 1.003355) < 0.0001

    def test_generate_isotope_pattern_charged(self):
        """Test isotope pattern for doubly charged ion."""
        peaks = generate_isotope_pattern(mono_mz=500.0, charge=2, n_isotopes=3)

        spacing_01 = peaks[1].mz - peaks[0].mz
        expected_spacing = 1.003355 / 2

        assert abs(spacing_01 - expected_spacing) < 0.0001

    def test_generate_spectrum_multiple_peaks(self):
        """Test spectrum generation with multiple peaks."""
        peaks = [
            SyntheticPeak(mz=400.0, intensity=5000.0),
            SyntheticPeak(mz=500.0, intensity=10000.0),
            SyntheticPeak(mz=600.0, intensity=7500.0),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks)

        # Check that m/z is sorted
        assert np.all(np.diff(mz) > 0)

        # Check that we have data around each peak
        assert np.any((mz > 399) & (mz < 401))
        assert np.any((mz > 499) & (mz < 501))
        assert np.any((mz > 599) & (mz < 601))


class TestNaiveCentroiding:
    """Tests for the naive centroiding algorithm using SCIEX-realistic parameters."""

    # Use SCIEX-realistic parameters by default
    SIGMA = SCIEX_PROFILE_PARAMS["sigma"]  # 0.002 Da
    MZ_SPACING = SCIEX_PROFILE_PARAMS["mz_spacing"]  # 0.005 Da
    PPM = SCIEX_PROFILE_PARAMS["centroiding_ppm"]  # 20.0

    def test_single_peak_mass_accuracy(self):
        """Test that a single peak is centroided with good mass accuracy."""
        true_mz = 500.0
        peak = SyntheticPeak(mz=true_mz, intensity=10000.0, sigma=self.SIGMA)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        # Should detect exactly one peak
        assert len(cent_mz) == 1, f"Expected 1 peak, got {len(cent_mz)}"

        # Mass accuracy should be very good for single peak
        error_ppm = abs(cent_mz[0] - true_mz) / true_mz * 1e6
        assert error_ppm < 5.0, f"Mass error {error_ppm:.2f} ppm exceeds 5 ppm"

    def test_single_peak_intensity_accuracy(self):
        """Test that peak intensity is correctly summed."""
        true_intensity = 10000.0
        peak = SyntheticPeak(mz=500.0, intensity=true_intensity, sigma=self.SIGMA)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        # Should detect one peak
        assert len(cent_int) == 1, f"Expected 1 peak, got {len(cent_int)}"

        # Intensity should be sum of profile points
        intensity_error = abs(cent_int[0] - true_intensity) / true_intensity * 100
        assert intensity_error < 5.0, f"Intensity error {intensity_error:.1f}% exceeds 5%"

    def test_two_well_separated_peaks(self):
        """Test centroiding of two well-separated peaks."""
        peaks = [
            SyntheticPeak(mz=400.0, intensity=5000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=600.0, intensity=10000.0, sigma=self.SIGMA),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        assert metrics.n_peaks_detected == 2, f"Expected 2 peaks, got {metrics.n_peaks_detected}"
        assert metrics.peak_detection_rate == 1.0
        assert metrics.mass_accuracy_ppm < 5.0

    def test_isotope_pattern_detection(self):
        """Test that isotope pattern is correctly detected."""
        peaks = generate_isotope_pattern(
            mono_mz=500.0, mono_intensity=10000.0, sigma=self.SIGMA
        )
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        # Should detect all isotope peaks
        assert metrics.peak_detection_rate >= 0.75, f"Detection rate {metrics.peak_detection_rate} < 0.75"

        # Check isotope spacing
        spacing_error = calculate_isotope_spacing_error(cent_mz)
        if spacing_error is not None:
            assert spacing_error < 20.0, f"Isotope spacing error {spacing_error:.1f} ppm"

    def test_isotope_pattern_mass_accuracy(self):
        """Test mass accuracy across an isotope pattern."""
        peaks = generate_isotope_pattern(
            mono_mz=500.0, mono_intensity=10000.0, sigma=self.SIGMA
        )
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        # Mass accuracy should be good
        assert metrics.mass_accuracy_ppm < 10.0

    def test_overlapping_peaks_resolvable(self):
        """Test centroiding of well-separated peaks that should be resolved."""
        # Two peaks 0.1 Da apart (200 ppm at m/z 500) - should be easily resolved
        peaks = [
            SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=500.1, intensity=8000.0, sigma=self.SIGMA),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        # Should detect both peaks (0.1 Da >> 20 ppm tolerance)
        assert len(cent_mz) == 2, f"Should detect 2 peaks, got {len(cent_mz)}"

    def test_very_close_peaks_merge(self):
        """Test that very close peaks merge appropriately."""
        # Two peaks 0.005 Da apart (10 ppm at m/z 500) - should merge with 20 ppm tolerance
        peaks = [
            SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=500.005, intensity=8000.0, sigma=self.SIGMA),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        # These peaks should merge (or be very close)
        # The result depends on algorithm, but total intensity should be preserved
        total_expected = 10000.0 + 8000.0
        total_detected = cent_int.sum()
        assert abs(total_detected - total_expected) / total_expected < 0.1

    def test_noise_robustness(self):
        """Test that centroiding handles noise reasonably."""
        peaks = [SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA)]
        mz, intensity = generate_synthetic_spectrum(
            peaks, mz_spacing=self.MZ_SPACING, noise_level=0.01, seed=42
        )

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        metrics = calculate_metrics(cent_mz, cent_int, peaks, tolerance_ppm=50.0)

        # Should still detect the main peak
        assert metrics.peak_detection_rate == 1.0
        # Mass accuracy may be slightly worse with noise
        assert metrics.mass_accuracy_ppm < 20.0

    def test_low_intensity_peak(self):
        """Test detection of low intensity peaks."""
        peaks = [SyntheticPeak(mz=500.0, intensity=100.0, sigma=self.SIGMA)]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        # Should still detect the peak
        assert len(cent_mz) >= 1
        metrics = calculate_metrics(cent_mz, cent_int, peaks)
        assert metrics.peak_detection_rate == 1.0

    def test_high_mz_mass_accuracy(self):
        """Test mass accuracy at high m/z values."""
        # At m/z 2000, use proportionally wider peak (same ppm width)
        sigma_high_mz = 0.008  # ~9 ppm FWHM at m/z 2000
        peaks = [SyntheticPeak(mz=2000.0, intensity=10000.0, sigma=sigma_high_mz)]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=0.02)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)
        assert metrics.mass_accuracy_ppm < 10.0

    def test_empty_spectrum(self):
        """Test handling of empty spectrum."""
        mz = np.array([], dtype=np.float64)
        intensity = np.array([], dtype=np.float64)

        cent_mz, cent_int = naive_centroid(mz, intensity)

        assert len(cent_mz) == 0
        assert len(cent_int) == 0


class TestNaiveCentroidingWidePeaks:
    """Tests for naive centroiding with wider peaks (requires higher PPM tolerance)."""

    SIGMA = WIDE_PEAK_PARAMS["sigma"]  # 0.01 Da
    MZ_SPACING = WIDE_PEAK_PARAMS["mz_spacing"]  # 0.004 Da
    PPM = WIDE_PEAK_PARAMS["centroiding_ppm"]  # 100.0

    def test_wide_single_peak(self):
        """Test that wider peaks require higher PPM tolerance."""
        peak = SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=self.PPM)

        # With correct PPM, should get one peak
        assert len(cent_mz) == 1
        error_ppm = abs(cent_mz[0] - 500.0) / 500.0 * 1e6
        assert error_ppm < 5.0

    def test_wide_peak_fails_with_low_ppm(self):
        """Demonstrate that wide peaks need higher PPM tolerance."""
        peak = SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=self.MZ_SPACING)

        # Using too-small PPM should fragment the peak
        cent_mz, cent_int = naive_centroid(mz, intensity, centroiding_ppm=20.0)

        # This demonstrates the limitation - peak gets fragmented
        assert len(cent_mz) > 1, "Expected peak fragmentation with low PPM"


class TestLocalMaximaCentroiding:
    """Tests for the local maxima centroiding algorithm."""

    # Use SCIEX-realistic parameters by default
    SIGMA = SCIEX_PROFILE_PARAMS["sigma"]  # 0.002 Da
    MZ_SPACING = SCIEX_PROFILE_PARAMS["mz_spacing"]  # 0.005 Da

    def test_single_peak_mass_accuracy(self):
        """Test that a single peak is centroided with good mass accuracy."""
        true_mz = 500.0
        peak = SyntheticPeak(mz=true_mz, intensity=10000.0, sigma=self.SIGMA)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        assert len(cent_mz) == 1, f"Expected 1 peak, got {len(cent_mz)}"

        error_ppm = abs(cent_mz[0] - true_mz) / true_mz * 1e6
        assert error_ppm < 5.0, f"Mass error {error_ppm:.2f} ppm exceeds 5 ppm"

    def test_single_peak_intensity_accuracy(self):
        """Test that peak intensity is correctly summed."""
        true_intensity = 10000.0
        peak = SyntheticPeak(mz=500.0, intensity=true_intensity, sigma=self.SIGMA)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        assert len(cent_int) == 1, f"Expected 1 peak, got {len(cent_int)}"

        intensity_error = abs(cent_int[0] - true_intensity) / true_intensity * 100
        assert intensity_error < 5.0, f"Intensity error {intensity_error:.1f}% exceeds 5%"

    def test_two_well_separated_peaks(self):
        """Test centroiding of two well-separated peaks."""
        peaks = [
            SyntheticPeak(mz=400.0, intensity=5000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=600.0, intensity=10000.0, sigma=self.SIGMA),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        assert metrics.n_peaks_detected == 2, f"Expected 2 peaks, got {metrics.n_peaks_detected}"
        assert metrics.peak_detection_rate == 1.0
        assert metrics.mass_accuracy_ppm < 5.0

    def test_isotope_pattern_detection(self):
        """Test that isotope pattern is correctly detected."""
        peaks = generate_isotope_pattern(
            mono_mz=500.0, mono_intensity=10000.0, sigma=self.SIGMA
        )
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        assert metrics.peak_detection_rate >= 0.75, f"Detection rate {metrics.peak_detection_rate} < 0.75"

        spacing_error = calculate_isotope_spacing_error(cent_mz)
        if spacing_error is not None:
            assert spacing_error < 20.0, f"Isotope spacing error {spacing_error:.1f} ppm"

    def test_isotope_pattern_mass_accuracy(self):
        """Test mass accuracy across an isotope pattern."""
        peaks = generate_isotope_pattern(
            mono_mz=500.0, mono_intensity=10000.0, sigma=self.SIGMA
        )
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        assert metrics.mass_accuracy_ppm < 10.0

    def test_overlapping_peaks_resolvable(self):
        """Test centroiding of well-separated peaks that should be resolved."""
        peaks = [
            SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=500.1, intensity=8000.0, sigma=self.SIGMA),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        assert len(cent_mz) == 2, f"Should detect 2 peaks, got {len(cent_mz)}"

    def test_wide_peak_no_fragmentation(self):
        """CRITICAL: Wide peaks should NOT fragment (unlike naive with low PPM)."""
        # This is the test that naive_centroid fails with 20 ppm
        peak = SyntheticPeak(mz=500.0, intensity=10000.0, sigma=0.01)
        mz, intensity = generate_synthetic_spectrum([peak], mz_spacing=0.004)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        # Should detect exactly 1 peak regardless of peak width
        assert len(cent_mz) == 1, f"Wide peak fragmented into {len(cent_mz)} peaks"

        error_ppm = abs(cent_mz[0] - 500.0) / 500.0 * 1e6
        assert error_ppm < 5.0, f"Mass error {error_ppm:.2f} ppm"

    def test_noise_robustness_with_snr(self):
        """Test that SNR filtering removes noise peaks."""
        peaks = [SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA)]
        mz, intensity = generate_synthetic_spectrum(
            peaks, mz_spacing=self.MZ_SPACING, noise_level=0.01, seed=42
        )

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=3.0)

        metrics = calculate_metrics(cent_mz, cent_int, peaks, tolerance_ppm=50.0)

        assert metrics.peak_detection_rate == 1.0
        assert metrics.mass_accuracy_ppm < 20.0

    def test_low_intensity_peak(self):
        """Test detection of low intensity peaks."""
        peaks = [SyntheticPeak(mz=500.0, intensity=100.0, sigma=self.SIGMA)]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        assert len(cent_mz) >= 1
        metrics = calculate_metrics(cent_mz, cent_int, peaks)
        assert metrics.peak_detection_rate == 1.0

    def test_high_mz_mass_accuracy(self):
        """Test mass accuracy at high m/z values."""
        sigma_high_mz = 0.008
        peaks = [SyntheticPeak(mz=2000.0, intensity=10000.0, sigma=sigma_high_mz)]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=0.02)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        metrics = calculate_metrics(cent_mz, cent_int, peaks)
        assert metrics.mass_accuracy_ppm < 10.0

    def test_empty_spectrum(self):
        """Test handling of empty spectrum."""
        mz = np.array([], dtype=np.float64)
        intensity = np.array([], dtype=np.float64)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity)

        assert len(cent_mz) == 0
        assert len(cent_int) == 0

    def test_single_point(self):
        """Test handling of single-point spectrum."""
        mz = np.array([500.0], dtype=np.float64)
        intensity = np.array([1000.0], dtype=np.float64)

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        assert len(cent_mz) == 1
        assert cent_mz[0] == 500.0
        assert cent_int[0] == 1000.0

    def test_intensity_preservation(self):
        """Test that total intensity is preserved across centroiding."""
        peaks = [
            SyntheticPeak(mz=400.0, intensity=5000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=500.0, intensity=10000.0, sigma=self.SIGMA),
            SyntheticPeak(mz=600.0, intensity=7500.0, sigma=self.SIGMA),
        ]
        mz, intensity = generate_synthetic_spectrum(peaks, mz_spacing=self.MZ_SPACING)

        total_before = intensity.sum()

        cent_mz, cent_int = centroid_local_maxima(mz, intensity, snr_threshold=0.0)

        total_after = cent_int.sum()

        # Total intensity should be preserved (within floating point tolerance)
        assert abs(total_after - total_before) / total_before < 0.01


class TestCentroidingMetrics:
    """Tests for the metrics calculation functions."""

    def test_perfect_detection(self):
        """Test metrics when all peaks are perfectly detected."""
        true_peaks = [
            SyntheticPeak(mz=500.0, intensity=10000.0),
            SyntheticPeak(mz=600.0, intensity=5000.0),
        ]
        detected_mz = np.array([500.0, 600.0])
        detected_int = np.array([10000.0, 5000.0])

        metrics = calculate_metrics(detected_mz, detected_int, true_peaks)

        assert metrics.mass_accuracy_ppm == 0.0
        assert metrics.intensity_accuracy_percent == 0.0
        assert metrics.peak_detection_rate == 1.0
        assert metrics.false_positive_rate == 0.0

    def test_missed_peak(self):
        """Test metrics when a peak is missed."""
        true_peaks = [
            SyntheticPeak(mz=500.0, intensity=10000.0),
            SyntheticPeak(mz=600.0, intensity=5000.0),
        ]
        detected_mz = np.array([500.0])  # Missing second peak
        detected_int = np.array([10000.0])

        metrics = calculate_metrics(detected_mz, detected_int, true_peaks)

        assert metrics.n_peaks_detected == 1
        assert metrics.n_peaks_expected == 2
        assert metrics.peak_detection_rate == 0.5

    def test_false_positive(self):
        """Test metrics when there's a false positive."""
        true_peaks = [SyntheticPeak(mz=500.0, intensity=10000.0)]
        detected_mz = np.array([500.0, 700.0])  # Extra peak at 700
        detected_int = np.array([10000.0, 1000.0])

        metrics = calculate_metrics(detected_mz, detected_int, true_peaks)

        assert metrics.n_peaks_detected == 2
        assert metrics.peak_detection_rate == 1.0
        assert metrics.false_positive_rate == 0.5


# =============================================================================
# Real-World Data Tests (requires WIFF file)
# =============================================================================


class TestRealWorldData:
    """
    Tests using real SCIEX WIFF data.

    These tests are marked with pytest.mark.slow and pytest.mark.real_data.
    They require the test WIFF file to be present.
    """

    WIFF_PATH = "/Users/michaelbaggiolorenz/Desktop/projects/data/alpharaw_development/20260203_Zeno2_Eno0_12p95min_TiHe_SA_H032_E269_G5.wiff"

    @pytest.fixture
    def wiff_data(self):
        """Load the WIFF file for testing."""
        try:
            from alpharaw.sciex import SciexWiffData

            data = SciexWiffData(centroided=False)
            data.load_raw(self.WIFF_PATH)
            return data
        except Exception as e:
            pytest.skip(f"Could not load WIFF file: {e}")

    @pytest.mark.slow
    @pytest.mark.real_data
    def test_wiff_file_loads(self, wiff_data):
        """Test that WIFF file loads correctly."""
        assert len(wiff_data.spectrum_df) > 0
        assert len(wiff_data.peak_df) > 0

    @pytest.mark.slow
    @pytest.mark.real_data
    def test_centroiding_reduces_peak_count(self, wiff_data):
        """Test that centroiding reduces the number of peaks (profile -> centroid)."""
        # Get first MS1 spectrum
        ms1_spectra = wiff_data.spectrum_df[wiff_data.spectrum_df["ms_level"] == 1]
        if len(ms1_spectra) == 0:
            pytest.skip("No MS1 spectra in file")

        spec = ms1_spectra.iloc[0]
        start_idx = int(spec["peak_start_idx"])
        stop_idx = int(spec["peak_stop_idx"])
        mzs = wiff_data.peak_df["mz"].values[start_idx:stop_idx].astype(np.float64)
        ints = wiff_data.peak_df["intensity"].values[start_idx:stop_idx].astype(
            np.float64
        )

        cent_mz, cent_int = naive_centroid(mzs, ints, centroiding_ppm=20.0)

        # Centroiding should reduce peak count significantly for profile data
        reduction_ratio = len(mzs) / len(cent_mz)
        assert reduction_ratio > 1.0, "Centroiding should reduce peak count"

    @pytest.mark.slow
    @pytest.mark.real_data
    def test_isotope_spacing_in_real_data(self, wiff_data):
        """Test that centroiding produces reasonable isotope spacing in real data."""
        # Get an MS1 spectrum with good peaks
        ms1_spectra = wiff_data.spectrum_df[wiff_data.spectrum_df["ms_level"] == 1]
        if len(ms1_spectra) == 0:
            pytest.skip("No MS1 spectra in file")

        # Find spectrum with many peaks
        peak_counts = ms1_spectra["peak_stop_idx"] - ms1_spectra["peak_start_idx"]
        best_idx = peak_counts.idxmax()
        spec = wiff_data.spectrum_df.loc[best_idx]

        start_idx = int(spec["peak_start_idx"])
        stop_idx = int(spec["peak_stop_idx"])
        mzs = wiff_data.peak_df["mz"].values[start_idx:stop_idx].astype(np.float64)
        ints = wiff_data.peak_df["intensity"].values[start_idx:stop_idx].astype(
            np.float64
        )

        cent_mz, cent_int = naive_centroid(mzs, ints, centroiding_ppm=20.0)

        # Check isotope spacing
        spacing_error = calculate_isotope_spacing_error(cent_mz)
        # Real data may have worse spacing due to various factors
        # Just check it's not wildly off
        if spacing_error is not None:
            assert spacing_error < 100.0, f"Isotope spacing error {spacing_error:.1f} ppm"


# =============================================================================
# Benchmark Data Extraction (for creating golden test sets)
# =============================================================================


def extract_test_spectra_from_wiff(
    wiff_path: str,
    n_ms1: int = 5,
    n_ms2: int = 10,
    min_peaks: int = 100,
) -> List[Dict]:
    """
    Extract representative spectra from a WIFF file for benchmarking.

    This function can be used to create a "golden" test set from real data.

    Parameters
    ----------
    wiff_path : str
        Path to the WIFF file
    n_ms1 : int
        Number of MS1 spectra to extract
    n_ms2 : int
        Number of MS2 spectra to extract
    min_peaks : int
        Minimum number of peaks required

    Returns
    -------
    List[Dict]
        List of spectrum dictionaries with 'mz', 'intensity', 'ms_level', 'rt'
    """
    from alpharaw.sciex import SciexWiffData

    data = SciexWiffData(centroided=False)
    data.load_raw(wiff_path)

    spectra = []

    for ms_level, n_spectra in [(1, n_ms1), (2, n_ms2)]:
        level_spectra = data.spectrum_df[data.spectrum_df["ms_level"] == ms_level]
        peak_counts = level_spectra["peak_stop_idx"] - level_spectra["peak_start_idx"]

        # Filter by minimum peaks
        valid_spectra = level_spectra[peak_counts >= min_peaks]

        # Take top N by peak count
        top_spectra = valid_spectra.nlargest(n_spectra, "peak_stop_idx")

        for _, spec in top_spectra.iterrows():
            start_idx = int(spec["peak_start_idx"])
            stop_idx = int(spec["peak_stop_idx"])
            mzs = data.peak_df["mz"].values[start_idx:stop_idx]
            ints = data.peak_df["intensity"].values[start_idx:stop_idx]

            spectra.append(
                {
                    "mz": mzs.copy(),
                    "intensity": ints.copy(),
                    "ms_level": int(spec["ms_level"]),
                    "rt": float(spec["rt"]),
                    "spec_idx": int(spec["spec_idx"]),
                }
            )

    return spectra


# =============================================================================
# Algorithm Comparison Framework
# =============================================================================


@dataclass
class AlgorithmBenchmark:
    """Results from benchmarking a centroiding algorithm."""

    algorithm_name: str
    synthetic_metrics: Dict[str, CentroidingMetrics]  # Test name -> metrics
    real_data_metrics: Optional[Dict[str, Dict]] = None  # Spectrum idx -> metrics


class CentroidingAlgorithm:
    """Base class for centroiding algorithms (for comparison)."""

    name: str = "base"

    def centroid(
        self, mz: np.ndarray, intensity: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply centroiding to profile data."""
        raise NotImplementedError


class NaiveCentroidingAlgorithm(CentroidingAlgorithm):
    """Wrapper for the naive centroiding algorithm."""

    def __init__(self, ppm: float = 20.0):
        self.ppm = ppm
        self.name = f"naive_{int(ppm)}ppm"

    def centroid(
        self, mz: np.ndarray, intensity: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        ppm = kwargs.get("ppm", self.ppm)
        return naive_centroid(mz.astype(np.float64), intensity.astype(np.float64), ppm)


class LocalMaximaCentroidingAlgorithm(CentroidingAlgorithm):
    """Wrapper for the local maxima centroiding algorithm."""

    def __init__(self, snr_threshold: float = 1.0):
        self.snr_threshold = snr_threshold
        self.name = f"local_maxima_snr{snr_threshold}"

    def centroid(
        self, mz: np.ndarray, intensity: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        snr = kwargs.get("snr_threshold", self.snr_threshold)
        return centroid_local_maxima(
            mz.astype(np.float64), intensity.astype(np.float64), snr
        )


def run_synthetic_benchmark(
    algorithm: CentroidingAlgorithm,
    test_cases: Optional[List[Dict]] = None,
) -> Dict[str, CentroidingMetrics]:
    """
    Run synthetic data benchmarks for a centroiding algorithm.

    Parameters
    ----------
    algorithm : CentroidingAlgorithm
        The algorithm to benchmark
    test_cases : Optional[List[Dict]]
        Custom test cases. If None, uses default test suite.

    Returns
    -------
    Dict[str, CentroidingMetrics]
        Test name -> metrics
    """
    if test_cases is None:
        # Default test suite
        test_cases = [
            {
                "name": "single_peak",
                "peaks": [SyntheticPeak(mz=500.0, intensity=10000.0)],
                "description": "Single isolated peak",
            },
            {
                "name": "isotope_pattern",
                "peaks": generate_isotope_pattern(mono_mz=500.0, mono_intensity=10000.0),
                "description": "4-peak isotope pattern",
            },
            {
                "name": "two_peaks_separated",
                "peaks": [
                    SyntheticPeak(mz=400.0, intensity=5000.0),
                    SyntheticPeak(mz=600.0, intensity=10000.0),
                ],
                "description": "Two well-separated peaks",
            },
            {
                "name": "close_peaks_50ppm",
                "peaks": [
                    SyntheticPeak(mz=500.0, intensity=10000.0),
                    SyntheticPeak(mz=500.025, intensity=8000.0),  # 50 ppm apart
                ],
                "description": "Two peaks 50 ppm apart",
            },
            {
                "name": "noisy_peak",
                "peaks": [SyntheticPeak(mz=500.0, intensity=10000.0)],
                "noise_level": 0.02,
                "description": "Single peak with 2% noise",
            },
            {
                "name": "low_intensity",
                "peaks": [SyntheticPeak(mz=500.0, intensity=100.0)],
                "description": "Low intensity peak",
            },
            {
                "name": "high_mz",
                "peaks": [SyntheticPeak(mz=2000.0, intensity=10000.0, sigma=0.008)],
                "mz_spacing": 0.02,
                "description": "Peak at high m/z",
            },
        ]

    results = {}

    for case in test_cases:
        peaks = case["peaks"]
        mz_spacing = case.get("mz_spacing", SCIEX_PROFILE_PARAMS["mz_spacing"])
        noise_level = case.get("noise_level", 0.0)

        mz, intensity = generate_synthetic_spectrum(
            peaks, mz_spacing=mz_spacing, noise_level=noise_level, seed=42
        )

        cent_mz, cent_int = algorithm.centroid(mz, intensity)
        metrics = calculate_metrics(cent_mz, cent_int, peaks)

        results[case["name"]] = metrics

    return results


def run_real_data_benchmark(
    algorithm: CentroidingAlgorithm,
    wiff_path: str,
    n_spectra: int = 10,
) -> Dict[int, Dict]:
    """
    Run real data benchmarks for a centroiding algorithm.

    Parameters
    ----------
    algorithm : CentroidingAlgorithm
        The algorithm to benchmark
    wiff_path : str
        Path to WIFF file
    n_spectra : int
        Number of spectra to test

    Returns
    -------
    Dict[int, Dict]
        Spectrum index -> results dict
    """
    spectra = extract_test_spectra_from_wiff(wiff_path, n_ms1=n_spectra // 2, n_ms2=n_spectra // 2)

    results = {}
    for spec in spectra:
        mz = spec["mz"].astype(np.float64)
        intensity = spec["intensity"].astype(np.float64)

        cent_mz, cent_int = algorithm.centroid(mz, intensity)

        # Calculate reduction ratio
        reduction_ratio = len(mz) / len(cent_mz) if len(cent_mz) > 0 else float("inf")

        # Calculate isotope spacing error
        spacing_error = calculate_isotope_spacing_error(cent_mz)

        results[spec["spec_idx"]] = {
            "ms_level": spec["ms_level"],
            "rt": spec["rt"],
            "n_profile_points": len(mz),
            "n_centroid_peaks": len(cent_mz),
            "reduction_ratio": reduction_ratio,
            "isotope_spacing_error_ppm": spacing_error,
            "total_intensity_profile": intensity.sum(),
            "total_intensity_centroid": cent_int.sum(),
        }

    return results


def compare_algorithms(
    algorithms: List[CentroidingAlgorithm],
    wiff_path: Optional[str] = None,
) -> Dict[str, AlgorithmBenchmark]:
    """
    Compare multiple centroiding algorithms.

    Parameters
    ----------
    algorithms : List[CentroidingAlgorithm]
        Algorithms to compare
    wiff_path : Optional[str]
        Path to WIFF file for real data tests

    Returns
    -------
    Dict[str, AlgorithmBenchmark]
        Algorithm name -> benchmark results
    """
    results = {}

    for alg in algorithms:
        print(f"Benchmarking {alg.name}...")
        synthetic_metrics = run_synthetic_benchmark(alg)

        real_data_metrics = None
        if wiff_path:
            real_data_metrics = run_real_data_benchmark(alg, wiff_path)

        results[alg.name] = AlgorithmBenchmark(
            algorithm_name=alg.name,
            synthetic_metrics=synthetic_metrics,
            real_data_metrics=real_data_metrics,
        )

    return results


def print_benchmark_comparison(benchmarks: Dict[str, AlgorithmBenchmark]):
    """Print a comparison table of algorithm benchmarks."""
    print("\n" + "=" * 80)
    print("CENTROIDING ALGORITHM COMPARISON")
    print("=" * 80)

    # Get all test names
    test_names = set()
    for b in benchmarks.values():
        test_names.update(b.synthetic_metrics.keys())

    # Print synthetic test results
    print("\n--- Synthetic Data Tests ---")
    print(f"{'Test':<25} | ", end="")
    for alg_name in benchmarks.keys():
        print(f"{alg_name:<15} | ", end="")
    print()
    print("-" * (30 + 18 * len(benchmarks)))

    for test in sorted(test_names):
        print(f"{test:<25} | ", end="")
        for alg_name, benchmark in benchmarks.items():
            if test in benchmark.synthetic_metrics:
                m = benchmark.synthetic_metrics[test]
                print(f"{m.mass_accuracy_ppm:>6.2f} ppm     | ", end="")
            else:
                print(f"{'N/A':>15} | ", end="")
        print()

    # Print real data summary if available
    if any(b.real_data_metrics for b in benchmarks.values()):
        print("\n--- Real Data Summary ---")
        header = f"{'Metric':<30} | "
        for alg_name in benchmarks.keys():
            header += f"{alg_name:>15} | "
        print(header)
        print("-" * len(header))

        # Collect metrics for each algorithm
        metrics_by_alg = {}
        for alg_name, benchmark in benchmarks.items():
            if benchmark.real_data_metrics:
                ratios = [r["reduction_ratio"] for r in benchmark.real_data_metrics.values()]
                spacing_errors = [
                    r["isotope_spacing_error_ppm"]
                    for r in benchmark.real_data_metrics.values()
                    if r["isotope_spacing_error_ppm"] is not None
                ]
                int_ratios = [
                    r["total_intensity_centroid"] / r["total_intensity_profile"]
                    for r in benchmark.real_data_metrics.values()
                    if r["total_intensity_profile"] > 0
                ]
                metrics_by_alg[alg_name] = {
                    "reduction_ratio": np.mean(ratios),
                    "isotope_error": np.mean(spacing_errors) if spacing_errors else None,
                    "intensity_ratio": np.mean(int_ratios) if int_ratios else None,
                }

        # Print reduction ratio
        row = f"{'Peak count reduction':<30} | "
        for alg_name in benchmarks.keys():
            if alg_name in metrics_by_alg:
                row += f"{metrics_by_alg[alg_name]['reduction_ratio']:>12.2f}x | "
            else:
                row += f"{'N/A':>15} | "
        print(row)

        # Print intensity ratio
        row = f"{'Intensity preservation':<30} | "
        for alg_name in benchmarks.keys():
            if alg_name in metrics_by_alg and metrics_by_alg[alg_name]["intensity_ratio"]:
                row += f"{metrics_by_alg[alg_name]['intensity_ratio']*100:>12.1f}% | "
            else:
                row += f"{'N/A':>15} | "
        print(row)

        # Print isotope spacing error
        row = f"{'Isotope spacing error (ppm)':<30} | "
        for alg_name in benchmarks.keys():
            if alg_name in metrics_by_alg and metrics_by_alg[alg_name]["isotope_error"]:
                row += f"{metrics_by_alg[alg_name]['isotope_error']:>12.1f}   | "
            else:
                row += f"{'N/A':>15} | "
        print(row)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Run benchmark comparison
        wiff_path = "/Users/michaelbaggiolorenz/Desktop/projects/data/alpharaw_development/20260203_Zeno2_Eno0_12p95min_TiHe_SA_H032_E269_G5.wiff"

        algorithms = [
            NaiveCentroidingAlgorithm(ppm=20.0),
            NaiveCentroidingAlgorithm(ppm=100.0),
            LocalMaximaCentroidingAlgorithm(snr_threshold=0.0),
            LocalMaximaCentroidingAlgorithm(snr_threshold=3.0),
        ]

        benchmarks = compare_algorithms(algorithms, wiff_path)
        print_benchmark_comparison(benchmarks)
    else:
        # Run basic tests
        pytest.main([__file__, "-v", "-m", "not slow"])
