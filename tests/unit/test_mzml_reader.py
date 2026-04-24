import numpy as np

from alpharaw.mzml import parse_mzml_entry


def make_selected_ion(selected_ion_mz: float = 400.0, charge_state: int = 2) -> dict:
    """Create a minimal selected-ion payload for mzML-like MS2 entries."""
    return {
        "selected ion m/z": float(selected_ion_mz),
        "charge state": int(charge_state),
    }


def make_isolation_window(
    target_mz: float = 400.0,
    lower_offset: float = 1.5,
    upper_offset: float = 1.5,
) -> dict:
    """Create a minimal isolation-window payload for mzML-like MS2 entries."""
    return {
        "isolation window target m/z": float(target_mz),
        "isolation window lower offset": float(lower_offset),
        "isolation window upper offset": float(upper_offset),
    }


def make_ms1_entry(
    rt: float = 1.0,
    mz_array: np.ndarray | None = None,
    intensity_array: np.ndarray | None = None,
) -> dict:
    """Create a small mzML-like MS1 spectrum dictionary."""
    mz_array = (
        np.array([100.0, 200.0, 300.0], dtype=np.float32)
        if mz_array is None
        else mz_array
    )
    intensity_array = (
        np.array([10.0, 20.0, 30.0], dtype=np.float32)
        if intensity_array is None
        else intensity_array
    )

    return {
        "scanList": {"scan": [{"scan start time": float(rt)}]},
        "m/z array": mz_array,
        "intensity array": intensity_array,
        "ms level": 1,
    }


def make_ms2_entry(
    rt: float = 2.0,
    precursor_mz: float = 400.0,
    charge_state: int = 2,
    isolation_lower_offset: float = 1.5,
    isolation_upper_offset: float = 1.5,
    filter_string: str | None = "FTMS + c NSI Full ms2 400.00@hcd27.00",
    mz_array: np.ndarray | None = None,
    intensity_array: np.ndarray | None = None,
    include_isolation_window: bool = True,
) -> dict:
    """Create a small mzML-like MS2 spectrum dictionary."""
    mz_array = (
        np.array([300.0, 400.0, 500.0], dtype=np.float32)
        if mz_array is None
        else mz_array
    )
    intensity_array = (
        np.array([15.0, 25.0, 35.0], dtype=np.float32)
        if intensity_array is None
        else intensity_array
    )

    precursor = {
        "selectedIonList": {
            "selectedIon": [
                make_selected_ion(
                    selected_ion_mz=precursor_mz,
                    charge_state=charge_state,
                )
            ]
        }
    }
    if include_isolation_window:
        precursor["isolationWindow"] = make_isolation_window(
            target_mz=precursor_mz,
            lower_offset=isolation_lower_offset,
            upper_offset=isolation_upper_offset,
        )

    scan_entry = {"scan start time": float(rt)}
    if filter_string is not None:
        scan_entry["filter string"] = filter_string

    return {
        "scanList": {"scan": [scan_entry]},
        "m/z array": mz_array,
        "intensity array": intensity_array,
        "ms level": 2,
        "precursorList": {"precursor": [precursor]},
    }


class FakeMzMLReader:
    """Simple iterable that mimics a pyteomics mzML reader object."""

    def __init__(self, entries: list[dict]):
        self._entries = list(entries)
        self._index = 0
        self.closed = False

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._index >= len(self._entries):
            raise StopIteration
        entry = self._entries[self._index]
        self._index += 1
        return entry

    def close(self) -> None:
        self.closed = True


def parse_entry(entry: dict) -> tuple:
    """Thin helper around parse_mzml_entry for upcoming behaviour tests."""
    return parse_mzml_entry(entry)


def test_parse_ms1_entry_without_precursor_fields():
    mz_array = np.array([111.1, 222.2, 333.3], dtype=np.float32)
    intensity_array = np.array([10.0, 0.0, 55.5], dtype=np.float32)
    entry = make_ms1_entry(rt=3.25, mz_array=mz_array, intensity_array=intensity_array)

    (
        rt,
        precursor_mz,
        isolation_lower_mz,
        isolation_upper_mz,
        ms_level,
        nce,
        precursor_charge,
        parsed_mz_array,
        parsed_intensity_array,
    ) = parse_entry(entry)

    assert ms_level == 1
    assert precursor_mz == -1.0
    assert precursor_charge == 0
    assert isolation_lower_mz == -1.0
    assert isolation_upper_mz == -1.0
    assert rt == 3.25
    assert nce == 0.0
    np.testing.assert_array_equal(parsed_mz_array, mz_array)
    np.testing.assert_array_equal(parsed_intensity_array, intensity_array)


def test_parse_ms2_entry_with_complete_precursor_fields():
    mz_array = np.array([350.0, 450.0, 550.0], dtype=np.float32)
    intensity_array = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    entry = make_ms2_entry(
        rt=5.5,
        precursor_mz=523.27,
        charge_state=3,
        isolation_lower_offset=0.8,
        isolation_upper_offset=1.2,
        filter_string="FTMS + c NSI Full ms2 523.27@hcd29.00",
        mz_array=mz_array,
        intensity_array=intensity_array,
    )

    (
        rt,
        precursor_mz,
        isolation_lower_mz,
        isolation_upper_mz,
        ms_level,
        nce,
        precursor_charge,
        parsed_mz_array,
        parsed_intensity_array,
    ) = parse_entry(entry)

    assert ms_level == 2
    assert rt == 5.5
    assert precursor_mz == 523.27
    assert precursor_charge == 3
    assert isolation_lower_mz == 522.47
    assert isolation_upper_mz == 524.47
    assert nce == 29.0
    np.testing.assert_array_equal(parsed_mz_array, mz_array)
    np.testing.assert_array_equal(parsed_intensity_array, intensity_array)


def test_parse_ms2_entry_parses_hcd_nce_from_filter_string():
    entry = make_ms2_entry(
        filter_string="FTMS + p NSI Full ms2 500.0000@hcd27.00 [100.0000-1000.0000]"
    )

    (_, _, _, _, _, nce, _, _, _) = parse_entry(entry)

    assert nce == 27.0


def test_parse_ms2_entry_parses_cid_nce_from_filter_string():
    entry = make_ms2_entry(
        filter_string="FTMS + p NSI Full ms2 500.0000@cid35.00 [100.0000-1000.0000]"
    )

    result = parse_mzml_entry(entry)

    assert result[5] == 35.0


def test_parse_ms2_entry_missing_filter_string_returns_nan_nce():
    entry = make_ms2_entry(filter_string=None)

    (_, _, _, _, _, nce, _, _, _) = parse_entry(entry)

    assert np.isnan(nce)


def test_parse_ms2_entry_missing_charge_state_defaults_to_zero():
    entry = make_ms2_entry(charge_state=2)
    entry["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0].pop(
        "charge state"
    )

    (_, _, _, _, _, _, precursor_charge, _, _) = parse_entry(entry)

    assert precursor_charge == 0


def test_parse_ms2_entry_invalid_charge_state_defaults_to_zero():
    entry = make_ms2_entry(charge_state=2)
    entry["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0][
        "charge state"
    ] = "not-a-charge"

    (_, _, _, _, _, _, precursor_charge, _, _) = parse_entry(entry)

    assert precursor_charge == 0


def test_parse_ms2_entry_missing_precursor_list_uses_safe_defaults():
    entry = make_ms2_entry()
    entry.pop("precursorList")

    (
        _,
        precursor_mz,
        isolation_lower_mz,
        isolation_upper_mz,
        _,
        _,
        precursor_charge,
        _,
        _,
    ) = parse_entry(entry)

    assert precursor_mz == -1.0
    assert precursor_charge == 0
    assert isolation_lower_mz == -1.0
    assert isolation_upper_mz == -1.0


def test_parse_ms2_entry_missing_selected_ion_uses_safe_defaults():
    entry = make_ms2_entry()
    entry["precursorList"]["precursor"][0]["selectedIonList"].pop("selectedIon")

    (
        _,
        precursor_mz,
        isolation_lower_mz,
        isolation_upper_mz,
        _,
        _,
        precursor_charge,
        _,
        _,
    ) = parse_entry(entry)

    assert precursor_mz == -1.0
    assert precursor_charge == 0
    assert isolation_lower_mz == -1.0
    assert isolation_upper_mz == -1.0


def test_parse_ms2_entry_missing_selected_ion_mz_uses_safe_defaults():
    entry = make_ms2_entry()
    entry["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0].pop(
        "selected ion m/z"
    )

    (_, precursor_mz, isolation_lower_mz, isolation_upper_mz, _, _, _, _, _) = parse_entry(
        entry
    )

    assert precursor_mz == -1.0
    assert isolation_lower_mz == -1.0
    assert isolation_upper_mz == -1.0
    assert isolation_lower_mz != -2.5
    assert isolation_upper_mz != 0.5
