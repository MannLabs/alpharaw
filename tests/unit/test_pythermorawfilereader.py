from types import SimpleNamespace

import pytest

from alpharaw.raw_access import pythermorawfilereader


class RawFileWithoutMs:
    """A raw file that opens but holds no MS device, as an aborted acquisition does."""

    IsOpen = True

    def GetInstrumentCountOfType(self, device):
        return 0

    def SelectInstrument(self, device, index):
        raise AssertionError(
            "SelectInstrument must not be reached without an MS device"
        )


def test_a_raw_file_without_ms_data_raises_a_clear_error(monkeypatch, tmp_path):
    adapter = SimpleNamespace(FileFactory=lambda path: RawFileWithoutMs())
    thermo_fisher = SimpleNamespace(
        CommonCore=SimpleNamespace(
            RawFileReader=SimpleNamespace(RawFileReaderAdapter=adapter)
        )
    )
    monkeypatch.setattr(pythermorawfilereader, "HAS_DOTNET", True)
    monkeypatch.setattr(
        pythermorawfilereader, "ThermoFisher", thermo_fisher, raising=False
    )
    monkeypatch.setattr(
        pythermorawfilereader, "Device", SimpleNamespace(MS="MS"), raising=False
    )

    with pytest.raises(OSError, match="contains no mass spectrometer data"):
        pythermorawfilereader.RawFileReader(str(tmp_path / "aborted.raw"))
