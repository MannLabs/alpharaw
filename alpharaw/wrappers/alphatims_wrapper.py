"""Deprecated wrapper for AlphaTimsReader and AlphaTimsWrapper. These classes have been moved to the alphatims package."""

_DEPRECATION_MSG = "has been moved to alphatims."


class AlphaTimsReader:
    def __init__(self, *args, **kwargs):
        raise ImportError(f"AlphaTimsReader {_DEPRECATION_MSG}")


class AlphaTimsWrapper:
    def __init__(self, *args, **kwargs):
        raise ImportError(f"AlphaTimsWrapper {_DEPRECATION_MSG}")
