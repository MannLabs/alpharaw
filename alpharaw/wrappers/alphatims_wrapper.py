"""Deprecated wrapper for AlphaTimsReader and AlphaTimsWrapper. These classes have been moved to the alphatims package."""

_DEPRECATION_MSG = (
    "has been moved to alphatims and will be removed from alpharaw in a future version."
)


class AlphaTimsReader:
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(f"AlphaTimsReader {_DEPRECATION_MSG}")


class AlphaTimsWrapper:
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(f"AlphaTimsWrapper {_DEPRECATION_MSG}")
