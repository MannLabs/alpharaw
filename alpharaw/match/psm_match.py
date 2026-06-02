_DEPRECATION_MSG = "has been moved to alphapeptdeep."


class PepSpecMatch:
    def __init__(self, *args, **kwargs):
        raise ImportError(f"PepSpecMatch {_DEPRECATION_MSG}")


class PepSpecMatch_DIA(PepSpecMatch):
    def __init__(self, *args, **kwargs):
        raise ImportError(f"PepSpecMatch_DIA {_DEPRECATION_MSG}")


def match_one_raw_with_numba(*args, **kwargs):
    raise ImportError(f"match_one_raw_with_numba {_DEPRECATION_MSG}")


def load_ms_data(*args, **kwargs):
    raise ImportError(f"load_ms_data {_DEPRECATION_MSG}")


def get_ion_count_scores(*args, **kwargs):
    raise ImportError(f"get_ion_count_scores {_DEPRECATION_MSG}")
