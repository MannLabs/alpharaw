_DEPRECATION_MSG = "has been moved to alphapeptdeep and will be removed from alpharaw in a future version."


class PepSpecMatch:
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(f"PepSpecMatch {_DEPRECATION_MSG}")


class PepSpecMatch_DIA(PepSpecMatch):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(f"PepSpecMatch_DIA {_DEPRECATION_MSG}")


def match_one_raw_with_numba(*args, **kwargs):
    raise DeprecationWarning(f"match_one_raw_with_numba {_DEPRECATION_MSG}")


def load_ms_data(*args, **kwargs):
    raise DeprecationWarning(f"load_ms_data {_DEPRECATION_MSG}")


def get_best_matched_intens(*args, **kwargs):
    raise DeprecationWarning(f"get_best_matched_intens {_DEPRECATION_MSG}")


def get_ion_count_scores(*args, **kwargs):
    raise DeprecationWarning(f"get_ion_count_scores {_DEPRECATION_MSG}")
