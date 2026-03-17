# TODO to be removed

_DEPRECATION_MSG = "has been moved to alphaviz and will be removed from alpharaw in a future version."


def load_ms_data_tims(*args, **kwargs):
    raise DeprecationWarning(
        f"load_ms_data_tims {_DEPRECATION_MSG}"
    )


class PepSpecMatch_AlphaTims:
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            f"PepSpecMatch_AlphaTims {_DEPRECATION_MSG}"
        )
