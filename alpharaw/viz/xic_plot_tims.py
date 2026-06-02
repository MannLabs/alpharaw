_DEPRECATION_MSG = (
    "has been moved to alphaviz and will be removed from alpharaw in a future version."
)


class XIC_Plot_Tims:
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(f"XIC_Plot_Tims {_DEPRECATION_MSG}")


class XIC_Trace_Tims:
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(f"XIC_Trace_Tims {_DEPRECATION_MSG}")


def get_plotting_slices(*args, **kwargs):
    raise DeprecationWarning(f"get_plotting_slices {_DEPRECATION_MSG}")
