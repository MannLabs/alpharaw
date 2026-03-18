_DEPRECATION_MSG = "has been moved to alphatims."


class XIC_Plot_Tims:
    def __init__(self, *args, **kwargs):
        raise ImportError(f"XIC_Plot_Tims {_DEPRECATION_MSG}")


class XIC_Trace_Tims:
    def __init__(self, *args, **kwargs):
        raise ImportError(f"XIC_Trace_Tims {_DEPRECATION_MSG}")


def get_plotting_slices(*args, **kwargs):
    raise ImportError(f"get_plotting_slices {_DEPRECATION_MSG}")
