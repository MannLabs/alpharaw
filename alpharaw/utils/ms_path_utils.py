import os

_SPECIAL_MS_EXTS: list = [
    ".ms_data.hdf",  # alphapept
    ".raw.hdf",  # alpharaw
    ".raw.hdf5",  # alpharaw
    ".tims.hdf",  # alphatims (from alpharaw)
    ".tims.hdf5",  # alphatims (from alpharaw)
    ".d.hdf",  # alphatims (from alpharaw)
    ".d.hdf5",  # alphatims (from alpharaw)
    ".atms.hdf",  # alphatims (from alpharaw)
    ".atms.hdf5",  # alphatims (from alpharaw)
    "_hcdft.mgf",  # p
]


def get_raw_name(ms_file: str) -> str:
    """
    Get `raw_name` (base name of RAW data file) from the MS file path
    by removing the extensions defined in :data:`_SPECIAL_MS_EXTS`.

    Parameters
    ----------
    ms_file : str
        The absolute or relative path of the RAW file.

    Returns
    -------
    str
        The `raw_name` without extension.

    Examples
    --------
    >>> get_raw_name("/MS/files/your_raw_name.raw")
    'your_raw_name'

    """
    raw_name = os.path.basename(ms_file)
    lower_name = raw_name.lower()
    for _ext in _SPECIAL_MS_EXTS:
        if lower_name.endswith(_ext.lower()):
            raw_name = raw_name[: -len(_ext)]
            break
    if len(raw_name) == len(lower_name):
        raw_name = os.path.splitext(raw_name)[0]
    return raw_name


def parse_ms_files_to_dict(
    ms_file_list: list,
) -> dict:
    """
    Parse spectrum file paths into a dict:
        "/Users/xxx/raw_name1.raw" -> {"raw_name1":"/Users/xxx/raw_name1.raw"}

    Parameters
    ----------
    spectrum_file_list : list
        File path list

    special_ms_exts : list
        Special MS file extensions that contain more than one dots.

    Returns
    -------
    dict
        {"raw_name1" : "/Users/xxx/raw_name1.raw", ...}
    """

    ms_file_dict = {}
    for ms_file in ms_file_list:
        raw_name = get_raw_name(ms_file, _SPECIAL_MS_EXTS)
        ms_file_dict[raw_name] = ms_file
    return ms_file_dict
