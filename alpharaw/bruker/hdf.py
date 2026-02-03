"""HDF5 serialization utilities for TimsTOF data persistence."""

import numpy as np

from alpharaw.utils.pjit import progress_callback


def create_hdf_group_from_dict(
    hdf_group,
    data_dict: dict,
    *,
    overwrite: bool = False,
    compress: bool = False,
    recursed: bool = False,
    chunked: bool = False
) -> None:
    """Save a dict to an open hdf group.

    Parameters
    ----------
    hdf_group : h5py.File.group
        An open and writable HDF group.
    data_dict : dict
        A dict that needs to be written to HDF.
        Keys always need to be strings. Values are stored as follows:

            - subdicts -> subgroups.
            - np.array -> array
            - pd.dataframes -> subdicts with "is_pd_dataframe: True" attribute.
            - bool, int, float and str -> attrs.
            - None values are skipped and not stored explicitly.

    overwrite : bool
        If True, existing subgroups, arrays and attrs are fully
        truncated/overwritten.
        If False, the existing value in HDF remains unchanged.
        Default is False.
    compress : bool
        If True, all arrays are compressed with binary shuffle and "lzf"
        compression.
        If False, arrays are saved as provided.
        On average, compression halves file sizes,
        at the cost of 2-10 time longer accession times.
        Default is False.
    recursed : bool
        If False, the default progress callback is added while itereating over
        the keys of the data_dict.
        If True, no callback is added, allowing subdicts to not trigger
        callback.
        Default is False.
    chunked : bool
        If True, all arrays are chunked.
        If False, arrays are saved as provided.
        Default is False.

    Raises
    ------
    ValueError
        When a value of data_dict cannot be converted to an HDF value
        (see data_dict).
    KeyError
        When a key of data_dict is not a string.
    """
    import pandas as pd
    import h5py
    if recursed:
        iterable_dict = data_dict.items()
    else:
        iterable_dict = progress_callback(data_dict.items())
    for key, value in iterable_dict:
        if not isinstance(key, str):
            raise KeyError(f"Key {key} is not a string.")
        if isinstance(value, pd.core.frame.DataFrame):
            new_dict = {key: dict(value)}
            new_dict[key]["is_pd_dataframe"] = True
            create_hdf_group_from_dict(
                hdf_group,
                new_dict,
                overwrite=overwrite,
                recursed=True,
                compress=compress,
                chunked=chunked,
            )
        elif isinstance(value, (np.ndarray, pd.core.series.Series)):
            if isinstance(value, (pd.core.series.Series)):
                value = value.values
            if overwrite and (key in hdf_group):
                del hdf_group[key]
            if key not in hdf_group:
                if value.dtype.type == np.str_:
                    value = value.astype(np.dtype('O'))
                if value.dtype == np.dtype('O'):
                    hdf_group.create_dataset(
                        key,
                        data=value,
                        dtype=h5py.string_dtype()
                    )
                else:
                    hdf_group.create_dataset(
                        key,
                        data=value,
                        compression="lzf" if compress else None,
                        # compression="gzip" if compress else None, # TODO slower to make, faster to load?
                        shuffle=compress,
                        chunks=True if chunked else None,
                    )
        elif isinstance(value, (bool, int, float, str, np.bool_)):
            if overwrite or (key not in hdf_group.attrs):
                hdf_group.attrs[key] = value
        elif isinstance(value, dict):
            if key not in hdf_group:
                hdf_group.create_group(key)
            create_hdf_group_from_dict(
                hdf_group[key],
                value,
                overwrite=overwrite,
                recursed=True,
                compress=compress,
            )
        elif value is None:
            continue
        else:
            raise ValueError(
                f"The type of {key} is {type(value)}, which "
                "cannot be converted to an HDF value."
            )


def create_dict_from_hdf_group(
    hdf_group,
    mmap_arrays=None,
    parent_file_name: str = None,
) -> dict:
    """Convert the contents of an HDF group and return as normal Python dict.

    Parameters
    ----------
    hdf_group : h5py.File.group
        An open and readable HDF group.
    mmap_arrays : iterable
        These array will be mmapped instead of pre-loaded.
        Default is None
    parent_file_name : str
        The parent_file_name. This is required when mmap_arrays is not None.
        Default is None.

    Returns
    -------
    : dict
        A Python dict.
        Keys of the dict are names of arrays, attrs and subgroups.
        Values are corresponding arrays and attrs.
        Subgroups are converted to subdicts.
        If a subgroup has an "is_pd_dataframe=True" attr,
        it is automatically converted to a pd.dataFrame.

    Raises
    ------
    ValueError
        When an attr value in the HDF group is not an int, float, str or bool.
    """
    import h5py
    import pandas as pd
    result = {}
    for key in hdf_group.attrs:
        value = hdf_group.attrs[key]
        if isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.float64):
            result[key] = float(value)
        elif isinstance(value, (str, bool, np.bool_)):
            result[key] = value
        else:
            raise ValueError(
                f"The type of {key} is {type(value)}, which "
                "cannot be converted properly."
            )
    for key in hdf_group:
        subgroup = hdf_group[key]
        if isinstance(subgroup, h5py.Dataset):
            if (mmap_arrays is not None) and (subgroup.name in mmap_arrays):
                offset = subgroup.id.get_offset()
                if offset is not None:
                    shape = subgroup.shape
                    import mmap
                    with open(parent_file_name, "rb") as raw_hdf_file:
                        mmap_obj = mmap.mmap(
                            raw_hdf_file.fileno(),
                            0,
                            access=mmap.ACCESS_READ
                        )
                        result[key] = np.frombuffer(
                            mmap_obj,
                            dtype=subgroup.dtype,
                            count=np.prod(shape),
                            offset=offset
                        ).reshape(shape)
                        # TODO WARNING: mmap is not closed!
                    # result[key] = np.memmap(
                    #     dia_data_file_name,
                    #     dtype=subgroup.dtype,
                    #     mode="r",
                    #     offset=offset,
                    #     shape=shape,
                    # )
                else:
                    raise IOError(
                        f"Array {subgroup.name} cannot be mmapped. "
                        "Perhaps it is compressed or chunked?"
                    )
            else:
                result[key] = subgroup[:]
        else:
            if "is_pd_dataframe" in subgroup.attrs:
                result[key] = pd.DataFrame(
                    {
                        column: subgroup[column][:] for column in sorted(
                            subgroup
                        )
                    }
                )
            else:
                result[key] = create_dict_from_hdf_group(
                    hdf_group[key],
                    mmap_arrays,
                    parent_file_name,
                )
    return result

