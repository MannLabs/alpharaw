import warnings

import numpy as np

from alpharaw.ms_data_base import MSData_Base


def test_add_column_in_spec_df_avoids_chained_assignment_warning():
    ms_data = MSData_Base()
    ms_data.create_spectrum_df(3)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ms_data.add_column_in_spec_df_by_spec_idxes(
            "ms_level",
            np.array([2, 3]),
            np.array([0, 2]),
            dtype=np.int8,
            na_value=1,
        )

    np.testing.assert_array_equal(ms_data.spectrum_df["ms_level"], [2, 1, 3])
    assert ms_data.spectrum_df["ms_level"].dtype == np.int8
