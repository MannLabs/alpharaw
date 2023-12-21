import pandas as pd

import typing

from alpharaw.ms_data_base import MSData_Base

from alpharaw.utils.df_processing import remove_unused_peaks
from alpharaw.utils.timstof import convert_to_alphatims
from alphatims.bruker import TimsTOF

class NormalDIAGrouper():
    def __init__(self, ms_data: MSData_Base):
        self.ms_data = ms_data
        self.ms_data.spectrum_df[
            "dia_group"
        ] = ms_data.spectrum_df.precursor_mz.astype(int)

    def get_grouped_ms_data(self, 
        dia_group:int=-1, 
        return_alpharaw_data: bool=True,
        return_alphatims_data: bool=True,
    )->typing.Union[MSData_Base, TimsTOF, typing.Tuple[MSData_Base, TimsTOF]]:
        """ Get compressed MS data for isolation window `dia_group`.

        Args:
            dia_group (int, optional): The DIA group, -1 means ms1. Defaults to -1.
            return_alphatims_data (bool, optional): If return `MSData_Base`. Defaults to True
            return_alphatims_data (bool, optional): If return alphatims object. Defaults to True.

        Returns:
            MSData_Base: Compressed MS data, if `return_alpharaw_data==True`
            TimsTOF: Alphatims object for the window, if `return_alphatims_data==True`
        """

        spec_df = self.ms_data.spectrum_df.query(f"dia_group == {dia_group}")

        if return_alphatims_data:
            ms_data, ms_tims = convert_to_alphatims(
                spec_df, self.ms_data.peak_df, dda=False
            )
            if return_alpharaw_data:
                return ms_data, ms_tims
            else:
                return ms_tims
        else:
            ms_data = MSData_Base()

            spec_df, peak_df = remove_unused_peaks(
                spec_df, self.ms_data.peak_df
            )

            ms_data.spectrum_df = spec_df
            ms_data.peak_df = peak_df
            return ms_data
        
