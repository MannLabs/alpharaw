import pandas as pd
import numpy as np
import tqdm
from typing import Union, Tuple

from alphatims.bruker import TimsTOF

from alpharaw.ms_data_base import (
    MSData_Base, ms_reader_provider
)

from alpharaw.match.match_utils import (
    match_closest_peaks, match_highest_peaks
)

from alpharaw.wrappers.alphatims_wrapper import (
    AlphaTimsWrapper
)

from .psm_match import PepSpecMatch
from ..utils.ms_path_utils import parse_ms_files_to_dict

alphatims_hdf_types = [
    'alphatims', 'alphatims_hdf',
    'tims.hdf',
]

def load_ms_data_tims(
    ms_file:Union[str, MSData_Base, TimsTOF],
    ms_file_type:str='alpharaw_hdf',
    dda:bool=False,
)->Tuple[MSData_Base, TimsTOF]:
    """Load ms data as TimsTOF object

    Parameters
    ----------
    ms_file : str
        ms2 file path

    ms_file_type : str, optional
        ms2 file type, could be 
        ["alpharaw_hdf","raw.hdf","thermo","sciex","alphapept_hdf","mgf"].
        Default to 'alphatims_hdf'

    dda : bool, optional
        if it is DDA data, by default False
    
    Returns
    -------
    tuple
        MSData_Base: alpharaw MS Data (Reader) object
        TimsTOF: AlphaTims object
    """
    if isinstance(ms_file, TimsTOF):
        return None, ms_file
    elif ms_file_type.lower() in alphatims_hdf_types:
        return None, TimsTOF(ms_file)
    else:
        if isinstance(ms_file, MSData_Base):
            raw_data = ms_file
        else:
            raw_data = ms_reader_provider.get_reader(
                ms_file_type
            )
            raw_data.import_raw(ms_file)

        tims_data = AlphaTimsWrapper(
            raw_data, dda=dda
        )
        return raw_data, tims_data


class PepSpecMatch_AlphaTims(PepSpecMatch):
    """
    Inherited from `alpharaw.match.psm_match.PepSpecMatch`, but
    this can be used for DIA PSM matching by selecting 
    spectra with RT (and IM) values.
    """

    #: RT win to get a MS2 spectrum by slicing
    rt_sec_win_to_slice_spectrum = 6.0

    #: IM win to get a MS2 spectrum by slicing
    im_win_to_slice_spectrum = 0.1

    #: find closest MS2 for the given RT when slicing
    find_closest_ms2_by_rt = True

    def get_peak_df(self,
        rt:float,
        precursor_mz:float,
        im:float=0.0,
    ):
        rt_sec = rt*60
        im_slice = (
            slice(None) if im == 0 else 
            slice(
                im-self.im_win_to_slice_spectrum/2,
                im+self.im_win_to_slice_spectrum/2
            )
        )
        rt_slice = slice(
            rt_sec-self.rt_sec_win_to_slice_spectrum/2,
            rt_sec+self.rt_sec_win_to_slice_spectrum/2,
        )

        spec_df = self.tims_data[
            rt_slice, im_slice
        ]
        spec_df = spec_df[
            (spec_df.quad_low_mz_values <= precursor_mz)
            &(spec_df.quad_high_mz_values >= precursor_mz)
        ]

        if self.find_closest_ms2_by_rt:
            closest_rt = 1000000
            for _, df in spec_df.groupby('frame_indices'):
                if (
                    abs(df.rt_values.values[0]-rt_sec) < closest_rt 
                ):
                    spec_df = df
                    closest_rt = abs(df.rt_values.values[0]-rt_sec)
        
        return spec_df.sort_values('mz_values').reset_index(drop=True)

    def get_peaks(
        self,
        rt:float,
        precursor_mz:float,
        im:float=0.0,
    ):
        spec_df = self.get_peak_df(rt, precursor_mz, im)
        return (
            spec_df.mz_values.values, 
            spec_df.intensity_values.values
        )

    def load_ms_data(self, ms_file, ms_file_type, dda=False):
        self.raw_data, self.tims_data = load_ms_data_tims(
            ms_file, ms_file_type, dda
        )

    def match_ms2_one_raw(self, 
        psm_df_one_raw: pd.DataFrame,
    )->tuple:
        """
        Matching psm_df_one_raw against 
        self.tims_data and self.raw_data
        after `self.load_ms_data()`

        Parameters
        ----------
        psm_df_one_raw : pd.DataFrame
            psm dataframe 
            that contains only one raw file

        Returns
        -------
        tuple:
            pd.DataFrame: psm dataframe with fragment index information.
            
            pd.DataFrame: fragment mz dataframe.
            
            pd.DataFrame: matched intensity dataframe.
            
            pd.DataFrame: matched mass error dataframe. 
            np.inf if a fragment is not matched.
            
        """
        self._preprocess_psms(psm_df_one_raw)
        
        psm_df_one_raw = self._add_missing_columns_to_psm_df(
            psm_df_one_raw
        )

        (
            fragment_mz_df, 
            matched_intensity_df,
            matched_mz_err_df,
        ) = self._prepare_matching_dfs(psm_df_one_raw)

        if 'mobility' in psm_df_one_raw:
            query_columns = [
                'rt', 'precursor_mz', 
                'mobility',
                'frag_start_idx', 
                'frag_stop_idx',
            ]
        else:
            query_columns = [
                'rt', 'precursor_mz', 
                'frag_start_idx', 
                'frag_stop_idx',
            ]
        
        for items in psm_df_one_raw[query_columns].values:
            frag_start_idx = int(items[-2])
            frag_stop_idx = int(items[-1])
            
            spec_mzs, spec_intens = self.get_peaks(
                *items[:-2],
            )

            self._match_one_psm(
                spec_mzs, spec_intens,
                fragment_mz_df, 
                matched_intensity_df,
                matched_mz_err_df,
                frag_start_idx, frag_stop_idx,
            )
        return (
            psm_df_one_raw, fragment_mz_df, 
            matched_intensity_df, matched_mz_err_df
        )

    def match_ms2_multi_raw(self, 
        psm_df: pd.DataFrame, 
        ms_files: Union[dict, list], 
        ms_file_type: str = 'alphatims',
        dda:bool = False,
    ):
        """
        Matching PSM dataframe against the ms2 files in ms_files
        This method will store matched values as attributes:
        - self.psm_df
        - self.fragment_mz_df
        - self.matched_intensity_df
        - self.matched_mz_err_df

        Parameters
        ----------
        psm_df : pd.DataFrame
            PSM dataframe

        ms_files : dict | list
            if dict: {raw_name: ms2 path}
            if list: [ms2 path1, ms2 path2]

        ms_file_type : str, optional
            One of ["alphatims_hdf","alpharaw_hdf","thermo","sciex","alphapept_hdf","mgf"]
            Defaults to 'alphapept'.
            
        Returns
        -------
        tuple:
            pd.DataFrame: psm dataframe with fragment index information.
            
            pd.DataFrame: fragment mz dataframe.
            
            pd.DataFrame: matched intensity dataframe.
            
            pd.DataFrame: matched mass error dataframe. 
            np.inf if a fragment is not matched.
            
        """
        raise NotImplementedError(
            "Not necessary for matching multiple raw files using AlphaTims, "
            "loop through `match_ms2_one_raw()`"
        )
