import numpy as np
import numba
import pandas as pd
import tqdm
from typing import Union, Tuple

from alphabase.peptide.fragment import (
    create_fragment_mz_dataframe, 
    get_charged_frag_types
)

from alpharaw.ms_data_base import (
    MSData_Base, ms_reader_provider
)

from alpharaw.match.match_utils import (
    match_closest_peaks, match_highest_peaks, 
)
from ..utils.ms_path_utils import parse_ms_files_to_dict

@numba.njit
def match_one_raw_with_numba(
    spec_idxes, frag_start_idxes, frag_stop_idxes,
    all_frag_mzs,
    all_spec_mzs, all_spec_intensities, 
    peak_start_idxes, peak_stop_idxes,
    matched_intensities, matched_mz_errs,
    use_ppm, tol, matched_closest=True,
):
    """ 
    Internel function to match fragment mz values to spectrum mz values.
    Matched_mz_errs[i] = np.inf if no peaks are matched.

    Results will saved in place of matched_intensities 
    and matched_mz_errs.
    """
    for spec_idx, frag_start, frag_end in zip(
        spec_idxes, frag_start_idxes, frag_stop_idxes
    ):
        peak_start = peak_start_idxes[spec_idx]
        peak_stop = peak_stop_idxes[spec_idx]
        if peak_stop == peak_start: continue
        spec_mzs = all_spec_mzs[peak_start:peak_stop]
        spec_intens = all_spec_intensities[peak_start:peak_stop]

        frag_mzs = all_frag_mzs[frag_start:frag_end,:].copy()
        
        if use_ppm:
            frag_mz_tols = frag_mzs*tol*1e-6
        else:
            frag_mz_tols = np.full_like(frag_mzs, tol)
        
        if matched_closest:
            matched_idxes = match_closest_peaks(
                spec_mzs, spec_intens, 
                frag_mzs, frag_mz_tols
            ).reshape(-1)
        else:
            matched_idxes = match_highest_peaks(
                spec_mzs, spec_intens, 
                frag_mzs, frag_mz_tols
            ).reshape(-1)

        matched_intens = spec_intens[matched_idxes]
        matched_intens[matched_idxes==-1] = 0

        matched_mass_errs = np.abs(
            spec_mzs[
                matched_idxes.reshape(-1)
            ]-frag_mzs.reshape(-1)
        )
        matched_mass_errs[matched_idxes==-1] = np.inf

        matched_intensities[
            frag_start:frag_end,:
        ] = matched_intens.reshape(frag_mzs.shape)

        matched_mz_errs[
            frag_start:frag_end,:
        ] = matched_mass_errs.reshape(frag_mzs.shape)


def load_ms_data(
    ms_file:Union[str, MSData_Base],
    ms_file_type:str='alpharaw_hdf',
)->MSData_Base:
    """Load MS files

    Parameters
    ----------
    ms_file : str | MSData_Base
        ms2 file path

    ms_file_type : str, optional
        ms2 file type, could be 
        ["alpharaw_hdf","thermo","sciex","alphapept_hdf","mgf"].
        Default to 'alpharaw_hdf'
    """
    if isinstance(ms_file, MSData_Base):
        return ms_file
    else:
        raw_data = ms_reader_provider.get_reader(
            ms_file_type
        )
        raw_data.import_raw(ms_file)
        return raw_data

class PepSpecMatch:
    """
    Extract fragment ions from MS2 data.
    """
    match_closest:bool = True
    use_ppm:bool = True
    #: matching mass tolerance
    tolerance:float = 20
    def __init__(self,
        charged_frag_types:list = get_charged_frag_types(
            ['b','y','b_modloss','y_modloss'], 2
        ), 
        match_closest:bool=True,
        use_ppm:bool = True,
        tol_value:float = 20.0
    ):
        """
        Parameters
        ----------
        charged_frag_types : list, optional
            fragment types with charge states, 
            e.g. ['b_z1', 'y_z2', 'b_modloss_z1', 'y_H2O_z2'].
            By default `get_charged_frag_types(['b','y','b_modloss','y_modloss'], 2)`

        match_closest : bool, optional
            if True, match the closest peak for a m/z;
            if False, matched the higest peak for a m/z in the tolerance range.
            By default True

        use_ppm : bool, optional
            If use ppm, by default True
            
        tol_value : float, optional
            tolerance value, by default 20.0
        """
        self.charged_frag_types = charged_frag_types
        self.match_closest = match_closest
        self.use_ppm = use_ppm
        self.tolerance = tol_value

    def _preprocess_psms(self, psm_df):
        pass

    def get_fragment_mz_df(self, psm_df):
        return create_fragment_mz_dataframe(
            psm_df, self.charged_frag_types
        )

    def _add_missing_columns_to_psm_df(self,
        psm_df:pd.DataFrame, raw_data=None
    ):
        if raw_data is None:
            raw_data = self.raw_data
        add_spec_info_list = []
        if 'rt' not in psm_df.columns:
            add_spec_info_list.append('rt')

        if (
            'mobility' not in psm_df.columns and 
            'mobility' in raw_data.spectrum_df.columns
        ):
            add_spec_info_list.append('mobility')

        if len(add_spec_info_list) > 0:
            # pfind does not report RT in the result file
            psm_df = psm_df.reset_index().merge(
                raw_data.spectrum_df[
                    ['spec_idx']+add_spec_info_list
                ],
                how='left',
                on='spec_idx',
            ).set_index('index')

            if 'rt' in add_spec_info_list:
                psm_df['rt_norm'] = (
                    psm_df.rt/raw_data.spectrum_df.rt.max()
                )
        # if 'rt_sec' not in psm_df.columns:
        #     psm_df['rt_sec'] = psm_df.rt*60
        return psm_df

    def _prepare_matching_dfs(self, psm_df):

        fragment_mz_df = self.get_fragment_mz_df(psm_df)
        
        matched_intensity_df = pd.DataFrame(
            np.zeros_like(
                fragment_mz_df.values, dtype=np.float64
            ), 
            columns=fragment_mz_df.columns
        )

        matched_mz_err_df = pd.DataFrame(
            np.full_like(
                fragment_mz_df.values, np.inf, 
                dtype=np.float64
            ), 
            columns=fragment_mz_df.columns
        )
        return (
            fragment_mz_df, matched_intensity_df, 
            matched_mz_err_df
        )

    def load_ms_data(self,
        ms_file:Union[str, MSData_Base],
        ms_file_type:str='alpharaw_hdf',
        **kwargs
    ):
        """Load MS files

        Parameters
        ----------
        ms_file : str | MSData_Base
            ms2 file path

        ms_file_type : str, optional
            ms2 file type, could be 
            ["alpharaw_hdf","thermo","sciex","alphapept_hdf","mgf"].
            Default to 'alpharaw_hdf'
        """
        self.raw_data = load_ms_data(ms_file, ms_file_type)

    def get_peaks(self,
        spec_idx:int,
        **kwargs
    ):
        return self.raw_data.get_peaks(spec_idx)

    def _match_one_psm(self,
        spec_mzs:np.ndarray, spec_intens:np.ndarray,
        fragment_mz_df:pd.DataFrame, 
        matched_intensity_df:pd.DataFrame,
        matched_mz_err_df:pd.DataFrame,
        frag_start_idx:int, frag_stop_idx:int,
    ):
        if len(spec_mzs)==0: return

        frag_mzs = fragment_mz_df.values[
            frag_start_idx:frag_stop_idx,:
        ]

        if self.use_ppm:
            mz_tols = frag_mzs*self.tolerance*1e-6
        else:
            mz_tols = np.full_like(frag_mzs, self.tolerance)

        if self.match_closest:
            matched_idxes = match_closest_peaks(
                spec_mzs, spec_intens, frag_mzs, mz_tols
            )
        else:
            matched_idxes = match_highest_peaks(
                spec_mzs, spec_intens, frag_mzs, mz_tols,
            )
        
        matched_intens = spec_intens[matched_idxes]
        matched_intens[matched_idxes==-1] = 0

        matched_mz_errs = np.abs(
            spec_mzs[matched_idxes]-frag_mzs
        )
        matched_mz_errs[matched_idxes==-1] = np.inf

        matched_intensity_df.values[
            frag_start_idx:frag_stop_idx,:
        ] = matched_intens

        matched_mz_err_df.values[
            frag_start_idx:frag_stop_idx,:
        ] = matched_mz_errs

    def match_ms2_one_raw(self, 
        psm_df_one_raw: pd.DataFrame,
        verbose:bool=False,
    )->tuple:
        """
        Matching psm_df_one_raw against self.raw_data 
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

        psm_iters = psm_df_one_raw[[
            'spec_idx', 'frag_start_idx', 
            'frag_stop_idx'
        ]].values
        if verbose:
            psm_iters = tqdm.tqdm(psm_iters)
        
        for (
            spec_idx, frag_start_idx, frag_stop_idx
        ) in psm_iters:
            (
                spec_mzs, spec_intens
            ) = self.get_peaks(spec_idx)

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

    def _match_ms2_one_raw_numba(self, raw_name, df_group):
        if raw_name in self._ms_file_dict:
            raw_data = load_ms_data(
                self._ms_file_dict[raw_name], self._ms_file_type
            )

            df_group = self._add_missing_columns_to_psm_df(
                df_group, raw_data
            )

            match_one_raw_with_numba(
                df_group.spec_idx.values,
                df_group.frag_start_idx.values,
                df_group.frag_stop_idx.values,
                self.fragment_mz_df.values,
                raw_data.peak_df.mz.values, 
                raw_data.peak_df.intensity.values,
                raw_data.spectrum_df.peak_start_idx.values,
                raw_data.spectrum_df.peak_stop_idx.values,
                self.matched_intensity_df.values,
                self.matched_mz_err_df.values,
                self.use_ppm, self.tolerance, 
                self.match_closest
            )
    
    def match_ms2_multi_raw(self,
        psm_df: pd.DataFrame,
        ms_files: Union[dict,list],
        ms_file_type:str = 'alpharaw_hdf',
    ):
        """Matching PSM dataframe against the ms2 files in ms_files
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
            Could be 'alpharaw_hdf', 'mgf' or 'thermo', 'sciex', 'alphapept_hdf'. 
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
        self._preprocess_psms(psm_df)
        self.psm_df = psm_df
        
        (
            self.fragment_mz_df, 
            self.matched_intensity_df,
            self.matched_mz_err_df,
        ) = self._prepare_matching_dfs(psm_df)
        
        if isinstance(ms_files, dict):
            self._ms_file_dict = ms_files
        else:
            self._ms_file_dict = parse_ms_files_to_dict(ms_files)

        self._ms_file_type = ms_file_type

        for raw_name, df_group in tqdm.tqdm(
            self.psm_df.groupby('raw_name')
        ):
            self._match_ms2_one_raw_numba(raw_name, df_group)

        return (
            self.psm_df, self.fragment_mz_df, 
            self.matched_intensity_df, self.matched_mz_err_df
        )
