import numpy as np
import numba
import pandas as pd
import tqdm
from typing import Union, Tuple

from alphabase.peptide.fragment import (
    create_fragment_mz_dataframe, 
    get_charged_frag_types,
    concat_precursor_fragment_dataframes
)

from alpharaw.ms_data_base import (
    MSData_Base, ms_reader_provider,
    PEAK_MZ_DTYPE, PEAK_INTENSITY_DTYPE
)

from alpharaw.match.match_utils import (
    match_closest_peaks, match_highest_peaks, 
)
from alpharaw.utils.ms_path_utils import parse_ms_files_to_dict

from alpharaw.dia.normal_dia import NormalDIAGrouper


class PepSpecMatch:
    """
    Extract fragment ions from MS2 data.
    """
    match_closest:bool = True
    use_ppm:bool = True
    #: matching mass tolerance
    tolerance:float = 20.0
    ms_loader_thread_num:int = 4
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

    def get_fragment_mz_df(self):
        return create_fragment_mz_dataframe(
            self.psm_df, self.charged_frag_types,
            dtype=PEAK_MZ_DTYPE,
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
            'nce' in raw_data.spectrum_df.columns 
            and 'nce' not in psm_df.columns
        ):
            add_spec_info_list.append('nce')

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

    def _prepare_matching_dfs(self):

        fragment_mz_df = self.get_fragment_mz_df()

        matched_intensity_df = pd.DataFrame(
            np.zeros_like(
                fragment_mz_df.values, 
                dtype=PEAK_INTENSITY_DTYPE
            ), 
            columns=fragment_mz_df.columns
        )

        matched_mz_err_df = pd.DataFrame(
            np.full_like(
                fragment_mz_df.values, np.inf, 
                dtype=PEAK_MZ_DTYPE
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
        process_count:int = 8,
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
        self.raw_data = load_ms_data(
            ms_file, ms_file_type, 
            process_count=process_count
        )

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

        spec_mzs = spec_mzs.astype(PEAK_MZ_DTYPE)

        frag_mzs = fragment_mz_df.values[
            frag_start_idx:frag_stop_idx,:
        ]

        if self.use_ppm:
            mz_tols = frag_mzs*self.tolerance*1e-6
        else:
            mz_tols = np.full_like(
                frag_mzs, self.tolerance
            )

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
        self.psm_df = psm_df_one_raw

        psm_df_one_raw = self._add_missing_columns_to_psm_df(
            psm_df_one_raw, self.raw_data
        )

        (
            fragment_mz_df, 
            matched_intensity_df,
            matched_mz_err_df,
        ) = self._prepare_matching_dfs()

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

    def _match_ms2_one_raw_numba(self, 
        raw_name:str, psm_df_one_raw:pd.DataFrame
    ):
        if raw_name in self._ms_file_dict:
            raw_data = load_ms_data(
                self._ms_file_dict[raw_name], self._ms_file_type,
                process_count=self.ms_loader_thread_num,
            )

            psm_df_one_raw = self._add_missing_columns_to_psm_df(
                psm_df_one_raw, raw_data
            )

            if self.use_ppm:
                all_frag_mz_tols = self.fragment_mz_df.values*self.tolerance*1e-6
            else:
                all_frag_mz_tols = np.full_like(self.fragment_mz_df.values, self.tolerance)

            match_one_raw_with_numba(
                psm_df_one_raw.spec_idx.values,
                psm_df_one_raw.frag_start_idx.values,
                psm_df_one_raw.frag_stop_idx.values,
                self.fragment_mz_df.values,
                all_frag_mz_tols, 
                raw_data.peak_df.mz.values, 
                raw_data.peak_df.intensity.values,
                raw_data.spectrum_df.peak_start_idx.values,
                raw_data.spectrum_df.peak_stop_idx.values,
                self.matched_intensity_df.values,
                self.matched_mz_err_df.values,
                self.match_closest
            )
        else:
            print(f"`{raw_name}` is not found in ms_file_dict.")
        return psm_df_one_raw
    
    def match_ms2_multi_raw(self,
        psm_df: pd.DataFrame,
        ms_files: Union[dict,list],
        ms_file_type:str = 'alpharaw_hdf',
        process_num:int = 1,
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
        ) = self._prepare_matching_dfs()
        
        if isinstance(ms_files, dict):
            self._ms_file_dict = ms_files
        else:
            self._ms_file_dict = parse_ms_files_to_dict(ms_files)

        self._ms_file_type = ms_file_type

        self.ms_loader_thread_num = process_num
        # if process_num <= 1 or len(self._ms_file_dict) <= 1:
        psm_df_list = []
        for raw_name, df_group in tqdm.tqdm(
            self.psm_df.groupby('raw_name')
        ):
            psm_df_list.append(self._match_ms2_one_raw_numba(raw_name, df_group))
        # else:
        #     with mp.get_context("spawn").Pool(processes=process_num) as p:
        #         df_groupby = self.psm_df.groupby('raw_name')
        #         def gen_group_df(df_groupby):
        #             for raw_name, df_group in df_groupby:
        #                 yield (raw_name, df_group)
        #         process_bar(
        #             p.imap_unordered(
        #                 self._match_ms2_one_raw_numba, 
        #                 gen_group_df(df_groupby)
        #             ), df_groupby.ngroups
        #         )

        self.psm_df = pd.concat(psm_df_list, ignore_index=True)     
        return (
            self.psm_df, self.fragment_mz_df, 
            self.matched_intensity_df, self.matched_mz_err_df
        )

class PepSpecMatch_DIA(PepSpecMatch):
    max_spec_per_query: int = 3
    min_frag_mz: float = 200.0

    def _add_missing_columns_to_psm_df(self,
        psm_df:pd.DataFrame, raw_data=None
    ):
        # DIA results do not have spec_idx/scan_num in psm_df, nothing to merge
        return psm_df
    
    def _prepare_matching_dfs(self):

        fragment_mz_df = self.get_fragment_mz_df()
        fragment_mz_df = pd.concat(
            [fragment_mz_df]*self.max_spec_per_query,
            ignore_index=True
        )
        if self.use_ppm:
            self.all_frag_mz_tols = fragment_mz_df.values*self.tolerance*1e-6
        else:
            self.all_frag_mz_tols = np.full_like(fragment_mz_df.values, self.tolerance)

        psm_df_list = []
        len_frags = len(fragment_mz_df)//self.max_spec_per_query
        for i in range(self.max_spec_per_query):
            psm_df = self.psm_df.copy()
            psm_df["frag_start_idx"] = psm_df.frag_start_idx+i*len_frags
            psm_df["frag_stop_idx"] = psm_df.frag_stop_idx+i*len_frags
            psm_df_list.append(psm_df)
        self.psm_df = pd.concat(psm_df_list, ignore_index=True)

        matched_intensity_df = pd.DataFrame(
            np.zeros_like(
                fragment_mz_df.values,
                dtype=PEAK_INTENSITY_DTYPE
        ), columns=fragment_mz_df.columns)

        matched_mz_err_df = pd.DataFrame(
            np.zeros_like(
                fragment_mz_df.values,
                dtype=PEAK_MZ_DTYPE
        ), columns=fragment_mz_df.columns)

        return (
            fragment_mz_df, matched_intensity_df, 
            matched_mz_err_df
        )
    def _match_ms2_one_raw_numba(self, 
        raw_name, psm_df_one_raw
    ):
        psm_df_one_raw = psm_df_one_raw.reset_index(drop=True)
        
        if raw_name in self._ms_file_dict:
            raw_data = load_ms_data(
                self._ms_file_dict[raw_name], self._ms_file_type,
                process_count=self.ms_loader_thread_num
            )

            psm_origin_len = len(psm_df_one_raw)//self.max_spec_per_query
            
            grouper = NormalDIAGrouper(raw_data)

            psm_groups = grouper.assign_dia_groups(
                psm_df_one_raw.precursor_mz.values[
                    :psm_origin_len
                ]
            )
            
            all_spec_idxes = np.full(
                len(psm_df_one_raw), -1, dtype=np.int32
            )

            for dia_group, group_df in grouper.dia_group_dfs:
                psm_idxes = psm_groups[dia_group]
                if len(psm_idxes) == 0: continue
                psm_idxes = np.array(psm_idxes, dtype=np.int32)
                spec_idxes = get_dia_spec_idxes(
                    group_df.rt.values,
                    psm_df_one_raw.rt.values[psm_idxes],
                    max_spec_per_query=self.max_spec_per_query
                )
                for i in range(spec_idxes.shape[-1]):
                    all_spec_idxes[
                        psm_idxes+psm_origin_len*i
                    ] = spec_idxes[:,i]

                match_one_raw_with_numba(
                    all_spec_idxes,
                    psm_df_one_raw.frag_start_idx.values,
                    psm_df_one_raw.frag_stop_idx.values,
                    self.fragment_mz_df.values,
                    self.all_frag_mz_tols, 
                    raw_data.peak_df.mz.values, 
                    raw_data.peak_df.intensity.values,
                    group_df.peak_start_idx.values,
                    group_df.peak_stop_idx.values,
                    self.matched_intensity_df.values,
                    self.matched_mz_err_df.values,
                    self.match_closest
                )
        else:
            print(f"`{raw_name}` is not found in ms_file_dict.")
        return psm_df_one_raw

    def match_ms2_multi_raw(self, 
        psm_df: pd.DataFrame, 
        ms_files: Tuple[dict,list], 
        ms_file_type: str = "alpharaw_hdf",
        process_num:int = 8,
    ):
        if isinstance(ms_files, list):
            ms_files = parse_ms_files_to_dict(ms_files)
        psm_df = psm_df[
            psm_df.raw_name.isin(ms_files)
        ].reset_index(drop=True)
        super().match_ms2_multi_raw(
            psm_df, ms_files, ms_file_type,
            process_num
        )

        return (
            self.psm_df, self.fragment_mz_df, 
            self.matched_intensity_df, self.matched_mz_err_df
        )

@numba.njit
def match_one_raw_with_numba(
    spec_idxes, frag_start_idxes, frag_stop_idxes,
    all_frag_mzs, all_frag_mz_tols,
    all_spec_mzs, all_spec_intensities, 
    peak_start_idxes, peak_stop_idxes,
    matched_intensities, matched_mz_errs, 
    matched_closest=True,
):
    """ 
    Internel function to match fragment mz values to spectrum mz values.
    Matched_mz_errs[i] = np.inf if no peaks are matched.

    Results will saved in place of matched_intensities 
    and matched_mz_errs.
    """
    for spec_idx, frag_start, frag_stop in zip(
        spec_idxes, frag_start_idxes, frag_stop_idxes
    ):
        if spec_idx == -1: continue
        peak_start = peak_start_idxes[spec_idx]
        peak_stop = peak_stop_idxes[spec_idx]
        if peak_stop == peak_start: continue
        spec_mzs = all_spec_mzs[peak_start:peak_stop]
        spec_intens = all_spec_intensities[peak_start:peak_stop]

        frag_mzs = all_frag_mzs[frag_start:frag_stop,:].copy()
        frag_mz_tols = all_frag_mz_tols[frag_start:frag_stop,:].copy()
        
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
            frag_start:frag_stop,:
        ] = matched_intens.reshape(frag_mzs.shape)

        matched_mz_errs[
            frag_start:frag_stop,:
        ] = matched_mass_errs.reshape(frag_mzs.shape)

def load_ms_data(
    ms_file:Union[str, MSData_Base],
    ms_file_type:str='alpharaw_hdf',
    process_count:int = 8,
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
            ms_file_type, process_count=process_count
        )
        raw_data.import_raw(ms_file)
        return raw_data

    
@numba.njit
def get_best_matched_intens(
    matched_intensity_values:np.ndarray,
    frag_start_idxes:np.ndarray,
    frag_stop_idxes:np.ndarray,
):
    ret_intens = np.zeros(
        shape = matched_intensity_values.shape[1:],
        dtype=matched_intensity_values.dtype
    )
    for i in range(len(frag_start_idxes)):
        start = frag_start_idxes[i]
        stop = frag_stop_idxes[i]
        i = np.argmax(np.matmul(
            matched_intensity_values[:,start:stop,:],
            matched_intensity_values[:,start:stop,:].T
        ).sum())
        ret_intens[start:stop,:] = matched_intensity_values[i,start:stop,:]
    return ret_intens

@numba.njit
def get_ion_count_scores(
    frag_mz_values:np.ndarray,
    matched_intens:np.ndarray,
    frag_start_idxes:np.ndarray,
    frag_stop_idxes:np.ndarray,
    min_mz:float = 200,
):
    scores = []
    for i in range(len(frag_start_idxes)):
        scores.append(np.count_nonzero(
            matched_intens[
                frag_start_idxes[i]:frag_stop_idxes[i],:
            ].copy().reshape(-1)[
                frag_mz_values[
                    frag_start_idxes[i]:frag_stop_idxes[i],:
                ].copy().reshape(-1)>=min_mz
            ]
        ))
    return np.array(scores,np.int32)

@numba.njit    
def get_dia_spec_idxes(
    spec_rt_values:np.ndarray, 
    query_rt_values:np.ndarray, 
    max_spec_per_query:int,
):
    rt_idxes = np.searchsorted(spec_rt_values, query_rt_values)
    
    spec_idxes = np.full(
        (len(rt_idxes),max_spec_per_query),
        -1, dtype=np.int32
    )
    n = max_spec_per_query // 2

    for iquery in range(len(rt_idxes)):
        if rt_idxes[iquery] < n:
            spec_idxes[iquery,:] = np.arange(0, max_spec_per_query)
        else:
            spec_idxes[iquery,:] = np.arange(
                rt_idxes[iquery]-n, 
                rt_idxes[iquery]-n+max_spec_per_query
            )
    return spec_idxes

            