import typing

import pandas as pd
import numpy as np

from alpharaw.match.match_utils import (
    match_closest_peaks, match_highest_peaks
)

from alphabase.peptide.precursor import (
    calc_precursor_mz,
    calc_precursor_isotope_intensity,
)

from alphabase.peptide.fragment import (
    create_fragment_mz_dataframe,
    flatten_fragments
)

from alphabase.constants.modification import MOD_MASS
from alphabase.constants.atom import MASS_ISOTOPE

def make_psm_plot_df_for_peptide(
    spec_masses:np.ndarray,
    spec_intensities:np.ndarray,
    sequence: str,
    mods: str,
    mod_sites: str,
    charge: int,
    rt_sec: float = 0.0,
    mobility: float = 0.0,
    ppm:float = 20.0,
    charged_frag_types: list = ["b_z1","b_z2","y_z1","y_z2"],
    include_fragments:bool=True,
    fragment_intensity_df:pd.DataFrame=None,
    include_precursor_isotopes:bool=False,
    max_isotope:int = 6,
    min_frag_mz:float = 100.0,
    match_mode:typing.Literal["closest","highest"]="closest",
)->pd.DataFrame:
    
    plot_df = make_xic_plot_df_for_peptide(
        sequence, mods, mod_sites, charge,
        rt_sec=rt_sec, mobility=mobility,
        charged_frag_types=charged_frag_types,
        include_fragments=include_fragments,
        fragment_intensity_df=fragment_intensity_df,
        include_precursor_isotopes=include_precursor_isotopes,
        max_isotope=max_isotope,
        min_frag_mz=min_frag_mz,
    )

    return make_psm_plot_df(
        spec_masses=spec_masses,
        spec_intensities=spec_intensities,
        query_masses=plot_df.mz.values,
        query_ion_names = plot_df.ion_name.values,
        query_mass_tols = plot_df.mz.values*ppm*1e-6,
        query_frag_idxes = plot_df.fragment_site.values,
        modified_sequence = plot_df.modified_sequence.values[0],
        mod_sites=mod_sites,
        query_intensities = plot_df.intensity.values 
            if "intensity" in plot_df.columns else None,
        match_mode = match_mode,    
    )

def make_xic_plot_df_for_peptide(
    sequence: str,
    mods: str,
    mod_sites: str,
    charge: int,
    rt_sec: float = 0.0,
    mobility: float = 0.0,
    ms_level:int=2,
    charged_frag_types: list = ["b_z1","b_z2","y_z1","y_z2"],
    include_fragments:bool=True,
    fragment_intensity_df:pd.DataFrame = None,
    include_precursor_isotopes:bool=False,
    max_isotope:int = 6,
    min_frag_mz:float = 100.0,
)->pd.DataFrame:
    if ms_level == 1:
        include_fragments = False
    precursor_df, fragment_mz_df = make_precursor_fragment_df(
        sequence, mods, mod_sites, charge,
        include_fragments=include_fragments,
        charged_frag_types=charged_frag_types,
        include_precursor_isotopes=include_precursor_isotopes,
        max_isotope=max_isotope,
    )
    if fragment_intensity_df is not None:
        columns = np.intersect1d(
            fragment_mz_df.columns.values,
            fragment_intensity_df.columns.values,
        )
        fragment_mz_df = fragment_mz_df[columns]
        fragment_intensity_df = fragment_intensity_df[columns]
        
    return translate_precursor_fragment_df_to_plot_df(
        precursor_df, fragment_mz_df,
        fragment_intensity_df=fragment_intensity_df,
        rt_sec=rt_sec,
        mobility=mobility,
        ms_level=ms_level,
        min_frag_mz=min_frag_mz,
    )

def make_psm_plot_for_dfs(
    spec_masses:np.ndarray,
    spec_intensities:np.ndarray,
    precursor_df:pd.DataFrame, 
    fragment_mz_df:pd.DataFrame,
    fragment_intensity_df:pd.DataFrame = None,
    ppm:float = 20.0,
    min_frag_mz: float = 100.0,
    min_frag_intensity: float = 0.001,
    match_mode:typing.Literal["closest","highest"]="closest",
):
    plot_df = translate_precursor_fragment_df_to_plot_df(
        precursor_df, 
        fragment_mz_df,
        fragment_intensity_df=fragment_intensity_df,
        min_frag_mz = min_frag_mz,
        min_frag_intensity=min_frag_intensity
    )

    return make_psm_plot_df(
        spec_masses=spec_masses,
        spec_intensities=spec_intensities,
        query_masses=plot_df.mz.values,
        query_ion_names = plot_df.ion_name.values,
        query_mass_tols = plot_df.mz.values*ppm*1e-6,
        query_frag_idxes = plot_df.fragment_site.values,
        modified_sequence = plot_df.modified_sequence.values[0],
        mod_sites=precursor_df.mod_sites.values[0],
        query_intensities = plot_df.intensity.values 
            if "intensity" in plot_df.columns else None,
        match_mode = match_mode,    
    )

def make_xic_plot_df(
    query_masses:np.ndarray,
    query_ion_names:typing.List[str],
    query_rt_sec:float, 
    precursor_mz:float,
    query_im:float = 0.0,
    query_intensities:np.ndarray = None,
)->pd.DataFrame:
    df = pd.DataFrame(dict(
        mz=query_masses,
        ion_name=query_ion_names,
        rt_sec=query_rt_sec,
        precursor_mz=precursor_mz,
    ))
    if query_im:
        df["im"] = query_im
    if query_intensities is not None:
        df["intensity"] = query_intensities
    return df

def make_psm_plot_df(
    spec_masses:np.ndarray,
    spec_intensities:np.ndarray,
    query_masses:np.ndarray,
    query_ion_names:typing.List[str],
    query_mass_tols:np.ndarray,
    query_frag_idxes:np.ndarray,
    modified_sequence:str = "",
    mod_sites:str = "",
    query_intensities:np.ndarray = None,
    match_mode:typing.Literal["closest","highest"]="closest"
)->pd.DataFrame:
    query_ion_names = np.array(query_ion_names, dtype="U")
    if match_mode == "highest":
        matched_idxes = match_highest_peaks(
            spec_masses, spec_intensities,
            query_mzs=query_masses, 
            query_mz_tols=query_mass_tols,
        )
    else:
        matched_idxes = match_closest_peaks(
            spec_masses, spec_intensities,
            query_mzs=query_masses,
            query_mz_tols=query_mass_tols
        )
    matched_bools = (matched_idxes!=-1)&(query_masses>0)
    matched_query_masses = query_masses[matched_bools]
    matched_frag_idxes = query_frag_idxes[matched_bools]
    matched_ion_names = query_ion_names[matched_bools]
    matched_idxes = matched_idxes[matched_bools]
    matched_spec_masses = spec_masses[matched_idxes]
    matched_spec_intens = spec_intensities[matched_idxes]
    matched_mass_errs = matched_spec_masses-matched_query_masses
    ppm_mass_errs = matched_mass_errs*1e6/matched_query_masses

    spec_df = pd.DataFrame(dict(
        modified_sequence=modified_sequence,
        mz=spec_masses,
        intensity=spec_intensities,
        fragment_site=-1,
        ppm_err=0,
        mass_err=0,
        ion_name="-",
    ))
    matched_df = pd.DataFrame(dict(
        modified_sequence=modified_sequence,
        mz=matched_query_masses,
        intensity=matched_spec_intens,
        fragment_site=matched_frag_idxes,
        ppm_err=ppm_mass_errs,
        mass_err=matched_mass_errs,
        ion_name=matched_ion_names,
    ))

    if query_intensities is None or len(matched_df) == 0:
        mirror_df = pd.DataFrame()
    else:
        query_intensities *= -(
            matched_spec_intens.max()
            /query_intensities[matched_bools].max()
        )
        mirror_df = pd.DataFrame(dict(
            modified_sequence=modified_sequence,
            mz=query_masses,
            intensity=query_intensities,
            fragment_site=-1,
            ppm_err=0,
            mass_err=0,
            ion_name=query_ion_names,
        ))

        def PCC_sim(x, y):
            x = x-np.mean(x)
            y = y-np.mean(y)
            return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-8)
        PCC = PCC_sim(
            -query_intensities[matched_bools],
            matched_spec_intens
        )

    df = pd.concat([
        spec_df, matched_df, mirror_df
    ], ignore_index=True)
    df["mod_sites"] = mod_sites
    if len(mirror_df) > 0:
        df["pcc"] = PCC
    return df

def get_modified_sequence(
    sequence:str, 
    mods:str, mod_sites:str, 
    charge:int = 0
):
    sequence = '_'+sequence+'_'
    mod_masses = np.zeros(len(sequence))
    if mods:
        for mod, mod_site in zip(
            mods.split(';'),mod_sites.split(';')
        ):
            mod_masses[int(mod_site)] += MOD_MASS[mod]
        mod_masses = np.rint(mod_masses)
    mod_seq = sequence
    for i in range(len(sequence)-1,-1,-1):
        if mod_masses[i] != 0:
            mod_seq = (
                mod_seq[:i+1]
                +f"[{mod_masses[i]:+.0f}]"
                +mod_seq[i+1:]
            )
    if charge != 0:
        return mod_seq + f"({charge}+)"
    else:
        return mod_seq

def translate_precursor_fragment_df_to_plot_df(
    precursor_df:pd.DataFrame, 
    fragment_mz_df:pd.DataFrame, 
    fragment_intensity_df:pd.DataFrame = None,
    rt_sec:float = 0.0, 
    mobility:float = 0.0,
    ms_level:int = 2,
    min_frag_mz:float = 100.0,
    min_frag_intensity:float = 0.001,
)->pd.DataFrame:
    fragment_mz_df = fragment_mz_df.mask(
        fragment_mz_df<min_frag_mz, 0
    )
    if fragment_intensity_df is None:
        fragment_intensity_df = pd.DataFrame()

    flat_columns = [
        'type','number','loss_type','charge'
    ]
    precursor_df, fragment_df = flatten_fragments(
        precursor_df, fragment_mz_df, 
        fragment_intensity_df,
        min_fragment_intensity=min_frag_intensity,
        custom_columns=flat_columns+["position"],
    )

    fragment_df.rename(
        columns={"position":"fragment_site"},
        inplace=True
    )

    fragment_df["ion_name"] = fragment_df[flat_columns].apply(
        lambda x: 
            chr(x[0])+str(x[1])
            +(f"{-x[2]:+}" if x[2]!=0 else "")
            +"+"*x[3],
        axis=1
    )

    isotope_names = [col for col in precursor_df.columns if col.startswith("i_")]
    if len(isotope_names) > 0:
        isotope_df = pd.DataFrame(np.zeros(
            (len(isotope_names),len(fragment_df.columns)),
        ), columns=fragment_df.columns)
        isotope_df["type"] = ord('M')
        charge = precursor_df.charge.values[0]
        precursor_mz = precursor_df.precursor_mz.values[0]
        isotope_mzs = []
        ion_names = []
        mono_idx = precursor_df.mono_isotope_idx.values[0]
        for i in range(len(isotope_names)):
            isotope_mzs.append(
                precursor_mz+(i-mono_idx)*MASS_ISOTOPE/charge
            )
            ion_names.append(f"M{i-mono_idx}")
        isotope_df["mz"] = np.array(
            isotope_mzs, dtype=fragment_mz_df.values.dtype
        )
        if "intensity" in fragment_df.columns:
            isotope_df["intensity"] = precursor_df[isotope_names].values.astype(
                fragment_intensity_df.values.dtype
            )
        isotope_df["ion_name"] = ion_names
        isotope_df["fragment_site"] = -1
        fragment_df = pd.concat((fragment_df, isotope_df), ignore_index=True)

    fragment_df["sequence"] = precursor_df.sequence.values[0]
    fragment_df["mods"] = precursor_df.mods.values[0]
    fragment_df["mod_sites"] = precursor_df.mod_sites.values[0]
    fragment_df["precursor_charge"] = precursor_df.charge.values[0]
    fragment_df["modified_sequence"] = get_modified_sequence(
        sequence=precursor_df.sequence.values[0], 
        mods=precursor_df.mods.values[0], 
        mod_sites=precursor_df.mod_sites.values[0], 
        charge=0 if "charge" not in precursor_df.columns 
                 else precursor_df.charge.values[0],
    )
    if rt_sec:
        fragment_df["rt_sec"] = rt_sec
    if mobility:
        fragment_df["im"] = mobility

    if ms_level == 2:
        fragment_df["precursor_mz"] = precursor_df.precursor_mz.values[0]

    return fragment_df
    
def make_precursor_fragment_df(
    sequence: str,
    mods: str,
    mod_sites: str,
    charge: int,
    include_fragments:bool=True,
    charged_frag_types: list = ["b_z1","b_z2","y_z1","y_z2"],
    include_precursor_isotopes:bool=False,
    max_isotope:int = 6,
):
    precursor_df = pd.DataFrame(dict(
        sequence=[sequence],
        mods=[mods],
        mod_sites=[mod_sites],
        charge=charge
    ))
    calc_precursor_mz(precursor_df)
    if include_precursor_isotopes:
        calc_precursor_isotope_intensity(
            precursor_df, max_isotope=max_isotope
        )
    if include_fragments:
        fragment_mz_df = create_fragment_mz_dataframe(
            precursor_df, charged_frag_types
        )
    else:
        precursor_df["frag_start_idx"] = 0
        precursor_df["frag_stop_idx"] = 0
        fragment_mz_df = pd.DataFrame(
            columns=charged_frag_types
        )
    return precursor_df, fragment_mz_df