import logging

import numpy as np
import pandas as pd

from alpharaw.feature.chem import averagine_aa, isotopes, maximum_offset
from alpharaw.feature.hills import (
    extract_hills,
    filter_hills,
    get_hill_data,
    remove_duplicate_hills,
    split_hills,
)
from alpharaw.feature.isotope_pattern import (
    feature_finder_report,
    get_isotope_patterns,
    get_pre_isotope_patterns,
    get_stats,
)


def find(
    spectrum_df,
    peak_df,
    centroid_tol=8,
    max_gap=2,
    hill_length_min=3,
    hill_split_level=1.3,
    iso_split_level=1.3,
    hill_smoothing=1,
    hill_check_large=40,
    iso_charge_min=1,
    iso_charge_max=6,
    iso_n_seeds=100,
    hill_nboot_max=300,
    hill_nboot=150,
    iso_mass_range=5,
    iso_corr_min=0.6,
) -> pd.DataFrame:
    logging.info("FF started")
    query_data = df_to_ap_query_data(spectrum_df, peak_df)

    int_data = np.array(query_data["int_list_ms1"])

    logging.info(
        f"Hill extraction with centroid_tol {centroid_tol} and max_gap {max_gap}"
    )

    hill_ptrs, hill_data, path_node_cnt, score_median, score_std = extract_hills(
        query_data, max_gap, centroid_tol
    )
    logging.info(
        f"Number of hills {len(hill_ptrs):,}, len = {np.mean(path_node_cnt):.2f}"
    )

    logging.info(
        f"Repeating hill extraction with centroid_tol {score_median+score_std*3:.2f}"
    )

    hill_ptrs, hill_data, path_node_cnt, score_median, score_std = extract_hills(
        query_data, max_gap, score_median + score_std * 3
    )
    logging.info(
        f"Number of hills {len(hill_ptrs):,}, len = {np.mean(path_node_cnt):.2f}"
    )

    hill_ptrs, hill_data = remove_duplicate_hills(hill_ptrs, hill_data, path_node_cnt)
    logging.info(f"After duplicate removal of hills {len(hill_ptrs):,}")

    hill_ptrs = split_hills(
        hill_ptrs,
        hill_data,
        int_data,
        hill_split_level=hill_split_level,
        window=hill_smoothing,
    )  # hill lenght is inthere already
    logging.info(f"After split hill_ptrs {len(hill_ptrs):,}")

    hill_data, hill_ptrs = filter_hills(
        hill_data,
        hill_ptrs,
        int_data,
        hill_check_large=hill_check_large,
        window=hill_smoothing,
    )

    logging.info(f"After filter hill_ptrs {len(hill_ptrs):,}")

    stats, sortindex_, idxs_upper, scan_idx, hill_data, hill_ptrs = get_hill_data(
        query_data,
        hill_ptrs,
        hill_data,
        hill_nboot_max=hill_nboot_max,
        hill_nboot=hill_nboot,
    )
    logging.info("Extracting hill stats complete")

    pre_isotope_patterns = get_pre_isotope_patterns(
        stats,
        idxs_upper,
        sortindex_,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        maximum_offset,
        iso_charge_min=iso_charge_min,
        iso_charge_max=iso_charge_max,
        iso_mass_range=iso_mass_range,
        cc_cutoff=iso_corr_min,
    )
    logging.info(f"Found {len(pre_isotope_patterns):,} pre isotope patterns.")

    isotope_patterns, iso_idx, isotope_charges = get_isotope_patterns(
        pre_isotope_patterns,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        stats,
        sortindex_,
        averagine_aa,
        isotopes,
        iso_charge_min=iso_charge_min,
        iso_charge_max=iso_charge_max,
        iso_mass_range=iso_mass_range,
        iso_n_seeds=iso_n_seeds,
        cc_cutoff=iso_corr_min,
        iso_split_level=iso_split_level,
        callback=None,
    )
    logging.info(f"Extracted {len(isotope_charges):,} isotope patterns.")

    feature_df, lookup_idx = feature_finder_report(
        query_data,
        isotope_patterns,
        isotope_charges,
        iso_idx,
        stats,
        sortindex_,
        hill_ptrs,
        hill_data,
    )

    # lookup_idx_df = pd.DataFrame(lookup_idx, columns = ['isotope_pattern', 'isotope_pattern_hill'])

    feature_isotopes_df = get_stats(isotope_patterns, iso_idx, stats)

    feature_df["last_isotope_mz"] = (
        feature_isotopes_df.groupby("feature_id").tail(1).mz_average.values
    )

    logging.info("FF finished.")

    return feature_df


def df_to_ap_query_data(spec_df, peak_df):
    query_data = {}

    query_data["scan_list_ms1"] = spec_df.spec_idx.values + 1
    query_data["rt_list_ms1"] = spec_df.rt.values
    query_data["mass_list_ms1"] = peak_df.mz.values
    query_data["int_list_ms1"] = peak_df.intensity.values
    indices = np.zeros(len(spec_df) + 1, dtype=np.int64)
    indices[1:] = spec_df.peak_stop_idx.values
    query_data["indices_ms1"] = indices
    return query_data
