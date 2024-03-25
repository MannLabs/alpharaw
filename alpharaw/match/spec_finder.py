import numpy as np
import numba

def find_multinotch_spec_idxes(
    spec_rts: np.ndarray,
    spec_multinotch_wins:list,
    spec_ms_levels: np.ndarray,
    query_start_rt: float,
    query_stop_rt: float,
    query_left_mz:float,
    query_right_mz:float,
)->np.ndarray:
    start_idx = np.searchsorted(spec_rts, query_start_rt)
    stop_idx = np.searchsorted(spec_rts, query_stop_rt)+1
    spec_idxes = []
    for ispec in range(start_idx, stop_idx):
        for win_left, win_right in spec_multinotch_wins[ispec]:
            if spec_ms_levels[ispec] == 1:
                if query_left_mz <= 0:
                    spec_idxes.append(ispec)
            elif max(query_left_mz, win_left) <= min(query_right_mz, win_right):
                spec_idxes.append(ispec)
    return np.array(spec_idxes)


@numba.njit    
def find_dia_spec_idxes_same_window(
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


@numba.njit    
def find_spec_idxes(
    spec_rts:np.ndarray, 
    spec_isolation_lower_mzs:np.ndarray, 
    spec_isolation_upper_mzs:np.ndarray,
    query_start_rt:float, 
    query_stop_rt:float,
    query_left_mz:float,
    query_right_mz:float,
):
    rt_start_idx = np.searchsorted(spec_rts, query_start_rt)
    rt_stop_idx = np.searchsorted(spec_rts, query_stop_rt)+1
    
    spec_idxes = []

    for ispec in range(rt_start_idx, rt_stop_idx):
        if (
            max(query_left_mz,spec_isolation_lower_mzs[ispec]) <=
            min(query_right_mz,spec_isolation_upper_mzs[ispec])
        ):
            spec_idxes.append(ispec)
    return np.array(spec_idxes)

@numba.njit    
def find_batch_spec_idxes(
    spec_rts:np.ndarray, 
    spec_isolation_lower_mzs:np.ndarray, 
    spec_isolation_upper_mzs:np.ndarray,
    query_start_rts:np.ndarray, 
    query_stop_rts:np.ndarray,
    query_left_mzs:np.ndarray,
    query_right_mzs:np.ndarray,
    max_spec_per_query:int,
):
    rt_start_idxes = np.searchsorted(spec_rts, query_start_rts)
    rt_stop_idxes = np.searchsorted(spec_rts, query_stop_rts)+1
    
    spec_idxes = np.full(
        (len(query_left_mzs),max_spec_per_query),
        -1, dtype=np.int32
    )
    for iquery in range(len(rt_start_idxes)):
        idx_list = []
        for ispec in range(rt_start_idxes[iquery], rt_stop_idxes[iquery]):
            if (
                max(query_left_mzs[iquery],spec_isolation_lower_mzs[ispec]) <=
                min(query_right_mzs[iquery],spec_isolation_upper_mzs[ispec])
            ):
                idx_list.append(ispec)
        if len(idx_list) > max_spec_per_query:
            spec_idxes[iquery,:] = idx_list[
                len(idx_list)/2-max_spec_per_query//2:
                len(idx_list)/2+max_spec_per_query//2+1
            ]
        else:
            spec_idxes[iquery,:len(idx_list)] = idx_list
    return spec_idxes