import numpy as np

def find_next_pair_index(left_to_vis, f_to_join_by):
    filtered_subset = left_to_vis[(left_to_vis.iloc[:, 0] == f_to_join_by) | (left_to_vis.iloc[:,1] == f_to_join_by)].iloc[:,2]
    if len(filtered_subset) > 0:
        idxmax = filtered_subset.idxmax()
        return idxmax, True
    else:
        return left_to_vis.index[0], False

def get_second_feature(prev_feature, row):
    if row["Feature 1"] == prev_feature:
        return row["Feature 2"]
    return row["Feature 1"]

def get_pd_pairs_values(method, pair):
    pair_key = method.pd_calculator._get_pair_key(pair)
    pair_values = method.pd_calculator.pd_pairs[pair_key].copy()
    if pair_key[0] != pair[0]:
        pair_values["f1_values"], pair_values["f2_values"] = pair_values["f2_values"], pair_values["f1_values"]
        pair_values["pd_values"] = pair_values["pd_values"].T
    return pair_values

def get_pd_dict(pd_calculator, to_vis):
    max_pd = 0
    min_pd = 1
    for i in range(len(to_vis)):
        pair = (to_vis.iloc[i, 0], to_vis.iloc[i, 1])
        pair_key = pd_calculator._get_pair_key(pair)
        pair_values = pd_calculator.pd_pairs[pair_key].copy()
        max_pd = max(np.max(pair_values["pd_values"]), max_pd)
        min_pd = min(np.min(pair_values["pd_values"]), min_pd)
    return min_pd, max_pd 