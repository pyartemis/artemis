from itertools import combinations
from typing import Dict, Callable, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from artemis.utilities.domain import ProgressInfoLog

from artemis.utilities.ops import get_predict_function


class PartialDependenceCalculator: 
    def __init__(self, model, X: pd.DataFrame, predict_function: Optional[Callable] = None, batchsize: int = 2000):
        self.model = model
        self.predict_function = get_predict_function(model) if predict_function is None else predict_function
        self.X = X
        self.X_len = len(self.X)
        self.batchsize = batchsize
        self.pd_single = {col: 
                            {   
                                "f_values": np.sort(np.unique(self.X[col].values)), 
                                "pd_values": np.full(len(np.unique(self.X[col].values)), np.nan) 
                            }
                            for col in self.X.columns}
        self.pd_pairs = {(col1, col2): 
                            {   
                                "f1_values": np.sort(np.unique(self.X[col1].values)), 
                                "f2_values": np.sort(np.unique(self.X[col2].values)),
                                "pd_values": np.full((len(np.unique(self.X[col1].values)), len(np.unique(self.X[col2].values))), np.nan)
                            }
                            for col1, col2 in combinations(self.X.columns, 2)}
        self.pd_minus_single = {col:
                            {   
                                "row_ids": np.arange(self.X_len),
                                "pd_values": np.full(self.X_len, np.nan) 
                            }
                            for col in self.X.columns}

    def get_pd_single(self, feature: str, feature_values: Optional[List[Any]] = None) -> np.ndarray:
        if feature_values is None:
            return self.pd_single[feature]["pd_values"]
        selected_values = np.zeros(len(feature_values))
        for i, feature_value in enumerate(feature_values):
            f_index = get_index(self.pd_single[feature]["f_values"], feature_value)
            selected_values[i] = self.pd_single[feature]["pd_values"][f_index]
        return selected_values

    def get_pd_pairs(self, feature1: str, feature2: str, feature_values: Optional[List[Tuple[Any, Any]]] = None) -> np.ndarray:
        pair_key = get_pair_key((feature1, feature2), self.pd_pairs.keys())
        all_matrix = self.pd_pairs[pair_key]["pd_values"]
        if feature_values is None:
            return all_matrix
        if pair_key != (feature1, feature2):
            feature_values = reorder_pair_values(feature_values)
        selected_values = np.zeros(len(feature_values))
        for i, pair in enumerate(feature_values):
            f1_index = get_index(self.pd_pairs[pair_key]["f1_values"], pair[0])
            f2_index = get_index(self.pd_pairs[pair_key]["f2_values"], pair[1])
            selected_values[i] = all_matrix[f1_index, f2_index]
        return selected_values
    
    def get_pd_minus_single(self, feature: str) -> np.ndarray:
        return self.pd_minus_single[feature]["pd_values"]

    def calculate_pd_single(self, features: Optional[List[str]] = None, show_progress: bool = False, desc: str = ProgressInfoLog.CALC_VAR_IMP):
        if features is None:
            features = self.X.columns
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        for feature in tqdm(features, desc=desc, disable=not show_progress):
            if np.isnan(self.pd_single[feature]["pd_values"]).any(): 
                for value in self.pd_single[feature]["f_values"]:
                    change_dict = {feature: value}
                    X_changed = self.X.copy().assign(**change_dict)
                    range_dict[(feature, value)] = (current_len, current_len+self.X_len)
                    current_len += self.X_len
                    X_full = pd.concat((X_full, X_changed))
                if current_len > self.batchsize:
                    self.fill_pd_single(range_dict, X_full)
                    current_len = 0
                    range_dict = {}
                    X_full = pd.DataFrame()
        if current_len > 0:
            self.fill_pd_single(range_dict, X_full)
    
    def calculate_pd_pairs(self, feature_pairs = None, all_combinations=True, show_progress: bool = False, desc: str = ProgressInfoLog.CALC_OVO):
        if feature_pairs is None:
            feature_pairs = self.pd_pairs.keys()
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        for feature1, feature2 in tqdm(feature_pairs, desc=desc, disable=not show_progress):
            feature1, feature2 = get_pair_key((feature1, feature2), self.pd_pairs.keys())
            if all_combinations:
                feature_values = [(f1, f2) for f1 in self.pd_pairs[(feature1, feature2)]["f1_values"] for f2 in self.pd_pairs[(feature1, feature2)]["f2_values"]]
            else:
                feature_values = zip(self.X[feature1].values, self.X[feature2].values)         
            for value1, value2 in feature_values:
                f1_ind = get_index(self.pd_pairs[(feature1, feature2)]["f1_values"], value1)
                f2_ind = get_index(self.pd_pairs[(feature1, feature2)]["f2_values"], value2)
                if np.isnan(self.pd_pairs[(feature1, feature2)]["pd_values"][f1_ind, f2_ind]): 
                    change_dict = {feature1: value1, feature2: value2}
                    X_changed = self.X.copy().assign(**change_dict)
                    range_dict[(feature1, feature2, value1, value2)] = (current_len, current_len+self.X_len)
                    current_len += self.X_len
                    X_full = pd.concat((X_full, X_changed))
                    if current_len > self.batchsize:
                        self.fill_pd_pairs(range_dict, X_full)
                        current_len = 0
                        range_dict = {}
                        X_full = pd.DataFrame()
        if current_len > 0:
            self.fill_pd_pairs(range_dict, X_full)
    
    def calculate_pd_minus_single(self, features: Optional[List[str]] = None, show_progress: bool = False, desc: str = ProgressInfoLog.CALC_OVA):
        if features is None:
            features = self.X.columns
        range_dict = {}
        current_len = 0
        X_full = pd.DataFrame()
        for feature in tqdm(features, desc=desc, disable=not show_progress):
            if np.isnan(self.pd_minus_single[feature]["pd_values"]).any(): 
                for i, row in self.X.copy().reset_index(drop=True).iterrows():
                    change_dict = {other_feature: row[other_feature] for other_feature in self.X.columns if other_feature != feature}
                    X_changed = self.X.copy().assign(**change_dict)
                    range_dict[(feature, i)] = (current_len, current_len+self.X_len)
                    current_len += self.X_len
                    X_full = pd.concat((X_full, X_changed))
                if current_len > self.batchsize:
                    self.fill_pd_minus_single(range_dict, X_full)
                    current_len = 0
                    range_dict = {}
                    X_full = pd.DataFrame()
        if current_len > 0:
            self.fill_pd_minus_single(range_dict, X_full)

    def fill_pd_single(self, range_dict, X_full): 
        y = self.predict_function(self.model, X_full)
        for var_name, var_val in range_dict.keys():
            start, end = range_dict[(var_name, var_val)]
            value_index = get_index(self.pd_single[var_name]["f_values"], var_val)
            self.pd_single[var_name]["pd_values"][value_index] = np.mean(y[start:end])

    def fill_pd_pairs(self, range_dict, X_full):
        y = self.predict_function(self.model, X_full)
        for var_name1, var_name2, var_val1, var_val2 in range_dict.keys():
            start, end = range_dict[(var_name1, var_name2, var_val1, var_val2)]
            value_index1 = get_index(self.pd_pairs[(var_name1, var_name2)]["f1_values"], var_val1)
            value_index2 = get_index(self.pd_pairs[(var_name1, var_name2)]["f2_values"], var_val2)
            self.pd_pairs[(var_name1, var_name2)]["pd_values"][value_index1, value_index2] = np.mean(y[start:end])

    def fill_pd_minus_single(self, range_dict, X_full): 
        y = self.predict_function(self.model, X_full)
        for var_name, row_id in range_dict.keys():
            start, end = range_dict[(var_name, row_id)]
            self.pd_minus_single[var_name]["pd_values"][row_id] = np.mean(y[start:end])

def get_index(array, value) -> int:
    return np.where(array == value)[0][0]

def reorder_pair_values(pair_values: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    return [(pair[1], pair[0]) for pair in pair_values]

def get_pair_key(pair: Tuple[str, str], keys: List[Tuple[str, str]]) -> Tuple[str, str]:
    if pair in keys:
        return pair
    else:
        return (pair[1], pair[0])

