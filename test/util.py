import random

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

CALIFORNIA_SUBSET = ["Longitude", "Latitude", "MedInc", "AveRooms"]
SAMPLE_SIZE = 5
N_REPEAT = 3
N = 100


def california_housing_random_forest(max_depth: int = 6, n_estimators: int = 25):
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators).fit(X, y)
    return model, X, y

def california_housing_boosting_models():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    model_xgb = XGBRegressor(n_estimators=10, max_depth=4).fit(X, y)
    model_lgbm = LGBMRegressor(n_estimators=10, max_depth=4).fit(X, y)
    model_xgb_bis = XGBRegressor(n_estimators=40, max_depth=8).fit(X.iloc[:,:3], y)
    model_lgbm_bis = LGBMRegressor(n_estimators=40, max_depth=8).fit(X.iloc[:,:3], y)
    return model_xgb, model_lgbm, model_xgb_bis, model_lgbm_bis, X, y


def toy_input():
    target = list(range(N))
    X = pd.DataFrame(
        {
            "important_feature": target,
            "noise_feature": [1 for _ in range(N)]
        }
    )
    y = target
    model = RandomForestRegressor().fit(X, y)

    return model, X, y


def has_decreasing_order(vals):
    return all(earlier >= later for earlier, later in zip(vals, vals[1:]))
