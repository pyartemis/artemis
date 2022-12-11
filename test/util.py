import random

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

CALIFORNIA_SUBSET = ["Longitude", "Latitude", "MedInc", "AveRooms"]
WINE_SUBSET = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash"]
SAMPLE_SIZE = 5
N_REPEAT = 3
N = 100


def california_housing_random_forest(max_depth: int = 6, n_estimators: int = 80):
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators).fit(X, y)
    return model, X, y


def wine_random_forest(max_depth: int = 6, n_estimators: int = 80):
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators).fit(X, y)
    return model, X, y


def california_housing_boosting_models():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    model_xgb = XGBRegressor(n_estimators=10, max_depth=4).fit(X, y)
    model_lgbm = LGBMRegressor(n_estimators=10, max_depth=4).fit(X, y)
    model_xgb_bis = XGBRegressor(n_estimators=40, max_depth=8).fit(X.iloc[:, :3], y)
    model_lgbm_bis = LGBMRegressor(n_estimators=40, max_depth=8).fit(X.iloc[:, :3], y)
    return model_xgb, model_lgbm, model_xgb_bis, model_lgbm_bis, X, y


def wine_boosting_models():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)[:N]
    y = wine.target[:N]
    model_xgb = XGBClassifier(n_estimators=10, max_depth=4).fit(X, y)
    model_lgbm = LGBMClassifier(n_estimators=10, max_depth=4).fit(X, y)
    model_xgb_bis = XGBClassifier(n_estimators=40, max_depth=8).fit(X.iloc[:, :3], y)
    model_lgbm_bis = LGBMClassifier(n_estimators=40, max_depth=8).fit(X.iloc[:, :3], y)
    return model_xgb, model_lgbm, model_xgb_bis, model_lgbm_bis, X, y


def california_housing_linear_regression():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    model = LinearRegression().fit(X, y)
    return model, X, y


def wine_random_logistic_regression():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    model = LogisticRegression().fit(X, y)
    return model, X, y


def toy_input_reg():
    target = list(range(N))
    X = pd.DataFrame({"important_feature": target, "noise_feature": [1 for _ in range(N)]})
    y = target
    model = RandomForestRegressor().fit(X, y)

    return model, X, y

def toy_input_cls():
    target = [0, 1] * (N // 2)
    X = pd.DataFrame({"important_feature": target, "noise_feature": [1 for _ in range(len(target))]})
    y = target
    model = RandomForestClassifier().fit(X, y)

    return model, X, y


def has_decreasing_order(vals):
    return all(earlier >= later for earlier, later in zip(vals, vals[1:]))
