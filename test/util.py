import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

CALIFORNIA_SUBSET = ["Longitude", "Latitude", "MedInc", "AveRooms"]
SAMPLE_SIZE = 5
N_REPEAT = 3


def california_housing_random_forest():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    model = RandomForestRegressor().fit(X, y)

    return model, X, y


def has_decreasing_order(vals):
    return all(earlier >= later for earlier, later in zip(vals, vals[1:]))
