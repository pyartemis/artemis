from re import search
import pandas as pd

from artemis.utilities.exceptions import ModelNotSupportedException


class GBTreesHandler:
    def __init__(self, model = None) -> None:
        if model is not None:
            self.unify_structure(model)

    def unify_structure(self, model) -> None:
        model_class = search("(?<=<class ').*(?='>)", str(type(model)))[0]
        self.package = model_class.split(".")[0]
        if self.package not in _SUPPORTED_PACKAGES:
            raise ModelNotSupportedException(self.package, model_class)
        if hasattr(model, "get_booster"):
            model = model.get_booster()
        if hasattr(model, "booster_"):
            model = model.booster_
        self.trees_df = _get_unified_trees_df(model, self.package)


def _get_unified_trees_df(model, package: str) -> pd.DataFrame:
    trees_df = model.trees_to_dataframe()
    if package == "xgboost":
        trees_df = trees_df.rename(columns=_xgboost_col_dict).drop(
            columns=["Split", "Missing", "Category"]
        )
        trees_df["leaf"] = trees_df["split_feature"] == "Leaf"
        trees_df["depth"] = None

    elif package == "lightgbm":
        trees_df = trees_df.rename(columns=_lightgbm_col_dict).drop(
            columns=[
                "threshold",
                "decision_type",
                "missing_direction",
                "missing_type",
                "value",
                "weight",
                "count",
                "parent_index",
            ]
        )
        trees_df["leaf"] = trees_df["split_feature"].values == None
        trees_df["depth"] = trees_df["depth"] - 1
        trees_df["cover"] = None
    return trees_df


# handler config
_xgboost_col_dict = {
    "Tree": "tree",
    "Node": "node",
    "Feature": "split_feature",
    "Yes": "left_child",
    "No": "right_child",
    "Gain": "gain",
    "Cover": "cover",
}

_lightgbm_col_dict = {
    "tree_index": "tree",
    "node_depth": "depth",
    "node_index": "ID",
    "split_gain": "gain",
}

_SUPPORTED_PACKAGES = ["xgboost", "lightgbm"]
