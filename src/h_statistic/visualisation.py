import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.domain.domain import ONE_VS_ALL
from src.h_statistic.calculation import calculate_h_stat


def ovo_heatmap(ovo, ax):
    ovo_copy = ovo.copy()
    ovo_copy['Feature 1'], ovo_copy['Feature 2'] = ovo_copy['Feature 2'], ovo_copy['Feature 1']
    ovo_all_pairs = pd.concat([ovo, ovo_copy])

    ax.set_title("H-statistic for pairs of features")
    sns.heatmap(ovo_all_pairs.pivot_table("H-statistic", "Feature 1", "Feature 2"), cmap="crest", annot=True, ax=ax)


def visualize_h_stat(model, X: pd.DataFrame, n: int = None, n_highest: int = 5):
    ovo = calculate_h_stat(model, X, n)
    ova = calculate_h_stat(model, X, n, ONE_VS_ALL)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ovo_heatmap(ovo, ax1)
    ova.head(n_highest).plot.bar(x="Feature", y="H-statistic", ylabel="H-statistic",
                                 title=f"Top {n_highest} features with highest h-statistic", ax=ax2)
