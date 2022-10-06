import numpy as np
from sklearn.inspection import partial_dependence


def find_index(pd_jk, column, value):
    return np.where(pd_jk['values'][column] == value)[0][0]


# naive execution of partial dependence
def partial_dependence_i_versus(model, X, i, versus):
    pd_i_versus = partial_dependence(model, X, [i] + versus, grid_resolution=len(X) + 1)
    pd_i = partial_dependence(model, X, [i], grid_resolution=len(X) + 1)
    pd_versus = partial_dependence(model, X, versus, grid_resolution=len(X) + 1)

    return pd_i, pd_versus, pd_i_versus
