import torch
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dict_to_dot(x):
    ret = []

    for key, value in x.items():
        assert isinstance(key, str)
        if isinstance(value, dict):
            child_ret = dict_to_dot(value)
            ret.extend([(f'{key}.{child_dotstring}', child_value)
                        for child_dotstring, child_value in child_ret])
        else:
            ret.append((key, value))

    return ret


def reverse_sorting(x, argsort):
    x_unsorted = np.empty_like(x)
    x_unsorted[argsort] = x

    return x_unsorted


def swap_entries(A, entry_1, entry_2):
    assert entry_1 < entry_2
    X = A.copy()
    row_1 = X[entry_1].copy()
    X[entry_1] = X[entry_2]
    X[entry_2] = row_1

    col_1 = X[:, entry_1].copy()
    X[:, entry_1] = X[:, entry_2]
    X[:, entry_2] = col_1

    return X
