import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # INSERT YOUR CODE

    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # находим индекс максимального значения в матричном профиле
        idx = int(np.argmax(mp))
        nn_idx = int(mpi[idx])
        dist = float(mp[idx])

        # записываем диссонанс (индекс, расстояние и индекс соседа)
        discords_idx.append(idx)
        discords_dist.append(dist)
        discords_nn_idx.append(nn_idx)

        # создаем зону исключения вокруг найденного диссонанса
        mp = apply_exclusion_zone(mp, idx, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
