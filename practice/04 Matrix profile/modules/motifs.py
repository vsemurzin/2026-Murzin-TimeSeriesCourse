import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE

    mp = matrix_profile['mp'].copy()
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # находим индекс минимального значения в матричном профиле
        idx = int(np.argmin(mp))
        nn_idx = int(mpi[idx])
        dist = float(mp[idx])

        # записываем пару (левый, правый) по возрастанию индекса
        left = min(idx, nn_idx)
        right = max(idx, nn_idx)
        motifs_idx.append((left, right))
        motifs_dist.append(dist)

        # создаем зоны исключения вокруг обоих индексов
        mp = apply_exclusion_zone(mp, idx, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, nn_idx, excl_zone, np.inf)    

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
