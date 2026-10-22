import numpy as np
import math


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0

    # INSERT YOUR CODE
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))
    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    # INSERT YOUR CODE

    n = len(ts1)
    m = len(ts2)

    # Преобразуем долю r в количество ячеек
    # Если r >= 1, то ограничения нет: полоса на всю ширину
    if r >= 1:
        r_cells = max(n, m)  # Чтобы покрыть всю матрицу
    else:
        r_cells = math.ceil(r * max(n, m))

    # Заполняем матрицу расстояний бесконечностями
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        # Задаем границы полосы для j
        j_start = max(1, i - r_cells)
        j_end = min(m, i + r_cells)

        for j in range(j_start, j_end + 1):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            # Минимум из трех соседних ячеек
            min_prev = min(dtw_matrix[i - 1, j],
                           dtw_matrix[i, j - 1],
                           dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + min_prev

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist
