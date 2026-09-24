import numpy as np


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

    n = len(ts1)

    # Скалярное произведение
    dot_product = np.dot(ts1, ts2)

    # Средние
    mu1 = np.mean(ts1)
    mu2 = np.mean(ts2)

    # Стандартные отклонения
    sigma1 = np.sqrt(np.mean(ts1 ** 2) - mu1 ** 2)
    sigma2 = np.sqrt(np.mean(ts2 ** 2) - mu2 ** 2)

    # Внутреннее выражение
    inner = 2 * n * (1 - (dot_product - n * mu1 * mu2) / (n * sigma1 * sigma2))

    # Модуль и корень
    norm_ed_dist = np.sqrt(np.abs(inner))

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

    # Инициализируем матрицу d размера (n+1) x (m+1)
    # d[0][0] = 0, d[i][0] = d[0][j] = inf
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0

    # Заполняем матрицу
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Евклидово расстояние (в квадрате) между элементами
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2

            # Минимум из 3 соседних ячеек
            d[i, j] = cost + min(
                d[i - 1, j],    # вертикальное движение
                d[i, j - 1],    # горизонтальное
                d[i - 1, j - 1] # диагональное
            )
    
    dtw_dist = d[n, m]

    return dtw_dist