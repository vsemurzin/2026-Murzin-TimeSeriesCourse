import numpy as np
from typing import Self
from modules.metrics import *
from modules.utils import z_normalize


default_metrics_params = {'euclidean': {'normalize': True},
                         'dtw': {'normalize': True, 'r': 0.05}
                         }

class TimeSeriesKNN:
    """
    KNN Time Series Classifier

    Parameters
    ----------
    n_neighbors: number of neighbors
    metric: distance measure between time series
             Options: {euclidean, dtw}
    metric_params: dictionary containing parameters for the distance metric being used
    """
    
    def __init__(self, n_neighbors: int = 3, metric: str = 'euclidean', metric_params: dict | None = None) -> None:

        self.n_neighbors: int = n_neighbors
        self.metric: str = metric
        self.metric_params: dict | None = default_metrics_params[metric].copy()
        if metric_params is not None:
            self.metric_params.update(metric_params)


    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> Self:
        """
        Fit the model using X_train as training data and Y_train as labels

        Parameters
        ----------
        X_train: train set with shape (ts_number, ts_length)
        Y_train: labels of the train set
        
        Returns
        -------
        self: the fitted model
        """
       
        self.X_train = X_train
        self.Y_train = Y_train

        return self


    def _distance(self, x_train: np.ndarray, x_test: np.ndarray) -> float:
        """
        Compute distance between the train and test samples
        
        Parameters
        ----------
        x_train: sample of the train set
        x_test: sample of the test set
        
        Returns
        -------
        dist: distance between the train and test samples
        """

        dist = 0

        # INSERT YOUR CODE

        params = self.metric_params
        normalize = params.get('normalize', False)

        if self.metric == 'euclidean':
            if normalize:
                return norm_ED_distance(x_train, x_test)
            else:
                return ED_distance(x_train, x_test)

        elif self.metric == 'dtw':
            if normalize:
                return DTW_distance(z_normalize(x_train), z_normalize(x_test))
            else:
                return DTW_distance(x_train, x_test)


    def _find_neighbors(self, x_test: np.ndarray) -> list[tuple[float, int]]:
        """
        Find the k nearest neighbors of the test sample

        Parameters
        ----------
        x_test: sample of the test set
        
        Returns
        -------
        neighbors: k nearest neighbors (distance between neighbor and test sample, neighbor label) for test sample
        """

        neighbors = []

        # INSERT YOUR CODE

        neighbors = []

        # 1. Считаем расстояния до всех train-объектов
        for i in range(len(self.X_train)):
            dist = self._distance(self.X_train[i], x_test)
            neighbors.append((dist, int(self.Y_train[i])))

        # 2. Сортируем по расстоянию
        neighbors.sort(key=lambda pair: pair[0])

        # 3. Берём k ближайших
        neighbors = neighbors[:self.n_neighbors]

        return neighbors


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for samples of the test set

        Parameters
        ----------
        X_test: test set with shape (ts_number, ts_length))

        Returns
        -------
        y_pred: class labels for each data sample from test set
        """

        y_pred = []

        # INSERT YOUR CODE

        for x_test in X_test:
            neighbors = self._find_neighbors(x_test)

            # Достаём метки соседей
            labels = [label for _, label in neighbors]

            # Находим наиболее часто встречающуюся метку
            # np.unique с return_counts=True даёт уникальные значения и их частоты
            unique, counts = np.unique(labels, return_counts=True)
            most_common = unique[np.argmax(counts)]

            y_pred.append(most_common)

        return np.array(y_pred)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy classification score

    Parameters
    ----------
    y_true: ground truth (correct) labels
    y_pred: predicted labels returned by a classifier

    Returns
    -------
    score: accuracy classification score
    """

    score = 0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            score = score + 1
    score = score/len(y_true)

    return score
