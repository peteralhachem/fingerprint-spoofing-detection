import numpy as np
from src.preprocessing.preprocess_class import UnsupervisedPreprocessor
from src.utilities.statistics import mean_and_covariance_of


class ZNorm(UnsupervisedPreprocessor):
    """
    Z-normalization preprocessor
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Z-norm"

    def fit(self, data_matrix):
        super().fit(data_matrix)

        self.mean = mean_and_covariance_of(data_matrix)[0]

        self.std = np.std(data_matrix, axis=1, keepdims=True)

    def preprocess(self, data_matrix):
        super().preprocess(data_matrix)
        centered_data = data_matrix - self.mean

        z = centered_data / self.std
        return z
