import numpy as np
from src.preprocessing.preprocess_class import UnsupervisedPreprocessor, SupervisedPreprocessor
from src.utilities.statistics import mean_and_covariance_of
from scipy.linalg import eigh


class LDA(SupervisedPreprocessor):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.labels = None
        self.between_covariance = None
        self.within_covariance = None
        self.W = None

    def preprocess(self, data_matrix, labels):
        super().preprocess(data_matrix, labels)
        self.between_covariance = 0
        self.within_covariance = 0

        for label in np.unique(self.labels):
            total_mean = mean_and_covariance_of(self.data_matrix[:, self.labels == label])[0] - \
                         mean_and_covariance_of(self.data_matrix)[0]
            variable = np.dot(total_mean, total_mean.T) * (self.data_matrix[:, self.labels == label].shape[1])
            self.between_covariance += variable

            new_value = (data_matrix[:, self.labels == label] -
                         mean_and_covariance_of(data_matrix[:, self.labels == label])[0])

            within_covariance_value = mean_and_covariance_of(new_value)[1]

            self.within_covariance += (within_covariance_value * self.data_matrix[:, self.labels == label].shape[1])

        self.between_covariance = self.between_covariance / self.data_matrix.shape[1]
        self.within_covariance = self.within_covariance / self.data_matrix.shape[1]

        # ---Generalized eigenvalue problem ---#
        eigenvalues, eigenvectors = eigh(self.between_covariance, self.within_covariance)

        self.W = eigenvectors[:, ::-1][:, 0:self.n_components]

        x_centered = self.data_matrix - mean_and_covariance_of(self.data_matrix)[0]
        x_transformed = np.dot(self.W.T, x_centered)

        return x_transformed
