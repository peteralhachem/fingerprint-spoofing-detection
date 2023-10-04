import numpy as np
from src.preprocessing.preprocess_class import UnsupervisedPreprocessor
from src.utilities.statistics import mean_and_covariance_of
import matplotlib.pyplot as plt


class PCA(UnsupervisedPreprocessor):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.labels = None
        self.eigenvalues = None
        self.components_matrix = None

    def __str__(self):
        return "%d-PCA" % self.n_components

    def fit(self, data_matrix):
        super().fit(data_matrix)

        self.mean, self.covariance = mean_and_covariance_of(data_matrix)
        self.eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)

        self.components_matrix = eigenvectors[:, ::-1][:, :self.n_components]

    def preprocess(self, data_matrix):
        super().preprocess(data_matrix)

        transformed_data = np.dot(self.components_matrix.T, data_matrix)
        return transformed_data

    def plot_explained_variance_pca(self):
        self.eigenvalues = self.eigenvalues[::-1]

        fractions = self.eigenvalues / np.sum(self.eigenvalues)
        cumulated_fraction_variances = np.cumsum(fractions)

        x_array = np.arange(1, 11)
        plt.plot(x_array, cumulated_fraction_variances, 'bo')
        plt.plot(x_array, cumulated_fraction_variances)

        plt.xlabel("PCA dimensions")
        plt.ylabel("Fraction of explained variance")
        plt.title("Explained Variance")
        plt.grid(True)

        plt.show()
