import numpy as np
from src.preprocessing.preprocess_class import UnsupervisedPreprocessor
from src.utilities.arrays import vcol


class QuadraticFeatureExpansion(UnsupervisedPreprocessor):
    """
    Quadratic feature expansion preprocessor
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "2D feature exp."

    def preprocess(self, data_matrix):
        super().preprocess(data_matrix)
        num_features = data_matrix.shape[0]
        num_samples = data_matrix.shape[1]

        expanded_features_samples = []

        for i in range(num_samples):
            sample = vcol(data_matrix[:, i])

            feature_cov = np.matmul(sample, sample.T)

            expanded_features_sample = np.vstack(
                [feature_cov[:, j:j + 1] for j in range(num_features)] +
                [sample]
            )  # 2-D column array

            expanded_features_samples.append(expanded_features_sample)

        expanded_features_matrix = np.hstack(expanded_features_samples)  # stack samples in a matrix

        return expanded_features_matrix
