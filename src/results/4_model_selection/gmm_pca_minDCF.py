import math

from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.utilities.arrays import flat_map
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set


def __save_results_to(_filename):
    data, labels = load_fingerprint_spoofing_detection_train_set()

    pca_dimensions = [9,8]

    true_class_gmm_lbg_estimator = GmmLbgEstimator(16, min_eigenvalue=0.01)
    false_class_gmm_lbg_estimator = GmmLbgEstimator(16, min_eigenvalue=0.01)

    # no PCA
    no_pca_gmm_classifiers = [
        BinaryGmmClassifier(
            int(math.pow(2, i)), int(math.pow(2, j)),
            true_class_gmm_lbg_estimator, false_class_gmm_lbg_estimator
        )
        for i in range(5) for j in range(5)
    ]

    # PCA

    def create_pca_gmm_classifiers(pca_dim):
        pca_true_class_gmm_lbg_estimator = GmmLbgEstimator(16, min_eigenvalue=0.01)
        pca_false_class_gmm_lbg_estimator = GmmLbgEstimator(16, min_eigenvalue=0.01)

        x_pca_gmm_classifiers = [
            BinaryGmmClassifier(
                int(math.pow(2, i)), int(math.pow(2, j)),
                pca_true_class_gmm_lbg_estimator, pca_false_class_gmm_lbg_estimator,
                preprocessors=[PCA(pca_dim)]
            )
            for i in range(5) for j in range(5)
        ]
        return x_pca_gmm_classifiers

    pca_gmm_classifiers = flat_map(
        lambda x: x,
        [
            create_pca_gmm_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    classifiers = no_pca_gmm_classifiers + pca_gmm_classifiers

    working_points = [
        (0.5, 1.0, 10.0),
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        # score_calibration_true_train_priors=[0.1, 0.2, 0.5]
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True
    )

    return cv_results.save(_filename)


def __load_and_print_results_from(_filename):
    # load results from file
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()


if __name__ == "__main__":
    # * Gaussian Mixture Model (GMM) classifiers with different components values for the 2 classes,
    #   for different values of PCA *
    filename = "4_model_selection/gmm_pca_minDCF"

    # filename = __save_results_to(filename)     # you might run it just once
    __load_and_print_results_from(filename)
