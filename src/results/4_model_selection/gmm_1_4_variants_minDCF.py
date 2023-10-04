from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set


def __save_results_to(_filename):
    data, labels = load_fingerprint_spoofing_detection_train_set()

    classifiers = [
        BinaryGmmClassifier(
            1,
            4,
            GmmLbgEstimator(
                1,
                min_eigenvalue=0.01,
                tied_covariance=tied_covariance_T,
                diagonal_covariance=diagonal_covariance_T
            ),
            GmmLbgEstimator(
                4,
                min_eigenvalue=0.01,
                tied_covariance=tied_covariance_F,
                diagonal_covariance=diagonal_covariance_F
            )
        )
        # all combinations of TC and DC, for the two classes
        for diagonal_covariance_T in [False,True] for tied_covariance_T in [False,True]
        for diagonal_covariance_F in [False,True] for tied_covariance_F in [False,True]
    ]

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
    # * Gaussian Mixture Model (GMM) classifiers with different 1 component for the true class,
    #   and 4 components for the false class, without PCA, trying all the different gaussian variants
    #   for the false class *
    filename = "4_model_selection/gmm_1_4_variants_minDCF"

    filename = __save_results_to(filename)     # you might run it just once
    __load_and_print_results_from(filename)
