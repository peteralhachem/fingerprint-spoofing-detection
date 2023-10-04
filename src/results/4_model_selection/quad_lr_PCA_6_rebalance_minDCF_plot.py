import math

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.preprocessing.quadratic_feature_expansion import QuadraticFeatureExpansion
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

_lambda = math.pow(10, -4)
pca_dimension = 6
rebalancing_priors = [0.05, 1.0/11.0, 0.2, 0.5]

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # no class rebalancing
    quad_lr_pca_6_classifier = BinaryLogisticRegressionClassifier(
        regularization_param=_lambda,
        preprocessors=[PCA(pca_dimension), QuadraticFeatureExpansion()]
    )

    # class rebalancing
    quad_lr_pca_6_rebalanced_classifiers = [
        BinaryLogisticRegressionClassifier(
            regularization_param=_lambda,
            rebalancing_true_prior_probability=pi_T,
            preprocessors=[PCA(pca_dimension), QuadraticFeatureExpansion()]
        )
        for pi_T in rebalancing_priors
    ]

    classifiers = [quad_lr_pca_6_classifier] + quad_lr_pca_6_rebalanced_classifiers

    working_points = [
        (0.5, 1.0, 10.0)
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data_matrix, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
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
    # * Quadratic Logistic Regression (Q-LR) Classifiers with lambda=10^-4,
    #   PCA=6 and different class rebalancing priors *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/quad_lr_PCA_6_rebalance_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_print_results_from(filename)
