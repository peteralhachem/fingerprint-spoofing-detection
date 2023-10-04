import math

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.preprocessing.quadratic_feature_expansion import QuadraticFeatureExpansion
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set
from src.utilities.kernels import PolynomialKernel


def __compute_1_2_3_fusion_results():
    data, labels = load_fingerprint_spoofing_detection_train_set()

    classifiers = [
        # [1]
        BinaryGmmClassifier(
            true_class_gmm_components=1,
            false_class_gmm_components=4,
            true_class_gmm_lbg_estimator=GmmLbgEstimator(1, min_eigenvalue=0.01),
            false_class_gmm_lbg_estimator=GmmLbgEstimator(4, min_eigenvalue=0.01)
        ),
        # [2]
        SvmClassifier(
            c=math.pow(10.0, -2),
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(8)]
        ),
        # [3]
        BinaryLogisticRegressionClassifier(
            regularization_param=math.pow(10.0, -4),
            rebalancing_true_prior_probability=0.2,
            preprocessors=[PCA(6), QuadraticFeatureExpansion()]
        )
    ]

    working_points = [
        (0.5, 1.0, 10.0),
    ]

    score_calibration_true_priors = [
        0.2
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True,
    )

    return cv_results

def __compute_1_2_fusion_results():
    data, labels = load_fingerprint_spoofing_detection_train_set()

    classifiers = [
        # [1]
        BinaryGmmClassifier(
            true_class_gmm_components=1,
            false_class_gmm_components=4,
            true_class_gmm_lbg_estimator=GmmLbgEstimator(1, min_eigenvalue=0.01),
            false_class_gmm_lbg_estimator=GmmLbgEstimator(4, min_eigenvalue=0.01)
        ),
        # [2]
        SvmClassifier(
            c=math.pow(10.0, -2),
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(8)]
        ),
    ]

    working_points = [
        (0.5, 1.0, 10.0),
    ]

    score_calibration_true_priors = [
        0.2
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True,
    )

    return cv_results

def __compute_1_3_fusion_results():
    data, labels = load_fingerprint_spoofing_detection_train_set()

    classifiers = [
        # [1]
        BinaryGmmClassifier(
            true_class_gmm_components=1,
            false_class_gmm_components=4,
            true_class_gmm_lbg_estimator=GmmLbgEstimator(1, min_eigenvalue=0.01),
            false_class_gmm_lbg_estimator=GmmLbgEstimator(4, min_eigenvalue=0.01)
        ),
        # [3]
        BinaryLogisticRegressionClassifier(
            regularization_param=math.pow(10.0, -4),
            rebalancing_true_prior_probability=0.2,
            preprocessors=[PCA(6), QuadraticFeatureExpansion()]
        )
    ]

    working_points = [
        (0.5, 1.0, 10.0),
    ]

    score_calibration_true_priors = [
        0.2
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True,
    )

    return cv_results

def __compute_2_3_fusion_results():
    data, labels = load_fingerprint_spoofing_detection_train_set()

    classifiers = [
        # [2]
        SvmClassifier(
            c=math.pow(10.0, -2),
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(8)]
        ),
        # [3]
        BinaryLogisticRegressionClassifier(
            regularization_param=math.pow(10.0, -4),
            rebalancing_true_prior_probability=0.2,
            preprocessors=[PCA(6), QuadraticFeatureExpansion()]
        )
    ]

    working_points = [
        (0.5, 1.0, 10.0),
    ]

    score_calibration_true_priors = [
        0.2
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True,
    )

    return cv_results

def __save_results_to(_filename):
    cv_results_1_2_3 = __compute_1_2_3_fusion_results()
    cv_results_1_2 = __compute_1_2_fusion_results()
    cv_results_1_3 = __compute_1_3_fusion_results()
    cv_results_2_3 = __compute_2_3_fusion_results()

    # remove redundant single classifiers results
    cv_results_1_2.delete_results_of_classifiers([0, 1])  # keep just [1] + [2]
    cv_results_1_3.delete_results_of_classifiers([0, 1])  # keep just [1] + [3]
    cv_results_2_3.delete_results_of_classifiers([0, 1])  # keep just [2] + [3]

    # merge results
    cv_results = (cv_results_1_2_3
                  .merge(cv_results_1_2)
                  .merge(cv_results_1_3)
                  .merge(cv_results_2_3))

    # save them into file
    return cv_results.save(_filename)


def __load_and_print_results_from(_filename):
    # load results from files
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()


if __name__ == "__main__":
    # * results of the fusion of the best 3 calibrated models [1] + [2] + [3]:
    #   min DCF, actual DCF and error rate *
    filename = "6_model_calibration/best_3_models_fusion_actDCF_minDCF"

    # filename = __save_results_to(filename)  # you might want to run it just once
    __load_and_print_results_from(filename)
