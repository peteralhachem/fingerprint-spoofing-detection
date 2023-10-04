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
from src.utilities.target_application import TargetApplication


def __save_results_to(_filename):
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

    prior_log_odds = list(range(-4, 5, 1))  # [-4, 4]

    score_calibration_true_priors = [0.2]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        prior_log_odds=prior_log_odds,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True,
        bayes_error_plot_values=True
    )

    return cv_results.save(_filename)

def __load_print_results_and_plot_bayes_error_plot_from(_filename, save_to=None):
    # load results from file
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()

    # plot
    cv_results.bayes_error_plot(
        labels=["GMM", "2D-Poly SVM", "Q-LR"],
        save_to=save_to,
        ylim=(0, 0.6),
        xlabel="log$\\frac{\\pi}{1-\\pi}$",
        ylabel="DCF and $DCF_{min}$",
        application_prior=TargetApplication.application_prior_log_odd()
    )


if __name__ == "__main__":
    # * Best 3 calibrated models Bayes error plot *
    filename = "6_model_calibration/best_3_models_calibrated_bayes_error_plot"

    # filename = __save_results_to(filename)  # you might run it just once
    __load_print_results_and_plot_bayes_error_plot_from(
        filename,
        save_to="best_3_models_calibrated_bayes_error_plot.pdf"
    )
