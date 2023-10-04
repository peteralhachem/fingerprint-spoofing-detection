import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.preprocessing.pca import PCA
from src.preprocessing.quadratic_feature_expansion import QuadraticFeatureExpansion
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set
from src.utilities.kernels import PolynomialKernel


def best_3_models_DET_plot(save_to=None):
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

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
    )

    results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True,
    )

    # print results
    results.print()

    # retrieved used models
    best_gmm = cv.models_per_working_point[0][0]
    best_2d_svm = cv.models_per_working_point[1][0]
    best_q_lr = cv.models_per_working_point[2][0]

    # compute DET plots
    x_det_gmm, y_det_gmm = best_gmm.DET_plot()
    x_det_2d_svm, y_det_2d_svm = best_2d_svm.DET_plot()
    x_det_q_lr, y_det_q_lr = best_q_lr.DET_plot()

    # plot them
    plt.figure(figsize=(9,6))
    plt.plot(x_det_gmm, y_det_gmm, label="GMM")
    plt.plot(x_det_2d_svm, y_det_2d_svm, label="2D-Poly SVM")
    plt.plot(x_det_q_lr, y_det_q_lr, label="Q-LR")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.legend(loc="upper right")

    if save_to is not None:
        plt.savefig(save_to)

    plt.show()


if __name__ == "__main__":
    # * print the results of the best 3 models and plot their DET plot *

    best_3_models_DET_plot(save_to="best_3_models_det_plot.pdf")
