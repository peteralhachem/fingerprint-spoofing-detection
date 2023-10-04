import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.preprocessing.quadratic_feature_expansion import QuadraticFeatureExpansion
from src.preprocessing.z_norm import ZNorm
from src.utilities.arrays import flat_map
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

lambdas = [0] + [
    math.pow(10.0, i)
    for i in range(-6, 0, 1)
]

pca_dimensions = [9,8,7,6,5,4]

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # no PCA

    quad_lr_classifiers = [
        BinaryLogisticRegressionClassifier(
            regularization_param=_lambda,
            preprocessors=[QuadraticFeatureExpansion()]
        )
        for _lambda in lambdas
    ]

    znorm_quad_lr_classifiers = [
        BinaryLogisticRegressionClassifier(
            regularization_param=_lambda,
            preprocessors=[ZNorm(), QuadraticFeatureExpansion()]
        )
        for _lambda in lambdas
    ]

    # PCA

    def create_quad_lr_pca_and_znorm_quad_lr_pca_classifiers(pca_dim):
        quad_lr_pca_classifiers = [
            BinaryLogisticRegressionClassifier(
                regularization_param=_lambda,
                preprocessors=[PCA(pca_dim), QuadraticFeatureExpansion()]
            )
            for _lambda in lambdas
        ]

        znorm_quad_lr_pca_classifiers = [
            BinaryLogisticRegressionClassifier(
                regularization_param=_lambda,
                preprocessors=[ZNorm(), PCA(pca_dim), QuadraticFeatureExpansion()]
            )
            for _lambda in lambdas
        ]

        return quad_lr_pca_classifiers + znorm_quad_lr_pca_classifiers

    quad_lr_pca_and_znorm_classifiers = flat_map(
        lambda x: x,
        [
            create_quad_lr_pca_and_znorm_quad_lr_pca_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    classifiers = quad_lr_classifiers + znorm_quad_lr_classifiers + quad_lr_pca_and_znorm_classifiers

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

def __load_and_plot_results_from(_filename, save_plot_into=None):
    # load results from file
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()

    # plot results
    minDCFs = cv_results.models_min_DCFs
    plt.figure(figsize=(10,7))

    # plot results with no pca
    quad_lr_minDCFs = list(map(lambda x: x[0], minDCFs[0:len(lambdas)]))
    znorm_quad_lr_minDCFs = list(map(lambda x: x[0], minDCFs[len(lambdas):2*len(lambdas)]))
    plt.plot(lambdas, quad_lr_minDCFs, label="Q-Log-Reg")
    plt.plot(lambdas, znorm_quad_lr_minDCFs, label="Q-Log-Reg (Z-norm)")

    for i,pca_dim in enumerate(pca_dimensions):
        i*=2
        # retrieve minDCFs
        quad_lr_pca_minDCFs = list(map(lambda x: x[0], minDCFs[(i+2)*len(lambdas):(i+3)*len(lambdas)]))
        znorm_quad_lr_pca_minDCFs = list(map(lambda x: x[0], minDCFs[(i+3)*len(lambdas):(i+4)*len(lambdas)]))

        # plot curves
        plt.plot(lambdas, quad_lr_pca_minDCFs, label="Q-Log-Reg (PCA=%d)" % pca_dim)
        plt.plot(lambdas, znorm_quad_lr_pca_minDCFs, label="Q-Log-Reg (PCA=%d, Z-norm)" % pca_dim)

    plt.xscale("log")
    plt.xlabel("$\\lambda$")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - Quadratic Logistic Regression - PCA + Z-norm effect")
    plt.grid()
    plt.legend(loc="upper left")

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * Quadratic Logistic Regression (Q-LR) Classifiers with different lambda hyperparameter values,
    #   with (or without) z-normalization preprocessing, for different values of PCA *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/quad_lr_znorm_PCA_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(
        filename,
        # save_plot_into="quad_lr_znorm_PCA_minDCF.pdf"
    )
