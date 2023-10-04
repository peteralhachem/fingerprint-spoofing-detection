import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

lambdas = [0] + [
    math.pow(10.0, i)
    for i in range(-8, 2, 1)
]

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    lr_classifiers = [
        BinaryLogisticRegressionClassifier(
            regularization_param=_lambda
        )
        for _lambda in lambdas
    ]

    z_norm_lr_classifiers = [
        BinaryLogisticRegressionClassifier(
            regularization_param=_lambda,
            preprocessors=[ZNorm()]
        )
        for _lambda in lambdas
    ]

    classifiers = lr_classifiers + z_norm_lr_classifiers

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
        # error_rates=True,
        min_DCFs=True,
        # actual_DCFs=True
    )

    return cv_results.save(_filename)

def __load_and_plot_results_from(_filename, save_plot_into=None):
    # load results from file
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()

    # plot results
    minDCFs = cv_results.models_min_DCFs
    lr_minDCFs = list(map(lambda x: x[0], minDCFs[0:len(lambdas)]))
    znorm_lr_minDCFs = list(map(lambda x: x[0], minDCFs[len(lambdas):2*len(lambdas)]))

    plt.figure()
    plt.plot(lambdas, lr_minDCFs, label="Log-Reg")
    plt.plot(lambdas, znorm_lr_minDCFs, label="Log-Reg (Z-norm)")
    plt.xscale("log")
    plt.xlabel("$\\lambda$")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - Logistic Regression - Z-norm effect")
    plt.grid()
    plt.legend()

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * Logistic Regression (LR) Classifiers with different lambda hyperparameter values,
    #   with (or without) z-normalization preprocessing *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/lr_znorm_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(filename, save_plot_into="lr_znorm_minDCF.pdf")
