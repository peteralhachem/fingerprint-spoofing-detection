import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.evaluation.BinaryEvaluator import BinaryEvaluator
from src.evaluation.evaluation_results import EvaluationResults
from src.preprocessing.pca import PCA
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set
from src.utilities.kernels import PolynomialKernel

c_values = [0] + [
    math.pow(10.0, i)
    for i in range(-5, 2, 1)    # [10^-5, ..., 10^-1]
]

def __compute_and_save_results_to(_cv_filename, _ev_filename):
    train_data_matrix, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data_matrix, test_labels = load_fingerprint_spoofing_detection_test_set()

    # create the best config SVM classifiers [2],
    # but different values of C hyperparameter

    best_svm_classifiers = [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(8)]
        )
        for c in c_values
    ]

    working_points = [
        (0.5, 1.0, 10.0)
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data_matrix, train_labels, k, seed=0)

    cv.cross_validation(
        best_svm_classifiers,
        working_points=working_points,
    )

    cv_results = cv.results(
        # error_rates=True,
        min_DCFs=True,
        # actual_DCFs=True
    )

    print("* Validated model *")

    # now build and test the configuration over the evaluation set

    best_svm_classifiers_ev = [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(8)]
        )
        for c in c_values
    ]

    evaluator = BinaryEvaluator()
    ev_results = evaluator.train_and_evaluate_models(
        best_svm_classifiers_ev,
        train_data_matrix, train_labels,
        test_data_matrix, test_labels,
        working_points[0]
    )

    return cv_results.save(_cv_filename), ev_results.save(_ev_filename)

def __load_and_plot_results_from(_cv_filename, _ev_filename, save_plot_into=None):
    # load results from files
    cv_results = CrossValidationResults.load(_cv_filename)
    cv_results.print(phase="Validation")

    ev_results = EvaluationResults.load(_ev_filename)
    ev_results.print()

    # plot results
    cv_svm_minDCFs = list(map(lambda x: x[0], cv_results.models_min_DCFs))
    ev_svm_minDCFs = list(map(lambda x: x[0], ev_results.models_min_DCFs))

    plt.figure()
    plt.plot(c_values, cv_svm_minDCFs, label="2D-Poly SVM (validation)")
    plt.plot(c_values, ev_svm_minDCFs, label="2D-Poly SVM (evaluation)")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.title("Best SVM (2D-Poly SVM): C effect - validation vs evaluation")
    plt.grid()
    plt.legend()

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * Best 2D Polynomial kernel SVM classifier [2]
    #   with different C hyperparameter values; comparison between validation
    #   and evaluation results *

    cv_filename = "7_model_evaluation/best_svm_postevaluation_c_minDCF_cv"
    ev_filename = "7_model_evaluation/best_svm_postevaluation_c_minDCF_ev"

    # run it just once, then comment it
    # cv_filename, ev_filename = __compute_and_save_results_to(cv_filename, ev_filename)

    # print results
    __load_and_plot_results_from(
        cv_filename,
        ev_filename,
        save_plot_into="best_svm_postevaluation_c_minDCF.pdf"
    )
