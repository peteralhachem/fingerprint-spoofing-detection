import math

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


def __compute_and_save_results_to(_cv_filename, _ev_filename):
    train_data_matrix, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data_matrix, test_labels = load_fingerprint_spoofing_detection_test_set()

    # create the best config SVM classifiers [2], with C=x
    # but different preprocessing strategies of PCA and Z-norm

    pca_dimensions = [9,8,7,6]
    c = math.pow(10.0, -2)   # chosen with a greedy approach

    # pca

    pca_best_svm_classifiers = [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
        )
    ] + [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[PCA(pca_dim)]
        )
        for pca_dim in pca_dimensions
    ]

    # z-norm + pca

    pca_znorm_best_svm_classifiers = [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm()]
        )
    ] + [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(pca_dim)]
        )
        for pca_dim in pca_dimensions
    ]

    classifiers = pca_best_svm_classifiers + pca_znorm_best_svm_classifiers

    working_points = [
        (0.5, 1.0, 10.0)
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data_matrix, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
    )

    cv_results = cv.results(
        # error_rates=True,
        min_DCFs=True,
        # actual_DCFs=True
    )

    print("* Validated model *")

    # now build and test the configuration over the evaluation set

    pca_best_svm_classifiers_ev = [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
        )
    ] + [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[PCA(pca_dim)]
        )
        for pca_dim in pca_dimensions
    ]

    # z-norm + pca

    pca_znorm_best_svm_classifiers_ev = [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm()]
        )
    ] + [
        SvmClassifier(
            c=c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm(), PCA(pca_dim)]
        )
        for pca_dim in pca_dimensions
    ]

    classifiers_ev = pca_best_svm_classifiers_ev + pca_znorm_best_svm_classifiers_ev

    evaluator = BinaryEvaluator()
    ev_results = evaluator.train_and_evaluate_models(
        classifiers_ev,
        train_data_matrix, train_labels,
        test_data_matrix, test_labels,
        working_points[0]
    )

    return cv_results.save(_cv_filename), ev_results.save(_ev_filename)

def __load_and_print_results_from(_cv_filename, _ev_filename):
    # load results from files
    cv_results = CrossValidationResults.load(_cv_filename)
    cv_results.print(phase="Validation")

    ev_results = EvaluationResults.load(_ev_filename)
    ev_results.print()


if __name__ == "__main__":
    # * Best 2D Polynomial kernel SVM classifier [2]
    #   with different PCA and Z-Norm strategies; comparison between validation
    #   and evaluation results *

    cv_filename = "7_model_evaluation/best_svm_postevaluation_pca_znorm_minDCF_cv"
    ev_filename = "7_model_evaluation/best_svm_postevaluation_pca_znorm_minDCF_ev"

    # run it just once, then comment it
    # cv_filename, ev_filename = __compute_and_save_results_to(cv_filename, ev_filename)

    # print results
    __load_and_print_results_from(
        cv_filename,
        ev_filename
    )
