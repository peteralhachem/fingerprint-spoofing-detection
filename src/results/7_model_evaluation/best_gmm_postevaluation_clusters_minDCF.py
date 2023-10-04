import math

from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.evaluation.BinaryEvaluator import BinaryEvaluator
from src.evaluation.evaluation_results import EvaluationResults
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set


def __compute_and_save_results_to(_cv_filename, _ev_filename):
    train_data_matrix, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data_matrix, test_labels = load_fingerprint_spoofing_detection_test_set()

    # create the best config GMM classifiers [1],
    # but different number of clusters for the two classes
    # (no PCA)

    true_class_gmm_lbg_estimator = GmmLbgEstimator(16, min_eigenvalue=0.01)
    false_class_gmm_lbg_estimator = GmmLbgEstimator(16, min_eigenvalue=0.01)

    best_gmm_classifiers = [
        BinaryGmmClassifier(
            true_class_gmm_components=int(math.pow(2, i)),
            false_class_gmm_components=int(math.pow(2, j)),
            true_class_gmm_lbg_estimator=true_class_gmm_lbg_estimator,
            false_class_gmm_lbg_estimator=false_class_gmm_lbg_estimator
        )
        for i in range(5) for j in range(5)
    ]

    working_points = [
        (0.5, 1.0, 10.0)
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data_matrix, train_labels, k, seed=0)

    cv.cross_validation(
        best_gmm_classifiers,
        working_points=working_points,
    )

    cv_results = cv.results(
        # error_rates=True,
        min_DCFs=True,
        # actual_DCFs=True
    )

    print("* Validated model *")

    # now build and test the configuration over the evaluation set

    best_gmm_classifiers_ev = [
        BinaryGmmClassifier(
            true_class_gmm_components=int(math.pow(2, i)),
            false_class_gmm_components=int(math.pow(2, j)),
            true_class_gmm_lbg_estimator=true_class_gmm_lbg_estimator,
            false_class_gmm_lbg_estimator=false_class_gmm_lbg_estimator
        )
        for i in range(5) for j in range(5)
    ]

    evaluator = BinaryEvaluator()
    ev_results = evaluator.train_and_evaluate_models(
        best_gmm_classifiers_ev,
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
    # * Best GMM classifier [1]
    #   with different combinations of true class clusters and false class clusters;
    #   comparison between validation and evaluation results *

    cv_filename = "7_model_evaluation/best_gmm_postevaluation_clusters_minDCF_cv"
    ev_filename = "7_model_evaluation/best_gmm_postevaluation_clusters_minDCF_ev"

    # run it just once, then comment it
    # cv_filename, ev_filename = __compute_and_save_results_to(cv_filename, ev_filename)

    # print results
    __load_and_print_results_from(
        cv_filename,
        ev_filename,
    )
