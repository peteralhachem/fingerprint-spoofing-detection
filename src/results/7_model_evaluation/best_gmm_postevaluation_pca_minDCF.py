import math

from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.evaluation.BinaryEvaluator import BinaryEvaluator
from src.evaluation.evaluation_results import EvaluationResults
from src.preprocessing.pca import PCA
from src.utilities.arrays import flat_map
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set


def __compute_and_save_results_to(_cv_filename, _ev_filename):
    train_data_matrix, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data_matrix, test_labels = load_fingerprint_spoofing_detection_test_set()

    # create the best config GMM classifiers [1],
    # with the best combinations of #clusters for the two classes
    # and different pca strategies
    true_class_highest_num_clusters = 2
    false_class_highest_num_clusters = 8
    pca_dimensions = [9, 8]


    def create_pca_gmm_classifiers(pca_dim):
        pca_true_class_gmm_lbg_estimator = GmmLbgEstimator(
            true_class_highest_num_clusters,
            min_eigenvalue=0.01
        )
        pca_false_class_gmm_lbg_estimator = GmmLbgEstimator(
            false_class_highest_num_clusters,
            min_eigenvalue=0.01
        )

        x_pca_gmm_classifiers = [
            BinaryGmmClassifier(
                int(math.pow(2, i)), int(math.pow(2, j)),
                pca_true_class_gmm_lbg_estimator, pca_false_class_gmm_lbg_estimator,
                preprocessors=[PCA(pca_dim)]
            )
            for i in [0, 1]
            for j in [2, 3]
        ]
        return x_pca_gmm_classifiers

    pca_gmm_classifiers = flat_map(
        lambda x: x,
        [
            create_pca_gmm_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    working_points = [
        (0.5, 1.0, 10.0)
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data_matrix, train_labels, k, seed=0)

    cv.cross_validation(
        pca_gmm_classifiers,
        working_points=working_points,
    )

    cv_results = cv.results(
        # error_rates=True,
        min_DCFs=True,
        # actual_DCFs=True
    )

    print("* Validated model *")

    # now build and test the configuration over the evaluation set

    pca_gmm_classifiers_ev = flat_map(
        lambda x: x,
        [
            create_pca_gmm_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    evaluator = BinaryEvaluator()
    ev_results = evaluator.train_and_evaluate_models(
        pca_gmm_classifiers_ev,
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
    # * Best GMM classifier [1] with the most promising
    #   combinations of true class clusters (1-2) and false class clusters (4-8), and
    #   different PCA preprocessing strategies;
    #   comparison between validation and evaluation results *

    cv_filename = "7_model_evaluation/best_gmm_postevaluation_pca_minDCF_cv"
    ev_filename = "7_model_evaluation/best_gmm_postevaluation_pca_minDCF_ev"

    # run it just once, then comment it
    # cv_filename, ev_filename = __compute_and_save_results_to(cv_filename, ev_filename)

    # print results
    __load_and_print_results_from(
        cv_filename,
        ev_filename,
    )
