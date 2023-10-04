from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.evaluation.BinaryEvaluator import BinaryEvaluator
from src.evaluation.evaluation_results import EvaluationResults
from src.preprocessing.pca import PCA
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set


def __compute_and_save_results_to(_cv_filename, _ev_filename):
    train_data_matrix, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data_matrix, test_labels = load_fingerprint_spoofing_detection_test_set()

    # create the best config GMM classifiers [1],
    # with the best combinations of #clusters for the two classes
    # best pca strategy, and different gaussian variants for the two clases
    true_class_best_num_clusters = 1
    false_class_best_num_clusters = 4
    pca_dim = 8

    best_gmm_classifiers = [
        BinaryGmmClassifier(
            true_class_best_num_clusters,
            false_class_best_num_clusters,
            GmmLbgEstimator(
                true_class_best_num_clusters,
                min_eigenvalue=0.01,
                tied_covariance=tied_covariance_T,
                diagonal_covariance=diagonal_covariance_T
            ),
            GmmLbgEstimator(
                false_class_best_num_clusters,
                min_eigenvalue=0.01,
                tied_covariance=tied_covariance_F,
                diagonal_covariance=diagonal_covariance_F
            ),
            preprocessors=[PCA(pca_dim)]
        )
        # all combinations of TC and DC, for the two classes
        for diagonal_covariance_T in [False,True] for tied_covariance_T in [False,True]
        for diagonal_covariance_F in [False,True] for tied_covariance_F in [False,True]
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
            true_class_best_num_clusters,
            false_class_best_num_clusters,
            GmmLbgEstimator(
                true_class_best_num_clusters,
                min_eigenvalue=0.01,
                tied_covariance=tied_covariance_T,
                diagonal_covariance=diagonal_covariance_T
            ),
            GmmLbgEstimator(
                false_class_best_num_clusters,
                min_eigenvalue=0.01,
                tied_covariance=tied_covariance_F,
                diagonal_covariance=diagonal_covariance_F
            ),
            preprocessors=[PCA(pca_dim)]
        )
        # all combinations of TC and DC, for the two classes
        for diagonal_covariance_T in [False, True] for tied_covariance_T in [False, True]
        for diagonal_covariance_F in [False, True] for tied_covariance_F in [False, True]
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
    # * Best GMM classifier [1] with the most promising
    #   combinations of true class clusters and false class clusters, the best
    #   PCA preprocessing strategy, and different GMM variants for the 2 classes;
    #   comparison between validation and evaluation results *

    cv_filename = "7_model_evaluation/best_gmm_postevaluation_variants_minDCF_cv"
    ev_filename = "7_model_evaluation/best_gmm_postevaluation_variants_minDCF_ev"

    # run it just once, then comment it
    cv_filename, ev_filename = __compute_and_save_results_to(cv_filename, ev_filename)

    # print results
    __load_and_print_results_from(
        cv_filename,
        ev_filename,
    )
