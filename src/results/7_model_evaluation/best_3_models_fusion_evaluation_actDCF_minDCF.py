import math

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.classifiers.fusion_classifier import BinaryFusionClassifier
from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.evaluation.BinaryEvaluator import BinaryEvaluator
from src.evaluation.evaluation_results import EvaluationResults
from src.preprocessing.pca import PCA
from src.preprocessing.quadratic_feature_expansion import QuadraticFeatureExpansion
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set
from src.utilities.kernels import PolynomialKernel


def __compute_1_2_3_fusion_eval_results():
    train_data, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data, test_labels = load_fingerprint_spoofing_detection_test_set()

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

    score_calibration_true_priors = [0.2]

    # perform Cross Validation (over train set) to train the score calibrator
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    # retrieve score calibrators for each of the 4 models
    trained_calibrator_for_1 = cv.calibrators_per_train_prior_and_working_point[0][0]
    trained_calibrator_for_2 = cv.calibrators_per_train_prior_and_working_point[1][0]
    trained_calibrator_for_3 = cv.calibrators_per_train_prior_and_working_point[2][0]
    trained_calibrator_for_1_2_3 = cv.calibrators_per_train_prior_and_working_point[3][0]

    # plug the score calibrators into the corresponding models

    # model [1]
    calibrated_model_1 = BinaryGmmClassifier(
        true_class_gmm_components=1,
        false_class_gmm_components=4,
        true_class_gmm_lbg_estimator=GmmLbgEstimator(1, min_eigenvalue=0.01),
        false_class_gmm_lbg_estimator=GmmLbgEstimator(4, min_eigenvalue=0.01),
        score_calibrators=[trained_calibrator_for_1]
    )

    # model [2]
    calibrated_model_2 = SvmClassifier(
        c=math.pow(10.0, -2),
        kernel=PolynomialKernel(2),
        preprocessors=[ZNorm(), PCA(8)],
        score_calibrators=[trained_calibrator_for_2]
    )

    # model [3]
    calibrated_model_3 = BinaryLogisticRegressionClassifier(
        regularization_param=math.pow(10.0, -4),
        rebalancing_true_prior_probability=0.2,
        preprocessors=[PCA(6), QuadraticFeatureExpansion()],
        score_calibrators=[trained_calibrator_for_3]
    )

    # model [1]+[2]+[3]
    calibrated_model_1_2_3 = BinaryFusionClassifier(
        classifiers=[
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
        ],
        score_calibrators=[trained_calibrator_for_1_2_3]
    )

    calibrated_models = [
        calibrated_model_1,
        calibrated_model_2,
        calibrated_model_3,
        calibrated_model_1_2_3
    ]

    print("* Models calibrated *")

    evaluator = BinaryEvaluator()
    eval_results_1_2_3 = evaluator.train_and_evaluate_models(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        working_points[0]
    )

    return eval_results_1_2_3

def __compute_1_2_fusion_eval_results():
    train_data, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data, test_labels = load_fingerprint_spoofing_detection_test_set()

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
        )
    ]

    working_points = [
        (0.5, 1.0, 10.0),
    ]

    score_calibration_true_priors = [0.2]

    # perform Cross Validation (over train set) to train the score calibrator
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    # retrieve score calibrators for each of the 3 models
    trained_calibrator_for_1 = cv.calibrators_per_train_prior_and_working_point[0][0]
    trained_calibrator_for_2 = cv.calibrators_per_train_prior_and_working_point[1][0]
    trained_calibrator_for_1_2 = cv.calibrators_per_train_prior_and_working_point[2][0]

    # plug the score calibrators into the corresponding models

    # model [1]
    calibrated_model_1 = BinaryGmmClassifier(
        true_class_gmm_components=1,
        false_class_gmm_components=4,
        true_class_gmm_lbg_estimator=GmmLbgEstimator(1, min_eigenvalue=0.01),
        false_class_gmm_lbg_estimator=GmmLbgEstimator(4, min_eigenvalue=0.01),
        score_calibrators=[trained_calibrator_for_1]
    )

    # model [2]
    calibrated_model_2 = SvmClassifier(
        c=math.pow(10.0, -2),
        kernel=PolynomialKernel(2),
        preprocessors=[ZNorm(), PCA(8)],
        score_calibrators=[trained_calibrator_for_2]
    )

    # model [1]+[2]
    calibrated_model_1_2 = BinaryFusionClassifier(
        classifiers=[
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
        ],
        score_calibrators=[trained_calibrator_for_1_2]
    )

    calibrated_models = [
        calibrated_model_1,
        calibrated_model_2,
        calibrated_model_1_2
    ]

    print("* Models calibrated *")

    evaluator = BinaryEvaluator()
    eval_results_1_2 = evaluator.train_and_evaluate_models(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        working_points[0]
    )

    return eval_results_1_2

def __compute_1_3_fusion_eval_results():
    train_data, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data, test_labels = load_fingerprint_spoofing_detection_test_set()

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

    score_calibration_true_priors = [0.2]

    # perform Cross Validation (over train set) to train the score calibrator
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    # retrieve score calibrators for each of the 3 models
    trained_calibrator_for_1 = cv.calibrators_per_train_prior_and_working_point[0][0]
    trained_calibrator_for_3 = cv.calibrators_per_train_prior_and_working_point[1][0]
    trained_calibrator_for_1_3 = cv.calibrators_per_train_prior_and_working_point[2][0]

    # plug the score calibrators into the corresponding models

    # model [1]
    calibrated_model_1 = BinaryGmmClassifier(
        true_class_gmm_components=1,
        false_class_gmm_components=4,
        true_class_gmm_lbg_estimator=GmmLbgEstimator(1, min_eigenvalue=0.01),
        false_class_gmm_lbg_estimator=GmmLbgEstimator(4, min_eigenvalue=0.01),
        score_calibrators=[trained_calibrator_for_1]
    )

    # model [3]
    calibrated_model_3 = BinaryLogisticRegressionClassifier(
        regularization_param=math.pow(10.0, -4),
        rebalancing_true_prior_probability=0.2,
        preprocessors=[PCA(6), QuadraticFeatureExpansion()],
        score_calibrators=[trained_calibrator_for_3]
    )

    # model [1]+[3]
    calibrated_model_1_3 = BinaryFusionClassifier(
        classifiers=[
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
        ],
        score_calibrators=[trained_calibrator_for_1_3]
    )

    calibrated_models = [
        calibrated_model_1,
        calibrated_model_3,
        calibrated_model_1_3
    ]

    print("* Models calibrated *")

    evaluator = BinaryEvaluator()
    eval_results_1_3 = evaluator.train_and_evaluate_models(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        working_points[0]
    )

    return eval_results_1_3

def __compute_2_3_fusion_eval_results():
    train_data, train_labels = load_fingerprint_spoofing_detection_train_set()
    test_data, test_labels = load_fingerprint_spoofing_detection_test_set()

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

    score_calibration_true_priors = [0.2]

    # perform Cross Validation (over train set) to train the score calibrator
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
        # * score calibration *
        score_calibration_true_train_priors=score_calibration_true_priors,
        compute_fusion_model=True
    )

    # retrieve score calibrators for each of the 4 models
    trained_calibrator_for_2 = cv.calibrators_per_train_prior_and_working_point[0][0]
    trained_calibrator_for_3 = cv.calibrators_per_train_prior_and_working_point[1][0]
    trained_calibrator_for_2_3 = cv.calibrators_per_train_prior_and_working_point[2][0]

    # plug the score calibrators into the corresponding models

    # model [2]
    calibrated_model_2 = SvmClassifier(
        c=math.pow(10.0, -2),
        kernel=PolynomialKernel(2),
        preprocessors=[ZNorm(), PCA(8)],
        score_calibrators=[trained_calibrator_for_2]
    )

    # model [3]
    calibrated_model_3 = BinaryLogisticRegressionClassifier(
        regularization_param=math.pow(10.0, -4),
        rebalancing_true_prior_probability=0.2,
        preprocessors=[PCA(6), QuadraticFeatureExpansion()],
        score_calibrators=[trained_calibrator_for_3]
    )

    # model [2]+[3]
    calibrated_model_2_3 = BinaryFusionClassifier(
        classifiers=[
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
        ],
        score_calibrators=[trained_calibrator_for_2_3]
    )

    calibrated_models = [
        calibrated_model_2,
        calibrated_model_3,
        calibrated_model_2_3
    ]

    print("* Models calibrated *")

    evaluator = BinaryEvaluator()
    eval_results_2_3 = evaluator.train_and_evaluate_models(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        working_points[0]
    )

    return eval_results_2_3


def __save_results_to(_filename):
    # compute the evaluation results for the single and the fusion models
    eval_results_1_2_3 = __compute_1_2_3_fusion_eval_results()
    eval_results_1_2 = __compute_1_2_fusion_eval_results()
    eval_results_1_3 = __compute_1_3_fusion_eval_results()
    eval_results_2_3 = __compute_2_3_fusion_eval_results()

    # remove redundant single classifiers results
    eval_results_1_2.delete_results_of_classifiers([0, 1])  # keep just [1] + [2]
    eval_results_1_3.delete_results_of_classifiers([0, 1])  # keep just [1] + [3]
    eval_results_2_3.delete_results_of_classifiers([0, 1])  # keep just [2] + [3]

    # merge results
    eval_results = (eval_results_1_2_3
                    .merge(eval_results_1_2)
                    .merge(eval_results_1_3)
                    .merge(eval_results_2_3))

    # save them into file
    return eval_results.save(_filename)


def __load_and_print_results_from(_filename):
    # load results from file
    eval_results = EvaluationResults.load(_filename)
    eval_results.print()


if __name__ == "__main__":
    # * Evaluation results of the fusion of the best 3 calibrated models [1] + [2] + [3]:
    #   min DCF, actual DCF and error rate *
    filename = "7_model_evaluation/best_3_models_fusion_evaluation_actDCF_minDCF"

    # filename = __save_results_to(filename)  # you might want to run it just once
    __load_and_print_results_from(filename)
