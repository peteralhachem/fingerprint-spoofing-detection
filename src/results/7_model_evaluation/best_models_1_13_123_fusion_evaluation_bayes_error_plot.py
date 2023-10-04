import math

import numpy as np

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.classifiers.fusion_classifier import BinaryFusionClassifier
from src.classifiers.generative_models.gmm.binary_gmm_classifier import BinaryGmmClassifier
from src.classifiers.generative_models.gmm.gmm_estimation import GmmLbgEstimator
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.evaluation.evaluation_results import EvaluationResults
from src.preprocessing.pca import PCA
from src.preprocessing.quadratic_feature_expansion import QuadraticFeatureExpansion
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set, \
    load_fingerprint_spoofing_detection_test_set
from src.utilities.kernels import PolynomialKernel
from src.utilities.target_application import TargetApplication


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

    working_point_for_calibration = (0.5, 1.0, 10.0)


    score_calibration_true_priors = [0.2]

    # perform Cross Validation (over train set) to train the score calibrator
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=[working_point_for_calibration],
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

    prior_log_odds = list(range(-4, 5, 1))  # [-4, 4]

    eval_results_1_2_3 = __train_and_evaluate_calibrated_models_for_bayes_plot(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        prior_log_odds
    )

    return eval_results_1_2_3


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

    working_point_for_calibration = (0.5, 1.0, 10.0)

    score_calibration_true_priors = [0.2]

    # perform Cross Validation (over train set) to train the score calibrator
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(train_data, train_labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=[working_point_for_calibration],
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

    prior_log_odds = list(range(-4, 5, 1))  # [-4, 4]

    eval_results_1_3 = __train_and_evaluate_calibrated_models_for_bayes_plot(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        prior_log_odds
    )

    return eval_results_1_3



def __save_results_to(_filename):
    # compute the evaluation results for the best models
    eval_results_1_3 = __compute_1_3_fusion_eval_results()
    eval_results_1_2_3 = __compute_1_2_3_fusion_eval_results()

    # remove redundant classifiers results
    eval_results_1_3.delete_results_of_classifiers([1])  # remove [3], keep [1] and [1]+[3]
    eval_results_1_2_3.delete_results_of_classifiers([0, 1, 2])  # keep just [1]+[2]+[3]

    # merge results
    eval_results = eval_results_1_3.merge(eval_results_1_2_3)

    # save them into file
    return eval_results.save(_filename)

def __train_and_evaluate_calibrated_models_for_bayes_plot(
        calibrated_models,
        train_data, train_labels,
        test_data, test_labels,
        prior_log_odds
):

    # * now train these 4 (calibrated) models over the train set,
    #   make them predict the test set labels
    #   and evaluate the results over the test set *

    models_per_working_point = []
    corresponding_pi_Ts = [1.0 / (1.0 + np.exp(-x)) for x in prior_log_odds]

    for model in calibrated_models:
        print("\n- %s" % model)
        print("Training...")
        model.train(train_data, train_labels)
        print("Predicting labels and evaluating results...")

        wp_models = []

        for corresponding_pi_T in corresponding_pi_Ts:
            model_clone = model.clone()

            model_clone.predict(test_data, corresponding_pi_T)
            model_clone.evaluate(test_labels, global_stats=True)

            wp_models.append(model_clone)

        models_per_working_point.append(wp_models)


    print("\n* Computing error rates, min DCFs and actual DCFs *")

    def compute_error_rate(model, pi_T, C_fn, C_fp):
        print("- (error rate) %s and working point (%.3f, %.3f, %.3f)" % (str(model), pi_T, C_fn, C_fp))
        return model.error_rate

    def compute_min_DCF(model, pi_T, C_fn, C_fp):
        print("- (min DCF) %s and working point (%.3f, %.3f, %.3f)" % (str(model), pi_T, C_fn, C_fp))
        return model.min_DCF(
            true_prior_probability=pi_T,
            error_costs=(C_fn, C_fp)
        )

    def compute_actual_DCF(model, pi_T, C_fn, C_fp):
        print("- (actual DCF) %s and working point (%.3f, %.3f, %.3f)" % (str(model), pi_T, C_fn, C_fp))
        return model.DCF(
            true_prior_probability=pi_T,
            error_costs=(C_fn, C_fp)
        )

    models_error_rates = [
        [
            compute_error_rate(model, pi_T, 1.0, 1.0)
            for model, pi_T in zip(wp_models, corresponding_pi_Ts)
        ]
        for wp_models in models_per_working_point
    ]

    models_min_DCFs = [
        [
            compute_min_DCF(model, pi_T, 1.0, 1.0)
            for model, pi_T in zip(wp_models, corresponding_pi_Ts)
        ]
        for wp_models in models_per_working_point
    ]

    models_actual_DCFs = [
        [
            compute_actual_DCF(model, pi_T, 1.0, 1.0)
            for model, pi_T in zip(wp_models, corresponding_pi_Ts)
        ]
        for wp_models in models_per_working_point
    ]

    models_actual_DCFs_per_working_point = np.vstack(models_actual_DCFs)
    models_min_DCFs_per_working_point = np.vstack(models_min_DCFs)

    eval_results = EvaluationResults(
        results={
            "models_error_rates": models_error_rates,
            "models_min_DCFs": models_min_DCFs,
            "models_actual_DCFs": models_actual_DCFs,
            "bayes_error_plot_values": [
                {
                    "prior_log_odds": prior_log_odds,
                    "actual_DCFs": list(models_actual_DCFs_per_working_point[i, :]),  # actual DCFs of i-th model
                    "min_DCFs": list(models_min_DCFs_per_working_point[i, :])  # min DCFs of i-th model
                }
                for i in range(len(models_per_working_point))
            ]
        },
        classifiers=list(map(lambda x: str(x), calibrated_models)),
        working_points=[(x, 1.0, 1.0) for x in corresponding_pi_Ts]
    )

    return eval_results


def __load_and_print_results_from(_filename, save_to=None):
    # load results from file
    eval_results = EvaluationResults.load(_filename)
    eval_results.print()

    eval_results.bayes_error_plot(
        labels=["[1]", "[1]+[3]", "[1]+[2]+[3]"],
        save_to=save_to,
        ylim=(0, 0.6),
        xlabel="log$\\frac{\\pi}{1-\\pi}$",
        ylabel="DCF and $DCF_{min}$",
        application_prior=TargetApplication.application_prior_log_odd()
    )


if __name__ == "__main__":
    # * Bayes error plot for the 3 best selected (fusion) model: [1]+[3], [1]+[2]+[3], [1]:
    #   min DCF, actual DCF and error rate *
    filename = "7_model_evaluation/best_models_1_13_123_fusion_evaluation_bayes_error_plot"

    # filename = __save_results_to(filename)  # you might want to run it just once
    __load_and_print_results_from(
        filename,
        # save_to="best_models_1_13_123_fusion_evaluation_bayes_error_plot.pdf"
    )
