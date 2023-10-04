from src.evaluation.evaluation_results import EvaluationResults


class BinaryEvaluator:
    def train_and_evaluate_models(
            self,
            calibrated_models,
            train_data, train_labels,
            test_data, test_labels,
            working_point
    ):
        pi_T, C_fn, C_fp = working_point

        # * now train these 4 (calibrated) models over the train set,
        #   make them predict the test set labels
        #   and evaluate the results over the test set *

        for model in calibrated_models:
            print("\n- %s" % model)
            print("Training...")
            model.train(train_data, train_labels)
            print("Predicting labels...")
            model.predict(test_data, pi_T, (C_fn, C_fp))
            print("Evaluating results...")
            model.evaluate(test_labels, global_stats=True)

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

        eval_results = EvaluationResults(
            results={
                "models_error_rates": [
                    [
                        compute_error_rate(model, pi_T, C_fn, C_fp)
                    ]
                    for model in calibrated_models
                ],
                "models_min_DCFs": [
                    [
                        compute_min_DCF(model, pi_T, C_fn, C_fp)
                    ]
                    for model in calibrated_models
                ],
                "models_actual_DCFs": [
                    [
                        compute_actual_DCF(model, pi_T, C_fn, C_fp)
                    ]
                    for model in calibrated_models
                ],
                "bayes_error_plot_values": None
            },
            classifiers=list(map(lambda x: str(x), calibrated_models)),
            working_points=[working_point]
        )

        return eval_results
