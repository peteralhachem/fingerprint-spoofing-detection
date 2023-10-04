import sys
import numpy as np

from src.classifiers.fusion_classifier import BinaryFusionClassifier
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.score_calibration.log_reg_score_calibrator import BinaryLogisticRegressionScoreCalibrator
from src.utilities.arrays import vrow
from src.utilities.datasets import equal_partitions_of
from src.utilities.statistics import prior_log_odd_of_working_point


class BinaryCrossValidator:
    """
    Object able to perform k-fold cross validation over a set of binary Classifiers,
    computing statistics over them
    """

    def __init__(self, folds_data=None, folds_labels=None):
        """
        Create a Binary k-fold cross validator, providing the folds and the
        corresponding labels over which computing the classifiers performances
        :param folds_data: (optional) a list with the partitions (one 2-D Numpy array for each partition)
        :param folds_labels: (optional) a list of the corresponding partitioned labels
         (one 1-D Numpy array for each partition)
        """
        self.folds_data = folds_data
        self.folds_labels = folds_labels
        self.num_folds = None  # k

        self.num_models = None
        self.models_per_working_point = None  # contains a list of (trained and validated)
        # models for each working point

        self.working_points = None  # list of working points (pi_T, C_fn, C_fp)
        self.prior_log_odds = None  # list of corresponding prior log odds

        # score calibration
        self.log_reg_score_calibration = None
        self.score_calibration_true_train_priors = None
        self.calibrators_per_train_prior_and_working_point = None   # contains a list of (trained and validated)
                                                                    # score calibrators, one for each model,
                                                                    # and for each working point; (each model
                                                                    # is repeated for each train prior)
        self.calibrated_models_per_train_prior_and_working_point = None     # contains a list of trained and
                                                                            # calibrated models, each repeated
                                                                            # for each train calibration prior,
                                                                            # and one for each working point
                                                                            # Note: they don't have cumulative info
        self.fusion_calibrators_per_working_point_and_prior = None  # it contains a list of fusion classifiers
                                                                    # for each training prior; in each list, one
                                                                    # fusion classifier for each application prior
        # results
        self.__results: dict[str, float or None] = {
            "models_error_rates": None,
            "models_min_DCFs": None,
            "models_actual_DCFs": None,
            "bayes_error_plot_values": None
        }

    def k_folds(self, data_matrix, labels, k, shuffle=True, seed=None):
        """
        Split provided data for k-fold cross validation and save internally the folds

        :param data_matrix: 2-D numpy array containing one sample per column
        :param labels: 1-D numpy array containing one label for each sample
        :param k: the number of folds into which split the data
        :param shuffle: (default: True) indicate if shuffling the dataset before the split or not
        :param seed: (default: None) seed used to shuffle data; if None, data are shuffled by default with
         a different seed at each call
        :return: a list of folds (one 2-D Numpy array for each fold) and a list of folds labels
         (one 1-D Numpy array for each fold)

        """
        self.folds_data, self.folds_labels = equal_partitions_of(data_matrix, labels, k, shuffle=shuffle, seed=seed)
        return self.folds_data, self.folds_labels

    def leave_one_out_folds(self, data_matrix, labels, shuffle=True, seed=None):
        """
        Split provided data for leave-one-out cross validation

        :param data_matrix: 2-D numpy array containing one sample per column
        :param labels: 1-D numpy array containing one label for each sample
        :param shuffle: (default: True) indicate if shuffling the dataset before the split or not
        :param seed: (default: None) seed used to shuffle data; if None, data are shuffled by default with
         a different seed at each call
        :return: a list of folds (one 2-D Numpy array for each sample) and a list of folds labels
         (one 1-D Numpy array for each fold labels)

        """
        num_samples = data_matrix.shape[1]
        return self.k_folds(
            data_matrix,
            labels,
            num_samples,
            shuffle=shuffle,
            seed=seed
        )

    def cross_validation(
            self,
            classifiers,
            working_points=None,
            prior_log_odds=None,
            score_calibration_true_train_priors=None,
            compute_fusion_model=False
    ):
        """
        Perform k-fold cross validation training all the provided binary Classifiers
        over the previously provided folds

        **Note 1**: one between *working_points* and *prior_log_odds* must be provided (former has
        precedence over the latter)
        :param classifiers: list of binary classifiers objects to be compared
        :param working_points: (optional) list of working points to be analyzed; each working point must
         be a tuple (pi_T, C_fn, C_fp) where pi_T is the prior probability of True class (label=1),
         C_fn is the cost of false negatives and C_fp is the cost of false positives
        :param prior_log_odds: (optional) list of prior log odds corresponding to the interested working points;
         each element is an effective prior pi_T corresponding to the working point (pi_T, 1.0, 1.0)
        :param score_calibration_true_train_priors: (optional) to be provided only when score calibration has to be
         performed: list of application true prior probabilities used to train the score calibrator
        :param compute_fusion_model: (optional) specify if computing also the overall fusion model, with
         all the provided classifiers, for each training prior and working point
        :return: a list having one element per model kind; each element is a list of
         classifiers clones, each one trained and used to make predictions for a different working point

        **Note 2**: Each model classifier must comply with the BinaryClassifier class. For more details
        look at the **BinaryClassifier** class
        """
        self.__init__(self.folds_data, self.folds_labels)
        self.log_reg_score_calibration = (isinstance(score_calibration_true_train_priors, list) and
                                          len(score_calibration_true_train_priors) > 0)
        self.score_calibration_true_train_priors = score_calibration_true_train_priors

        if working_points is None and prior_log_odds is None:
            print("Error: you must provide at least one between working_points and prior_log_odds in order to "
                  "perform cross validation", file=sys.stderr)
            return None

        if self.log_reg_score_calibration and \
                (score_calibration_true_train_priors is None or not len(score_calibration_true_train_priors) > 0):
            print("Error: you must provide at least one train application prior to perform score calibration",
                  file=sys.stderr)
            return None

        self.num_folds = len(self.folds_data)
        self.num_models = len(classifiers)

        # save working points
        self.working_points = working_points if working_points is not None \
            else [(1.0 / (1.0 + np.exp(-x)), 1.0, 1.0) for x in prior_log_odds]     # transform prior log odds into
                                                                                    # working points
        # compute the corresponding prior log odds
        self.prior_log_odds = [prior_log_odd_of_working_point(x) for x in working_points] \
            if working_points is not None else prior_log_odds

        # create a clone of the classifier for each working point to be analyzed
        self.models_per_working_point = [
            [
                classifier.clone()  # one clone of the classifier
                for _ in range(len(self.working_points))  # for each working point
            ]
            for classifier in classifiers  # (a list for each kind of classifier)
        ]

        for i in range(self.num_folds):
            print("\n* %d-fold Cross Validation - Iteration #%d started *" % (self.num_folds, i + 1))

            # separate the cross_validation fold from the training ones
            evaluation_fold = self.folds_data[i]
            training_folds = [fold for j, fold in enumerate(self.folds_data) if j != i]

            evaluation_fold_labels = self.folds_labels[i]
            training_folds_labels = [fold_labels for j, fold_labels in enumerate(self.folds_labels) if j != i]

            # retrieve the training set and the cross_validation set for the current cross validation iteration
            training_set = np.hstack(training_folds)  # 2-D array
            training_labels = np.hstack(training_folds_labels)  # 1-D array

            evaluation_set = evaluation_fold
            evaluation_labels = evaluation_fold_labels

            for j in range(self.num_models):
                for k, (pi_T, C_fn, C_fp) in enumerate(self.working_points):
                    # extract the classifier to use
                    classifier = self.models_per_working_point[j][k]

                    print("- %s for (%.3f, %.3f, %.3f)" % (str(classifier), pi_T, C_fn, C_fp))

                    classifier.train(training_set, training_labels)

                    # predict labels using model j for the k-th working point
                    classifier.predict(evaluation_set, true_prior_probability=pi_T, error_costs=(C_fn, C_fp))

                    # evaluate model predictions
                    classifier.evaluate(evaluation_labels)

        print("\n* Done with Cross Validation *")

        if self.log_reg_score_calibration:
            print("\n* Score Calibration started * ")

            def __create_and_train_score_calibrator(classifier, _working_point, application_true_prior):
                cumulative_test_scores = vrow(np.hstack(classifier.cumulative_info["test_scores"]))  # 2D array
                cumulative_real_labels = np.hstack(classifier.cumulative_info["real_labels"])  # 1D array

                # train the LR score calibrator over the scores
                score_calibrator = BinaryLogisticRegressionScoreCalibrator()
                score_calibrator.train(
                    cumulative_test_scores,
                    cumulative_real_labels,
                    self.num_folds,
                    application_true_prior,
                    _working_point,
                    BinaryCrossValidator()
                )

                return score_calibrator

            models_per_train_prior_and_working_point = []
            calibrators_per_train_prior_and_working_point = []

            for wp_models in self.models_per_working_point:
                for true_train_prior in self.score_calibration_true_train_priors:
                    # one list for each combination of model and train prior
                    models_per_working_point = []
                    calibrators_per_working_point = []

                    for application_prior_model, working_point in zip(wp_models, self.working_points):
                        # one score calibrator and one classifier clone for each working point
                        trained_score_calibrator = __create_and_train_score_calibrator(
                            application_prior_model,
                            working_point,
                            true_train_prior
                        )

                        # save score calibrator
                        calibrators_per_working_point.append(trained_score_calibrator)

                        # clone the classifier and plug in score calibrator
                        cloned_classifier = application_prior_model.clone()
                        cloned_classifier.score_calibrators.append(trained_score_calibrator)

                        # save this classifier for the specified working point (and train prior)
                        models_per_working_point.append(cloned_classifier)

                    models_per_train_prior_and_working_point.append(models_per_working_point)
                    calibrators_per_train_prior_and_working_point.append(calibrators_per_working_point)

            if compute_fusion_model:
                def __create_and_train_fusion_score_calibrator(_classifiers, _working_point, application_true_prior):
                    scores = []
                    fusion_labels = None

                    for classifier in _classifiers:
                        cumulative_test_scores = vrow(np.hstack(classifier.cumulative_info["test_scores"]))  # 2D array
                        cumulative_real_labels = np.hstack(classifier.cumulative_info["real_labels"])  # 1D array
                        scores.append(cumulative_test_scores)
                        fusion_labels = cumulative_real_labels

                    pre_fusion_scores = np.vstack(scores)  # each column is a set of scores coming from the classifiers

                    # train the LR score calibrator over the scores
                    score_calibrator = BinaryLogisticRegressionScoreCalibrator()
                    score_calibrator.train(
                        pre_fusion_scores,
                        fusion_labels,
                        self.num_folds,
                        application_true_prior,
                        _working_point,
                        BinaryCrossValidator()
                    )

                    return score_calibrator

                for true_train_prior in self.score_calibration_true_train_priors:
                    # one list for each train prior
                    models_per_working_point = []
                    calibrators_per_working_point = []

                    for w, working_point in enumerate(self.working_points):
                        classifiers_to_fuse = list(map(
                            lambda models: models[w],
                            self.models_per_working_point
                        ))

                        # one score calibrator and one fusion classifier clone for each working point
                        trained_score_calibrator = __create_and_train_fusion_score_calibrator(
                            classifiers_to_fuse,
                            working_point,
                            true_train_prior
                        )

                        # save score calibrator
                        calibrators_per_working_point.append(trained_score_calibrator)

                        # create the fusion classifiers and plug in the trained score calibrator
                        fusion_classifier = BinaryFusionClassifier(
                            classifiers_to_fuse,
                            [trained_score_calibrator]
                        )

                        # save the fusion classifier for the specified working point (and train prior)
                        models_per_working_point.append(fusion_classifier)

                    models_per_train_prior_and_working_point.append(models_per_working_point)
                    calibrators_per_train_prior_and_working_point.append(calibrators_per_working_point)

                    # save fusion classifiers for this training prior
                    if self.fusion_calibrators_per_working_point_and_prior is None:
                        self.fusion_calibrators_per_working_point_and_prior = []

                    self.fusion_calibrators_per_working_point_and_prior.append(calibrators_per_working_point)

            self.calibrated_models_per_train_prior_and_working_point = models_per_train_prior_and_working_point
            self.calibrators_per_train_prior_and_working_point = calibrators_per_train_prior_and_working_point

            print("\n* Done with Score Calibration *")

        return self.models_per_working_point

    def models_error_rates(self):
        """
        For each provided working point, compute the cumulative error rate of each classifier
        over all the Cross Validation iterations.
        :return: a list of lists, one for each kind of classifier, each one having (in order)
         an error rate for each working point
        """
        print("\n* Computing models error rates in Cross Validation *")

        def from_classifier_and_working_point_to_error_rate(trio):
            model_for_stats, model_for_name, working_point = trio
            pi_T, C_fn, C_fp = working_point
            print("- %s and working point (%.3f, %.3f, %.3f)" % (model_for_name, pi_T, C_fn, C_fp))
            return model_for_stats.error(over_cumulative_predictions=True)

        models_for_stats = self.calibrators_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        models_for_names = self.calibrated_models_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        models_avg_error_rates_per_working_point = [
            # create a list of error rates for each kind of model
            list(
                map(
                    lambda trio: from_classifier_and_working_point_to_error_rate(trio),
                    zip(models_for_stats[model], models_for_names[model], self.working_points)
                )
            )
            for model in range(len(models_for_stats))
        ]

        self.__results["models_error_rates"] = models_avg_error_rates_per_working_point
        return models_avg_error_rates_per_working_point

    def models_min_DCFs(self):
        """
        For each provided working point, compute the **minimum** DCF of each classifier over all
        the Cross Validation iterations
        :return: a list of lists, one for each kind of classifier, each one having (in order)
         the minimum DCF value for working point
        """
        print("\n* Computing models minimum DCF for Cross Validation *")

        def from_classifier_and_working_point_to_min_DCF(trio):
            model_for_stats, model_for_name, working_point = trio
            pi_T, C_fn, C_fp = working_point

            print("- %s and working point (%.3f, %.3f, %.3f)" % (model_for_name, pi_T, C_fn, C_fp))

            return model_for_stats.min_DCF(
                true_prior_probability=pi_T,
                error_costs=(C_fn, C_fp),
                over_cumulative_predictions=True
            )

        models_for_stats = self.calibrators_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        models_for_names = self.calibrated_models_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        models_min_DCFs_per_working_point = [
            # create a list of min DCFs for each kind of model
            list(
                map(
                    lambda trio: from_classifier_and_working_point_to_min_DCF(trio),
                    zip(models_for_stats[model], models_for_names[model], self.working_points)
                )
            )
            for model in range(len(models_for_stats))
        ]

        self.__results["models_min_DCFs"] = models_min_DCFs_per_working_point
        return models_min_DCFs_per_working_point

    def models_actual_DCFs(self):
        """
        For each working point, compute the **actual** DCF of each classifier over all the
        Cross Validation iterations
        :return: a list of lists, one for each kind of model, each one having (in order)
         the actual DCF value for working point
        """
        print("\n* Computing models actual DCF for Cross Validation *")

        def from_classifier_and_working_point_to_actual_DCF(trio):
            model_for_stats, model_for_name, working_point = trio
            pi_T, C_fn, C_fp = working_point

            print("- %s and working point (%.3f, %.3f, %.3f)" % (model_for_name, pi_T, C_fn, C_fp))

            return model_for_stats.DCF(
                true_prior_probability=pi_T,
                error_costs=(C_fn, C_fp),
                over_cumulative_predictions=True
            )

        models_for_stats = self.calibrators_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        models_for_names = self.calibrated_models_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        models_actual_DCFs_per_working_point = [
            # create a list of DCFs for each kind of model
            list(
                map(
                    lambda trio: from_classifier_and_working_point_to_actual_DCF(trio),
                    zip(models_for_stats[model], models_for_names[model], self.working_points)
                )
            )
            for model in range(len(models_for_stats))
        ]

        self.__results["models_actual_DCFs"] = models_actual_DCFs_per_working_point
        return models_actual_DCFs_per_working_point

    def models_bayes_error_plots(self):
        """
        Return bayes error plot values (i.e. actual and minimum DCF values) for the analyzed models over the
        test set, using k fold cross validation and using the provided working points (or prior log odds)
        as interested points.

        **Note**: you must first perform cross_validation() before calling this method
        :return: list of objects, one for each model, each one containing:
         - a list of prior log odds (the same for each model)
         - a list of actual DCFs of the model over those prior log odds
         - a list of minimum DCFs of the model over those prior log odds
        """
        print("\n* Computing models bayes error plots values for Cross Validation *")

        if self.num_models is None:
            print("Error: you must first perform cross validation before retrieving bayes error plot values",
                  file=sys.stderr)
            return None

        models_actual_DCFs_per_working_point = np.vstack(self.__results["models_actual_DCFs"]
                if self.__results["models_actual_DCFs"] is not None else self.models_actual_DCFs())

        models_min_DCFs_per_working_point = np.vstack(self.__results["models_min_DCFs"]
                if self.__results["models_min_DCFs"] is not None else self.models_min_DCFs())

        models_for_stats = self.calibrators_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        results = [
            # one set of values for each kind of model
            {
                "prior_log_odds": self.prior_log_odds,
                "actual_DCFs": list(models_actual_DCFs_per_working_point[i, :]),  # actual DCFs of i-th model
                "min_DCFs": list(models_min_DCFs_per_working_point[i, :])  # min DCFs of i-th model
            }
            for i in range(len(models_for_stats))
        ]

        self.__results["bayes_error_plot_values"] = results
        return results

    def results(self, error_rates=False, actual_DCFs=False, min_DCFs=False, bayes_error_plot_values=False):
        """
        Returns an object containing all the Cross Validation results computed so far.
        Optionally specify the metrics to compute before returning the results
        :param error_rates: (default: False) compute the models error rates before returning the result
        :param actual_DCFs: (default: False) compute the models actual DCFs before returning the result
        :param min_DCFs: (default: False) compute the models minimum DCFs before returning the result
        :param bayes_error_plot_values: (default: False) compute the models
         bayes error plot values before returning the result
        :return: an instance of CrossValidationResults with the computed results
        """
        if error_rates:
            self.models_error_rates()

        if actual_DCFs:
            self.models_actual_DCFs()

        if min_DCFs:
            self.models_min_DCFs()

        if bayes_error_plot_values:
            self.models_bayes_error_plots()

        models_for_names = self.calibrated_models_per_train_prior_and_working_point \
            if self.log_reg_score_calibration else self.models_per_working_point

        classifiers_names = list(map(lambda m: str(m[0]), models_for_names))
        return CrossValidationResults(self.__results, classifiers_names, self.working_points)
