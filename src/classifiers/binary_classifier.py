import sys
from typing import Iterable

import numpy as np


class BinaryClassifier:
    def __init__(self, preprocessors=None, score_calibrators=None):
        self.unique_labels = None       # array of unique labels (one per class)

        # data/scores processors
        self.preprocessors = preprocessors \
            if preprocessors is not None else []    # list of (ordered) preprocessors to be applied to input
                                                    # data both in training and prediction phases
        self.score_calibrators = score_calibrators \
            if score_calibrators is not None else []    # list of (ordered) preprocessors to be applied to output
                                                        # scores before comparing them with the threshold
        # predictions
        self.test_scores = None         # 1-D Numpy array of the scores used to classify the last test samples
        self.predicted_labels = None    # 1-D array of (ordered) predicted labels, one per sample

        # cross_validation results
        self.evaluation_true_prior_probability = None   # prior probability of the True class used for cross_validation
        self.evaluation_error_costs = None              # false negative cost and false positive cost
                                                        # used for cross_validation
        self.effective_prior = None                     # effective prior computed starting from
                                                        # cross_validation priors and error costs

        self.real_labels = None         # 1-D Numpy array containing the real labels for the last evaluated samples
        self.confusion_matrix = None    # 2-D Numpy array representing the confusion matrix of
                                        # the last cross_validation: (i,j) is the number of test samples
                                        # of class 'j' predicted as of class 'i'
        self.classes_eval_stats = None
        self.correct_predictions = None
        self.num_test_samples = None
        self.error_rate = None
        self.accuracy = None

        # cumulative info (initialize it just once)
        if not hasattr(self, "cumulative_info"):
            self.cumulative_info: dict[str, None or Iterable] = {
                "real_labels": None,        # list of 1-D arrays of all the real labels
                                            # for the previous evaluations' samples
                "test_scores": None,        # list of 1-D arrays of scores of all the previous evaluations
                "confusion_matrix": None    # confusion matrix containing all the previous evaluations info
            }

    # (custom function specific for the classifier)
    def _internal_clone(self, clone):
        clone.unique_labels = list(self.unique_labels) if self.unique_labels is not None else None
        clone.test_scores = np.copy(self.test_scores) if self.test_scores is not None else None
        clone.predicted_labels = np.copy(self.predicted_labels) if self.predicted_labels is not None else None
        clone.evaluation_true_prior_probability = self.evaluation_true_prior_probability
        clone.evaluation_error_costs = self.evaluation_error_costs
        clone.effective_prior = self.effective_prior
        clone.real_labels = np.copy(self.real_labels) if self.real_labels is not None else None
        clone.confusion_matrix = np.copy(self.confusion_matrix) if self.confusion_matrix is not None else None
        clone.classes_eval_stats = self.classes_eval_stats
        clone.correct_predictions = self.correct_predictions
        clone.num_test_samples = self.num_test_samples
        clone.error_rate = self.error_rate
        clone.accuracy = self.accuracy
        return clone

    # *** customize this function specifically for the classifier ***
    def clone(self):
        pass

    # *** customize this function specifically for the classifier ***
    def train(self, data_matrix, labels):
        # save labels
        self.unique_labels = list(sorted(set(labels)))

        # preprocess data
        for preprocessor in self.preprocessors:
            preprocessor.fit(data_matrix)
            data_matrix = preprocessor.preprocess(data_matrix)

        return data_matrix

    # *** customize function specifically for the classifier ***
    # (for some models, working point info might not be useful for predictions, e.g. SVM)
    def predict(self, data_matrix, true_prior_probability=None, error_costs=None):
        # preprocess data
        for preprocessor in self.preprocessors:
            data_matrix = preprocessor.preprocess(data_matrix)

        return data_matrix

    def _internal_predict(self, threshold):
        # save cumulative scores
        if self.cumulative_info["test_scores"] is None:
            self.cumulative_info["test_scores"] = []

        self.cumulative_info["test_scores"].append(self.test_scores)

        # predict the labels assigning 1 for scores above the threshold, 0 otherwise
        self.predicted_labels = np.array(
            [1 if score > threshold else 0 for score in self.test_scores],
            dtype=np.int32
        )

    def calibrate(self, scores):
        """
        Calibrate scores using the already provided score calibrators objects
        :param scores: the 1-D Numpy array of scores to calibrate
        :return: a 1-D Numpy array of calibrated scores
        """

        for score_calibrator in self.score_calibrators:
            scores = score_calibrator.calibrate(scores)

        return scores

    def evaluate(self, real_labels, classes_stats=False, global_stats=False):
        """
        Evaluate performances of the classifier considering the specified real labels:
        compute confusion matrix, accuracy, error rate and other useful classification information
        :param real_labels: 1-D Numpy array of the real labels of the test samples
        :param classes_stats: (default: True) compute global cross_validation statistics
        :param global_stats: (default: True) compute cross_validation statistics of each class
        """
        # save real labels
        self.real_labels = real_labels

        # compute confusion matrix
        self.confusion_matrix = self.__compute_confusion_matrix(self.predicted_labels, real_labels)

        # save real labels in cumulative info
        if self.cumulative_info["real_labels"] is None:
            self.cumulative_info["real_labels"] = []

        self.cumulative_info["real_labels"].append(real_labels)

        # save confusion matrix in cumulative info
        if self.cumulative_info["confusion_matrix"] is None:
            self.cumulative_info["confusion_matrix"] = self.confusion_matrix
        else:
            self.cumulative_info["confusion_matrix"] += self.confusion_matrix

        # compute classes and global cross_validation statistics
        if classes_stats:
            self.classes_eval_stats = self.__compute_classes_evaluation_stats(self.confusion_matrix)

        if global_stats:
            self.__compute_global_evaluation_stats()

    def error(self, over_cumulative_predictions=False):
        if not over_cumulative_predictions:
            return self.error_rate
        else:
            cumulative_correct_predictions = self.cumulative_info["confusion_matrix"].trace()
            tot_predictions = self.cumulative_info["confusion_matrix"].sum()
            accuracy = float(cumulative_correct_predictions) / float(tot_predictions)
            error_rate = 1.0 - accuracy
            return error_rate

    def empirical_bayes_risk(
        self,
        true_prior_probability=0.5,
        error_costs=(1.0,1.0),
        over_cumulative_predictions=False
    ):
        """
        Compute the empirical Bayes Risk (or unnormalized Detection Cost Function) of the predictions,
        starting from the computed confusion matrix, and considering the provided priors and error costs
        :param true_prior_probability: (optional) prior probability of the True class (label=1); if not
         provided, classes are considered as balanced (default: 0.5)
        :param error_costs: (optional) tuple containing the cost of false negatives and the cost
         of false positives (C_fn, C_fp); if not provided, unit costs are used (default: (1.0,1.0))
        :param over_cumulative_predictions: (default: False) compute the result over all the predictions
         made so far, and not just over the last ones
        :return: the value of the empirical Bayes Risk for the test samples
        """
        if self.confusion_matrix is None or self.cumulative_info["confusion_matrix"] is None:
            print("Error: before computing the empirical Bayes Risk you must compute a confusion matrix",
                  file=sys.stderr)
            return None

        # save prior and costs
        self.evaluation_true_prior_probability = true_prior_probability
        self.evaluation_error_costs = error_costs

        C_fn, C_fp = error_costs
        pi_T = true_prior_probability

        # compute and save effective prior
        self.effective_prior = float(pi_T * C_fn) / float(pi_T * C_fn + (1.0 - pi_T) * C_fp)

        if over_cumulative_predictions:
            examined_confusion_matrix = self.cumulative_info["confusion_matrix"]
        else:
            examined_confusion_matrix = self.confusion_matrix

        # compute FNR and FPR
        false_negative_rate = float(examined_confusion_matrix[0,1]) / float(examined_confusion_matrix[:,1].sum())
        false_positive_rate = float(examined_confusion_matrix[1,0]) / float(examined_confusion_matrix[:,0].sum())

        return float(pi_T * C_fn) * false_negative_rate + float((1.0 - pi_T) * C_fp) * false_positive_rate

    def u_DCF(
        self,
        true_prior_probability=0.5,
        error_costs=(1.0,1.0),
        over_cumulative_predictions=False
    ):
        """
        Compute the unnormalized Detection Cost Function over the made predictions
        (Alias of empirical_bayes_risk())
        """
        return self.empirical_bayes_risk(
            true_prior_probability=true_prior_probability,
            error_costs=error_costs,
            over_cumulative_predictions=over_cumulative_predictions
        )

    def DCF(
        self,
        true_prior_probability=0.5,
        error_costs=(1.0,1.0),
        over_cumulative_predictions=False
    ):
        """
        Compute the (normalized) Detection Cost Function over the predictions from the computed confusion
        matrix, using the provided prior probability and error costs; it is the ratio between the
        unnormalized Detection Cost Function and the bayes cost of the best dummy system
        :param true_prior_probability: (optional) prior probability of the True class (label=1); if not
         provided, classes are considered as balanced (default: 0.5)
        :param error_costs: (optional) tuple containing the cost of false negatives and the cost
         of false positives (C_fn, C_fp); if not provided, unit costs are used (default: (1.0,1.0))
        :param over_cumulative_predictions: (default: False) compute the result over all the predictions
         made so far, and not just over the last ones
        :return: the value of the (normalized) Detection Cost Function
        """
        C_fn, C_fp = error_costs
        pi_T = true_prior_probability

        u_DCF = self.u_DCF(
            true_prior_probability=true_prior_probability,
            error_costs=error_costs,
            over_cumulative_predictions=over_cumulative_predictions
        )

        best_dummy_system_cost = min(float(pi_T * C_fn), float((1.0 - pi_T) * C_fp))
        return u_DCF / best_dummy_system_cost

    def min_DCF(
        self,
        true_prior_probability=0.5,
        error_costs=(1.0,1.0),
        real_labels=None,
        over_cumulative_predictions=False
    ):
        """
        Compute the minimum (normalized) Detection Cost Function over the computed scores, for the
        specified working point (prior probability of True class, false negative cost and false positive cost)
        :param true_prior_probability: (optional) prior probability of the True class (label=1); if not
         provided, classes are considered as balanced (default: 0.5)
        :param error_costs: (optional) tuple containing the cost of false negatives and the cost
         of false positives (C_fn, C_fp); if not provided, unit costs are used (default: (1.0,1.0))
        :param real_labels: (optional) 1-D Numpy array of real labels to be used for the computation;
         if not provided, the previously stored ones are used
        :param over_cumulative_predictions: (default: False) compute the result over all the predictions
         made so far, and not just over the last ones
        :return: the value of the minimum (normalized) Detection Cost Function
        """

        if self.test_scores is None or self.cumulative_info["test_scores"] is None:
            print("Error: before computing minDCF you must compute scores over the test samples", file=sys.stderr)
            return None

        pi_T = true_prior_probability
        C_fn, C_fp = error_costs

        if over_cumulative_predictions:
            examined_scores = np.hstack(self.cumulative_info["test_scores"])
            examined_real_labels = np.hstack(self.cumulative_info["real_labels"]) \
                if real_labels is None else real_labels
        else:
            examined_scores = self.test_scores
            examined_real_labels = self.real_labels if real_labels is None else real_labels

        if examined_real_labels is None:
            print("Error: real labels cannot be None in min DCF computation", file=sys.stderr)
            return None

        min_dcf = None      # minimum detection cost
        best_dummy_system_dcf = min(float(pi_T * C_fn), float((1.0 - pi_T) * C_fp))

        # use -inf, s0, s1, ..., sn, inf as thresholds
        thresholds = np.array([-np.inf] + sorted(list(examined_scores)) + [np.inf])

        # try all the thresholds to find the optimal one, and save the minimum detection cost
        for threshold in thresholds:
            # compute the predicted labels, predicting '1' if the score is above the threshold, '0' otherwise
            predicted_labels = np.array([1 if score > threshold else 0 for score in examined_scores])

            # compute the resulting confusion matrix, given the samples' real labels
            confusion_matrix = self.__compute_confusion_matrix(predicted_labels, examined_real_labels)

            # compute FNR and FPR over the resulting matrix
            false_negative_rate = float(confusion_matrix[0, 1]) / float(confusion_matrix[:, 1].sum())
            false_positive_rate = float(confusion_matrix[1, 0]) / float(confusion_matrix[:, 0].sum())

            # compute DCF for the current threshold and the given working point
            u_dcf = float(pi_T * C_fn) * false_negative_rate + float((1.0 - pi_T) * C_fp) * false_positive_rate
            dcf = u_dcf / best_dummy_system_dcf

            # save min DCF
            if min_dcf is None or dcf < min_dcf:
                min_dcf = dcf

        return min_dcf

    def DET_plot(
        self,
        real_labels=None,
        over_cumulative_predictions=False
    ):
        """
        Compute the DET (Detection Error Tradeoff) plot over the test set,
        plotting the FNR on the y-axis and the FPR on the x-axis
        :param real_labels: (optional) 1-D Numpy array containing the real labels of the test set used to
         compute FPR and FNR
        :param over_cumulative_predictions: (default: False) compute the result over all the predictions
         made so far, and not just over the last ones
        :return: (in order) the x (FPR) and the y (FNR) arrays values to plot
        """
        if self.test_scores is None or self.cumulative_info["test_scores"] is None:
            print("Error: before computing the DET plot you must compute scores over the test samples", file=sys.stderr)
            return None

        if over_cumulative_predictions:
            examined_scores = np.hstack(self.cumulative_info["test_scores"])
            examined_real_labels = np.hstack(self.cumulative_info["real_labels"]) \
                if real_labels is None else real_labels
        else:
            examined_scores = self.test_scores
            examined_real_labels = self.real_labels if real_labels is None else real_labels

        if examined_real_labels is None:
            print("Error: real labels cannot be None in DET plot computation", file=sys.stderr)
            return None

        # use -inf, s0, s1, ..., sn, inf as thresholds
        thresholds = np.array([-np.inf] + sorted(list(examined_scores)) + [np.inf])

        fnr_y = []
        fpr_x = []

        # compute FPR and FNR over the test samples trying all the possible thresholds
        for threshold in thresholds:
            # compute the predicted labels, predicting '1' if the score is above the threshold, '0' otherwise
            predicted_labels = np.array([1 if score > threshold else 0 for score in examined_scores])

            # compute the resulting confusion matrix, given the samples' real labels
            confusion_matrix = self.__compute_confusion_matrix(predicted_labels, examined_real_labels)

            # compute FNR and FPR over the resulting matrix
            false_negative_rate = float(confusion_matrix[0, 1]) / float(confusion_matrix[:, 1].sum())
            false_positive_rate = float(confusion_matrix[1, 0]) / float(confusion_matrix[:, 0].sum())

            fnr_y.append(false_negative_rate)
            fpr_x.append(false_positive_rate)

        return fpr_x, fnr_y

    # * internal functions *

    def __compute_confusion_matrix(self, predicted_labels, real_labels):
        num_classes = len(self.unique_labels)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for predicted_label, real_label in zip(predicted_labels, real_labels):
            confusion_matrix[predicted_label][real_label] += 1

        return confusion_matrix

    def __compute_classes_evaluation_stats(self, confusion_matrix):
        if confusion_matrix is None:
            print("Error: you have first to compute the confusion matrix before computing cross_validation stats")
            return None

        classes_eval_stats = []

        for class_label in sorted(self.unique_labels):
            class_stats = {
                "label": int(class_label),
                "correct_predictions": int(confusion_matrix[class_label, class_label]),
                "num_samples": int(confusion_matrix[:, class_label].sum()),
                "num_predictions": int(confusion_matrix[class_label, :].sum())
            }

            # compute class accuracy and error rate
            class_stats["accuracy"] = float(class_stats["correct_predictions"]) / float(class_stats["num_samples"]) \
                if class_stats["num_samples"] > 0 else "(not defined)"
            class_stats["error_rate"] = 1.0 - class_stats["accuracy"]   \
                if class_stats["num_samples"] > 0 else "(not defined)"

            # save class stats
            classes_eval_stats.append(class_stats)

        return classes_eval_stats

    def __compute_global_evaluation_stats(self):
        # compute num of global correct predictions
        self.correct_predictions = self.confusion_matrix.trace()

        # compute global accuracy and error rate
        self.num_test_samples = len(self.predicted_labels)
        self.accuracy = float(self.correct_predictions) / float(self.num_test_samples)
        self.error_rate = 1.0 - self.accuracy
