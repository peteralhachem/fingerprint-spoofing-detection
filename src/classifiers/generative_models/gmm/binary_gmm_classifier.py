import sys

from src.classifiers.binary_classifier import BinaryClassifier
from src.utilities.gaussian import log_GMM
import numpy as np


class BinaryGmmClassifier(BinaryClassifier):
    """
    Gaussian Mixture Model (GMM) classifier
    """
    def __init__(self,
                 true_class_gmm_components, false_class_gmm_components,
                 true_class_gmm_lbg_estimator, false_class_gmm_lbg_estimator,
                 preprocessors=None, score_calibrators=None):
        """
        Create a Gaussian Mixture Model classifier, specifying the GMM LBG estimator for
         each class; the estimator estimates each class GMM distribution with the LBG algorithm.
        :param true_class_gmm_components: the number of GMM components to be
         estimated in the **True** class distribution
        :param false_class_gmm_components: the number of GMM components to be
         estimated in the **False** class distribution
        :param true_class_gmm_lbg_estimator: the GMM LBG estimator for the **True** class
        :param false_class_gmm_lbg_estimator: the GMM LBG estimator for the **False** class
        """
        super().__init__(preprocessors=preprocessors, score_calibrators=score_calibrators)

        self.true_class_gmm_components = true_class_gmm_components
        self.false_class_gmm_components = false_class_gmm_components
        self.true_class_gmm_lbg_estimator = true_class_gmm_lbg_estimator
        self.false_class_gmm_lbg_estimator = false_class_gmm_lbg_estimator

        self.classes_estimated_GMM_params = None    # list containing a set of GMM parameters for each class;
                                                    # each set is an iterable of (weight, mean, covariance), one
                                                    # for each GMM component

        # partial results
        self.log_likelihoods = None     # 2-D numpy matrix of log-likelihoods of the samples:
                                        # (i, j) element is the log of the class conditional probability
                                        # of the j-th sample for the i-th class

        self.log_likelihood_ratios = None   # 1-D Numpy array of log-likelihood ratios of the samples:
                                            # the i-th element is the log ratio between likelihood of
                                            # class 1 and likelihood of class 0 for the i-th sample

    def clone(self):
        clone = BinaryGmmClassifier(
            self.true_class_gmm_components,
            self.false_class_gmm_components,
            self.true_class_gmm_lbg_estimator,
            self.false_class_gmm_lbg_estimator,
            preprocessors=list(self.preprocessors),
            score_calibrators=list(self.score_calibrators)
        )

        super()._internal_clone(clone)
        clone.classes_estimated_GMM_params = list(self.classes_estimated_GMM_params) \
            if self.classes_estimated_GMM_params is not None else None
        clone.log_likelihoods = np.copy(self.log_likelihoods) if self.log_likelihoods is not None else None
        clone.log_likelihood_ratios = np.copy(self.log_likelihood_ratios) \
            if self.log_likelihood_ratios is not None else None
        return clone

    def train(self, data_matrix, labels):
        """
        Train the GMM model computing the classes' components parameters, leveraging the LBG algorithm
        :param data_matrix: 2-D Numpy array containing one train sample for each column
        :param labels: 1-D Numpy array having one label for each train sample
        """
        if len(set(labels)) != 2:
            print("Error: to train a Binary GMM Classifier you need a training set with (just) "
                  "2 different labels",
                  file=sys.stderr)
            return

        original_data_matrix_ref = data_matrix

        # preprocess data, if necessary
        data_matrix = super().train(data_matrix, labels)

        # * estimate GMM distribution for each class with LBG algorithm *

        # True class
        true_class_samples = data_matrix[:, labels==1]
        true_class_estimated_GMM_params = self.true_class_gmm_lbg_estimator.estimate(
            original_data_matrix_ref,
            true_class_samples,
            self.true_class_gmm_components
        )

        # False class
        false_class_samples = data_matrix[:, labels == 0]
        false_class_estimated_GMM_params = self.false_class_gmm_lbg_estimator.estimate(
            original_data_matrix_ref,
            false_class_samples,
            self.false_class_gmm_components
        )

        # save params
        self.classes_estimated_GMM_params = [
            false_class_estimated_GMM_params,
            true_class_estimated_GMM_params
        ]

    def predict(self, data_matrix, true_prior_probability=0.5, error_costs=(1.0, 1.0)):
        """
        Predict labels of a set of samples using the trained GMM model
        :param data_matrix: 2-D Numpy array having one test sample per column
        :param true_prior_probability: (optional) prior probability of the True class (label=1),
         if not provided, classes are considered as balanced (default: 0.5)
        :param error_costs: (optional) bayes prediction costs of the two classes: cost of false negatives
         and cost of false positives; if not provided, unit costs are considered (default: (1.0, 1.0))
        """
        # preprocess test samples
        data_matrix = super().predict(data_matrix)

        # 1 - for each class GMM distribution, compute the log-likelihoods of all the samples
        log_likelihoods = [
            log_GMM(data_matrix, self.classes_estimated_GMM_params[0]),     # False class
            log_GMM(data_matrix, self.classes_estimated_GMM_params[1])      # True class
        ]

        self.log_likelihoods = np.vstack(log_likelihoods)  # vertically stack the log-likelihoods

        # 2 - compute log likelihood ratios (llr) for each test sample
        self.log_likelihood_ratios = self.log_likelihoods[1, :] - self.log_likelihoods[0, :]

        # calibrate and save test scores (using as original scores the log likelihood ratios)
        self.test_scores = self.calibrate(self.log_likelihood_ratios)

        # 3 - compute the threshold, considering both prior probabilities and error costs
        if true_prior_probability is None:
            true_prior_probability = 0.5

        if error_costs is None:
            error_costs = (1.0, 1.0)

        pi_T = true_prior_probability
        C_fn, C_fp = error_costs

        threshold = -np.log(pi_T * C_fn) + np.log((1.0 - pi_T) * C_fp)

        # predict labels
        super()._internal_predict(threshold)

    def __str__(self):
        result = ("GMM classifier (TC: #%s%s - FC: #%s%s)%s" % (
            self.true_class_gmm_components,
            " (%s%s)" % (
                "D" if self.true_class_gmm_lbg_estimator.diagonal_covariance else "",
                "T" if self.true_class_gmm_lbg_estimator.tied_covariance else ""
            ) if (
                self.true_class_gmm_lbg_estimator.diagonal_covariance or
                self.true_class_gmm_lbg_estimator.tied_covariance
            ) else "",
            self.false_class_gmm_components,
            " (%s%s)" % (
                "D" if self.false_class_gmm_lbg_estimator.diagonal_covariance else "",
                "T" if self.false_class_gmm_lbg_estimator.tied_covariance else ""
            ) if (
                self.false_class_gmm_lbg_estimator.diagonal_covariance or
                self.false_class_gmm_lbg_estimator.tied_covariance
            ) else "",
            (", %s" % ", ".join([str(x) for x in self.preprocessors])) if len(self.preprocessors) > 0 else ""
        ))

        if self.true_class_gmm_lbg_estimator.min_eigenvalue is not None:
            result += " - with min eigenvalues %f" % self.true_class_gmm_lbg_estimator.min_eigenvalue

        if len(self.score_calibrators) > 0:
            result += " - score calibration (pi=%f)" % self.score_calibrators[0].application_true_prior

        return result
