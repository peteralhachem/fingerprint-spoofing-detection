import sys
import numpy as np
from src.classifiers.binary_classifier import BinaryClassifier
from src.utilities.gaussian import log_MGD, maximum_likelihood_density_estimation_for


class BinaryGaussianClassifier(BinaryClassifier):
    def __init__(self, naive_bayes_assumption=False, tied_covariance_matrices=False,
                 preprocessors=None, score_calibrators=None):
        """
        Build a brand new Gaussian Classifier
        :param naive_bayes_assumption: (optional) assume stochastic independence between features
         (Naïve Bayes assumption); default: False
        :param tied_covariance_matrices: assume tied covariance matrices of the different classes
         (estimate only one covariance matrix for all the classes); default: False
        :param preprocessors: (optional) list of (ordered) preprocessors to be applied to input data
         before training and prediction phases for preprocessing strategies
        :param score_calibrators: (optional) list of (ordered) score calibration models to be applied to
         output scores before comparing them with the threshold and making predictions
        """
        super().__init__(preprocessors=preprocessors, score_calibrators=score_calibrators)

        self.naive_bayes_assumption = naive_bayes_assumption
        self.tied_covariance_matrices = tied_covariance_matrices

        self.means = None           # array of means (one per class)
        self.covariances = None     # array of covariance matrices (one per class)
        self.num_train_samples = None

        # partial results
        self.log_likelihoods = None     # 2-D numpy matrix of log-likelihoods of the samples:
                                        # (i, j) element is the log of the class conditional probability
                                        # of the j-th sample for the i-th class

        self.log_likelihood_ratios = None   # 1-D Numpy array of log-likelihood ratios of the samples:
                                            # the i-th element is the log ratio between likelihood of
                                            # class 1 and likelihood of class 0 for the i-th sample

    def clone(self):
        clone = BinaryGaussianClassifier(
            naive_bayes_assumption=self.naive_bayes_assumption,
            tied_covariance_matrices=self.tied_covariance_matrices,
            preprocessors=list(self.preprocessors),
            score_calibrators=list(self.score_calibrators)
        )

        super()._internal_clone(clone)
        clone.means = list(self.means) if self.means is not None else None
        clone.covariances = list(self.covariances) if self.covariances is not None else None
        clone.num_train_samples = self.num_train_samples
        clone.log_likelihoods = np.copy(self.log_likelihoods) if self.log_likelihoods is not None else None
        clone.log_likelihood_ratios = np.copy(self.log_likelihood_ratios) \
            if self.log_likelihood_ratios is not None else None
        return clone

    def train(self, data_matrix, labels):
        """
        Train the Gaussian classifier estimating the parameters of the
        Gaussian distributions for each class
        :param data_matrix: 2-D numpy array representing the samples (one per column)
        :param labels: 1-D numpy array representing the samples' labels
        """
        # initialize parameters
        self.__init__(
            naive_bayes_assumption=self.naive_bayes_assumption,
            tied_covariance_matrices=self.tied_covariance_matrices,
            preprocessors=self.preprocessors,
            score_calibrators=self.score_calibrators
        )

        if len(set(labels)) != 2:
            print("Error: to train a Binary Gaussian Classifier you need a training set with (just) "
                  "2 different labels",
                  file=sys.stderr)
            return

        # preprocess data, if necessary
        data_matrix = super().train(data_matrix, labels)

        self.num_train_samples = data_matrix.shape[1]

        means = []
        covariances = []

        for class_label in self.unique_labels:
            # extract samples belonging to the i-th class
            class_samples = data_matrix[:, labels==class_label]

            # estimate mean and covariance of the class Gaussian distribution with
            # Maximum Likelihood solution (covariance is a matrix if naive bayes
            # assumption is False, or an array of variances otherwise)
            class_mean, class_covariance = \
                maximum_likelihood_density_estimation_for(class_samples, self.naive_bayes_assumption)

            # save them
            means.append(class_mean)
            covariances.append(class_covariance)

        if self.tied_covariance_matrices:
            # compute the weighted average of the classes' covariances (i.e. the within class
            # covariance matrix, or just its diagonal in case of naive bayes assumption)
            # and use it as class covariance for each class
            num_dimensions = data_matrix.shape[0]
            tied_covariance = np.zeros((num_dimensions, num_dimensions)) \
                if not self.naive_bayes_assumption else np.zeros(num_dimensions)

            for i in range(len(self.unique_labels)):
                class_label = self.unique_labels[i]
                class_num_samples = (labels == class_label).sum()

                tied_covariance += class_num_samples * covariances[i]

            tied_covariance /= self.num_train_samples

            # place the same tied covariance instead of each class covariance matrices
            covariances = [tied_covariance for _ in self.unique_labels]

        # save means vectors and covariance matrices
        self.means = means
        self.covariances = covariances

    def predict(self, data_matrix, true_prior_probability=0.5, error_costs=(1.0, 1.0)):
        """
        Classify a set of data samples computing the corresponding labels
        :param data_matrix: 2-D numpy array containing one test sample for each column
        :param true_prior_probability: (optional) prior probability of the True class (label=1),
         if not provided, classes are considered as balanced (default: 0.5)
        :param error_costs: (optional) bayes prediction costs of the two classes: cost of false negatives
         and cost of false positives; if not provided, unit costs are considered (default: (1.0, 1.0))
        """
        # preprocess test samples
        data_matrix = super().predict(data_matrix)

        # 1 - for each class Gaussian distribution, compute the log-likelihoods of all the samples
        log_likelihoods = []

        for i in list(sorted(self.unique_labels)):
            class_mean = self.means[i]
            class_covariance = self.covariances[i]

            # array of log conditional densities for the current class
            class_log_likelihoods = \
                log_MGD(data_matrix, class_mean, class_covariance, diagonal_matrix=self.naive_bayes_assumption)
            log_likelihoods.append(class_log_likelihoods)

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
        return "%s%sBinary Gaussian Classifier%s%s" % \
            (
                "Naive Bayes "     if self.naive_bayes_assumption   else "",
                "Tied Covariance " if self.tied_covariance_matrices else "",
                " (%s)" % ", ".join([str(x) for x in self.preprocessors]) if len(self.preprocessors) > 0 else "",
                " with score calibration (pi=%f)"
                % self.score_calibrators[0].application_true_prior if len(self.score_calibrators) > 0 else ""
            )
