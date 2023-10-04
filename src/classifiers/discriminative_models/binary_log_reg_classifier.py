import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.utilities.arrays import vrow
from src.classifiers.binary_classifier import BinaryClassifier


class BinaryLogisticRegressionClassifier(BinaryClassifier):
    """
    Logistic Regression model for binary classification tasks
    """

    def __init__(self, regularization_param=None, rebalancing_true_prior_probability=None,
                 preprocessors=None, score_calibrators=None):
        """
        Create a new binary Logistic Regression classifier with a specified value of lambda
        :param regularization_param: (lambda) the regularization hyperparameter
        :param rebalancing_true_prior_probability: (optional) prior probability of True class
         used to rebalance classes logistic loss during training
        :param preprocessors: (optional) list of (ordered) preprocessors to be applied to input data
         before training and prediction phases for preprocessing strategies
        :param score_calibrators: (optional) list of (ordered) score calibration models to be applied to
         output scores before comparing them with the threshold and making predictions
        """
        super().__init__(preprocessors=preprocessors, score_calibrators=score_calibrators)

        # regularization hyper-parameter
        self._lambda = regularization_param if regularization_param is not None else 0.0

        # class rebalancing
        self.rebalancing_true_prior_probability = rebalancing_true_prior_probability

        self.empirical_prior = None     # empirical prior of class 1

        # model parameters
        self.w = None  # K dim vector (K = #dimensions of feature vectors)
        self.b = None  # scalar

        # optimization info of numerical solver
        self.obj_function_min_value = None
        self.opt_info = None

    def clone(self):
        """
        Create a clone of this Binary Logistic Regression classifier
        :return: a new instance of BinaryLogisticRegressionClassifier with the same parameters
        """
        clone = BinaryLogisticRegressionClassifier(
            regularization_param=self._lambda,
            rebalancing_true_prior_probability=self.rebalancing_true_prior_probability,
            preprocessors=list(self.preprocessors),
            score_calibrators=list(self.score_calibrators)
        )

        super()._internal_clone(clone)
        clone.empirical_prior = self.empirical_prior
        clone.w = self.w
        clone.b = self.b
        clone.obj_function_min_value = self.obj_function_min_value
        clone.opt_info = self.opt_info
        return clone

    def train(self, data_matrix, labels):
        """
        Train the Logistic regression classifier computing the model parameters 'w' and 'b',
        where 'w' is a vector and 'b' is a scalar.
        :param data_matrix: 2-D Numpy array having one train sample per column
        :param labels: 1-D Numpy array having one integer label per sample
        """
        # initialize classifier
        self.__init__(
            regularization_param=self._lambda,
            rebalancing_true_prior_probability=self.rebalancing_true_prior_probability,
            preprocessors=self.preprocessors,
            score_calibrators=self.score_calibrators
        )

        # preprocess data, if necessary
        data_matrix = super().train(data_matrix, labels)

        num_samples_class_1 = data_matrix[:, labels == 1].shape[1]
        num_samples = data_matrix.shape[1]
        self.empirical_prior = float(num_samples_class_1) / float(num_samples)

        # define the objective function we want to minimize to solve this Logistic Regression task
        def avg_LogLoss_objective_function(args):
            """
            Logistic Regression objective function J (to minimize) for binary classification task:
            regularized average logistic loss of the training samples' scores
            :param args: Numpy 1-D array containing input [w,b], with 'w' of dimension K and 'b' scalar value
             (K is the number of features)
            :return: the scalar value of the objective function J(w,b)
            """
            # retrieve input parameters
            w, b = args[:-1], args[-1]

            # map labels from (0,1) to (-1,+1)
            z = 2 * labels - 1

            # compute train samples scores
            samples_scores = compute_scores(data_matrix, w, b)

            # compute obj function value

            regularization_term = 1.0 / 2.0 * self._lambda * (np.linalg.norm(w) ** 2)

            if self.rebalancing_true_prior_probability is not None:
                r_pi_T = self.rebalancing_true_prior_probability
                class_0_scores = samples_scores[labels == 0]
                class_0_labels = z[labels == 0]
                class_1_scores = samples_scores[labels == 1]
                class_1_labels = z[labels == 1]

                # compute balanced logistic loss for each class
                avg_logistic_loss_class_0 = (
                        (1.0 - r_pi_T) * np.mean(np.logaddexp(0, -class_0_labels * class_0_scores)))

                avg_logistic_loss_class_1 = (
                        r_pi_T * np.mean(np.logaddexp(0, -class_1_labels * class_1_scores)))

                avg_logistic_loss = avg_logistic_loss_class_0 + avg_logistic_loss_class_1
            else:
                avg_logistic_loss = np.mean(np.logaddexp(0, -z * samples_scores))

            return regularization_term + avg_logistic_loss

        x0 = np.zeros(data_matrix.shape[0] + 1)

        # minimize objective function
        min_point, min_value, opt_info = fmin_l_bfgs_b(avg_LogLoss_objective_function, x0, approx_grad=True)

        # save model parameters and results
        self.w = min_point[:-1]
        self.b = min_point[-1]
        self.obj_function_min_value = min_value
        self.opt_info = opt_info

    def predict(self, data_matrix, true_prior_probability=None, error_costs=None):
        """
        Predict the labels of test samples using the Logistic Regression
        model parameters (w,b) already estimated
        :param data_matrix: 2-D Numpy array containing one test sample per column
        :param true_prior_probability: (optional) prior probability of the True class (label=1),
         if not provided, classes are considered as balanced (default: 0.5)
        :param error_costs: (optional) bayes prediction costs of the two classes: cost of false negatives
         and cost of false positives; if not provided, unit costs are considered (default: (1.0, 1.0))
        """
        # preprocess test samples
        data_matrix = super().predict(data_matrix)

        self.test_scores = compute_scores(data_matrix, self.w, self.b)

        if self.rebalancing_true_prior_probability is not None:
            training_prior = self.rebalancing_true_prior_probability
        else:
            # training prior is the empirical prior
            training_prior = self.empirical_prior

        # remove training prior log odds to get llr
        self.test_scores -= np.log(training_prior / (1.0 - training_prior))

        # calibrate scores
        self.test_scores = self.calibrate(self.test_scores)

        # compute the threshold
        if true_prior_probability is None:
            true_prior_probability = 0.5

        if error_costs is None:
            error_costs = (1.0, 1.0)

        pi_T = true_prior_probability  # application prior
        C_fn, C_fp = error_costs

        # compute the threshold taking bayes costs and application prior into account
        threshold = -np.log(pi_T * C_fn) + np.log((1.0 - pi_T) * C_fp)

        super()._internal_predict(threshold)

    def __str__(self):

        return "Binary Logistic Regression Classifier (λ=%s%s)%s%s" % (
            self._lambda,
            (", %s" % ", ".join([str(x) for x in self.preprocessors])) if len(self.preprocessors) > 0 else "",
            " - class rebalancing (pi_1 = %f)" % self.rebalancing_true_prior_probability
            if self.rebalancing_true_prior_probability is not None else "",
            " with score calibration (pi=%f)"
            % self.score_calibrators[0].application_true_prior if len(self.score_calibrators) > 0 else ""
        )


def compute_scores(X, w, b):
    """
    Compute linear scores of some samples, given 'w' and 'b' model parameters
    :param X: 2-D Numpy array of samples (one per column)
    :param w: 1-D Numpy array model parameter of dimension K (# features)
    :param b: scalar model parameter
    :return: a 1-D Numpy array containing the ordered scores, one for each sample
    """
    return np.matmul(vrow(w), X).flatten() + b
