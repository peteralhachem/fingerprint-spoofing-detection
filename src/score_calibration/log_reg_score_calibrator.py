import numpy as np

from src.classifiers.discriminative_models.binary_log_reg_classifier import BinaryLogisticRegressionClassifier
from src.utilities.arrays import vrow

class BinaryLogisticRegressionScoreCalibrator:
    def __init__(self, w=None, b=None, application_true_prior=None):
        """
        Create a new Score Calibrator using prior-weighted logistic regression for binary classification tasks,
        """
        self.binary_logistic_regression_classifier = None  # base model
        self.cross_validator = None

        # calibration params
        self.w = w
        self.b = b
        self.application_true_prior = application_true_prior

        self.big_model = None       # LR model trained over all the training samples
        self.test_scores = None
        self.predicted_labels = None

    def train(self, scores_matrix, labels, k, application_true_prior, working_point, cross_validator):
        """
        Train the prior-weighted logistic regression score calibrator over the provided scores
        using k-fold cross validation, and specifying a working point for the application
        :param scores_matrix: 2-D Numpy array containing the (multidimensional) scores to train the score calibrator.
         Each column represent one score; it can be 1-dimensional, if it comes from a single model, or multidimensional,
         if it wraps scores coming from multiple models, for model fusions.
        :param labels: 1-D Numpy array containing the ordered labels corresponding to the scores
        :param k: number of folds for k-fold cross validation
        :param application_true_prior: application prior of true class for which the score calibration is optimized
        :param working_point: target working point
        :param cross_validator: new instance of cross validator
        """
        # save train prior
        self.application_true_prior = application_true_prior

        # setup cross validator with the proper folds
        self.cross_validator = cross_validator
        self.cross_validator.k_folds(scores_matrix, labels, k, shuffle=True, seed=0)

        # create a new logistic regression classifier
        self.binary_logistic_regression_classifier = BinaryLogisticRegressionClassifier(
            rebalancing_true_prior_probability=application_true_prior
        )

        # start cross validation to train the logistic regression models and
        # obtain the scores useful for performance analysis
        self.cross_validator.cross_validation(
            [self.binary_logistic_regression_classifier],
            [working_point]
        )

        # retrieve the trained clone of the logistic regression classifier from the cross validator
        self.binary_logistic_regression_classifier = self.cross_validator.models_per_working_point[0][0]

        # * create a big model trained over all scores, and save its final params *
        self.big_model = BinaryLogisticRegressionClassifier()
        self.big_model.train(scores_matrix, labels)

        # save calibration params
        self.w = self.big_model.w    # 1-D array
        self.b = self.big_model.b    # scalar


    def calibrate(self, scores):
        """
        Calibrate the provided scores according to the trained score calibrator
        :param scores: 2D or 1D Numpy array containing the ordered scores,
         even multidimensional ones, one for each column
        :return: 1-D Numpy array of the new calibrated scores
        """
        if scores.ndim == 1:
            scores = vrow(scores)   # make it 2D

        log_odds = np.log(self.application_true_prior) - np.log(1.0 - self.application_true_prior)

        return np.matmul(vrow(self.w), scores).flatten() + self.b - log_odds

    def predict(self, data_matrix, true_prior_probability=None, error_costs=None):
        # calibrate and predict
        self.big_model.predict(data_matrix, true_prior_probability=true_prior_probability, error_costs=error_costs)
        self.test_scores = self.big_model.test_scores
        self.predicted_labels = self.big_model.predicted_labels

    def error(
        self,
        over_cumulative_predictions=False
    ):
        return self.binary_logistic_regression_classifier.error(
            over_cumulative_predictions=over_cumulative_predictions
        )

    def DCF(
        self,
        true_prior_probability=0.5,
        error_costs=(1.0,1.0),
        over_cumulative_predictions=False
    ):
        return self.binary_logistic_regression_classifier.DCF(
            true_prior_probability=true_prior_probability,
            error_costs=error_costs,
            over_cumulative_predictions=over_cumulative_predictions
        )

    def min_DCF(
        self,
        true_prior_probability=0.5,
        error_costs=(1.0,1.0),
        real_labels=None,
        over_cumulative_predictions=False
    ):
        return self.binary_logistic_regression_classifier.min_DCF(
            true_prior_probability=true_prior_probability,
            error_costs=error_costs,
            real_labels=real_labels,
            over_cumulative_predictions=over_cumulative_predictions
        )
