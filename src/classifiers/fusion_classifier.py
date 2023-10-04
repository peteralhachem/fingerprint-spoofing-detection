from src.classifiers.binary_classifier import BinaryClassifier
import sys
import numpy as np

from src.utilities.arrays import vrow


class BinaryFusionClassifier(BinaryClassifier):
    """
    Fusion classifier made of multiple classifiers for binary classification tasks
    """
    def __init__(self, classifiers, score_calibrators, preprocessors=None):
        """
        Create a fusion classifiers specifying its classifiers and its score calibrators,
        which produce 1-dimensional scores
        :param classifiers: list of classifiers to fuse
        :param score_calibrators: list of score calibrators to calibrate the classifiers' scores
        :param preprocessors: (optional) list of preprocessors to preprocess dats
        """
        super().__init__(
            preprocessors=preprocessors,
            score_calibrators=score_calibrators
        )

        self.classifiers = classifiers


    def clone(self):
        clone = BinaryFusionClassifier(
            classifiers=list(map(lambda classifier: classifier.clone(), self.classifiers)),
            score_calibrators=list(self.score_calibrators),
            preprocessors=list(self.preprocessors)
        )
        super()._internal_clone(clone)
        return clone

    def train(self, data_matrix, labels):
        self.__init__(
            classifiers=self.classifiers,
            score_calibrators=self.score_calibrators,
            preprocessors=self.preprocessors
        )

        if len(set(labels)) != 2:
            print("Error: to train a Binary Fusion Classifier you need a training set with (just) "
                  "2 different labels",
                  file=sys.stderr)
            return

        # preprocess data, if necessary
        data_matrix = super().train(data_matrix, labels)

        # train separately each classifier
        for classifier in self.classifiers:
            classifier.train(data_matrix, labels)


    def predict(self, data_matrix, true_prior_probability=None, error_costs=None):
        # preprocess test samples
        data_matrix = super().predict(data_matrix)

        scores = []

        for classifier in self.classifiers:
            # compute the scores of the internal classifiers
            classifier.predict(data_matrix, true_prior_probability=true_prior_probability, error_costs=error_costs)
            scores.append(classifier.test_scores)

        # join the scores in a 2D matrix
        scores = np.vstack(scores)

        for score_calibrator in self.score_calibrators[:-1]:
            scores = score_calibrator.calibrate(scores)

        if scores.ndim == 1:
            scores = vrow(scores)

        # (now scores is again a 2D array for sure)
        # last score calibrator is going to both calibrate the scores and predict the labels
        last_score_calibrator = self.score_calibrators[-1]
        last_score_calibrator.predict(
            scores,
            true_prior_probability=true_prior_probability,
            error_costs=error_costs
        )

        # save scores produced by the internal model of the score calibrator
        self.test_scores = last_score_calibrator.test_scores

        # save cumulative scores
        if self.cumulative_info["test_scores"] is None:
            self.cumulative_info["test_scores"] = []

        self.cumulative_info["test_scores"].append(self.test_scores)

        # save labels predicted by the internal model of the score calibrator
        self.predicted_labels = last_score_calibrator.predicted_labels


    def __str__(self):
        return "Fusion: %s" % ", ".join(map(lambda classifier: str(classifier), self.classifiers))
