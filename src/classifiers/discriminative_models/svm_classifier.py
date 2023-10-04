import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from src.classifiers.binary_classifier import BinaryClassifier
from src.utilities.arrays import vcol
from src.utilities.kernels import LinearKernel


# noinspection PyMethodOverriding
class SvmClassifier(BinaryClassifier):
    """
    Support Vector Machine model for binary classification tasks and generic kernel
    """
    def __init__(self, c, k=1.0, kernel=None, rebalancing_true_prior_probability=None,
                 preprocessors=None, score_calibrators=None):
        """
        Create a new SVM classifier with the specified hyperparameter 'c', weighting the misclassified
        points contribute, and a custom kernel function to produce an n-dimensional hyperplane
        :param c: SVM **hyperparameter** to weight the misclassified points contribute with respect to
         the hyperplane margin width (for high values, the importance of minimizing misclassified
         points becomes more relevant, rather than finding a large margin)
        :param k: (optional) **hyperparameter** used to add a regularized bias:
         k^2 is added to the kernel function
        :param kernel: (optional) kernel object used to compute dot products between samples
         in an expanded space; if None, the linear kernel is used, and this becomes a Linear
         SVM classifier. It must have a .dot() method receiving two 1-D Numpy arrays and returning
         their dot product (a scalar) in the expanded features' space
        :param rebalancing_true_prior_probability: (optional) prior probability of True class
         used to rebalance classes hinge loss during training
        :param preprocessors: (optional) list of (ordered) preprocessors to be applied to input data
         before training and prediction phases for preprocessing strategies
        :param score_calibrators: (optional) list of (ordered) score calibration models to be applied to
         output scores before comparing them with the threshold and making predictions
        """
        super().__init__(preprocessors=preprocessors, score_calibrators=score_calibrators)

        self.empirical_prior = None    # empirical prior of the True (1) class
        self.rebalancing_true_prior_probability = rebalancing_true_prior_probability

        # hyperparameters
        self.c = c          # SVM hyperparameter
        self.k = k          # k^2 = eps

        if kernel is None:
            kernel = LinearKernel()
        self.kernel = kernel    # kernel function used to compute dot products in the expanded features space

        # intermediate results
        self.H = None               # dual problem matrix H: H[i,j] = z[i] * z[j] * np.dot(x[i],x[j])
        self.opt_info = None
        self.dual_loss = None       # max value of the objective function (to be maximized) of the dual problem

        # dual model parameters
        self.alpha = None                   # 1-D Numpy array with optimal Lagrange multipliers (alpha)
        self.support_vectors = None         # 2-D Numpy array containing only (extended) Support Vectors, one per column
        self.support_vectors_labels = None  # 1-D Numpy array containing the Support Vectors' labels
        self.support_vectors_alpha = None   # 1-D Numpy array containing the Support Vectors' alpha lagrange multipliers

        # predictions (defined in BinaryClassifier)
        # self.test_scores = None         # test samples' scores computed with the *dual* formulation (using 'alpha')
        # self.predicted_labels = None    # predictions made using the scores of the *dual* formulation

    def clone(self):
        clone = SvmClassifier(
            self.c,
            k=self.k,
            kernel=self.kernel,
            rebalancing_true_prior_probability=self.rebalancing_true_prior_probability,
            preprocessors=list(self.preprocessors),
            score_calibrators=list(self.score_calibrators)
        )

        super()._internal_clone(clone)
        clone.empirical_prior = self.empirical_prior
        clone.H = self.H
        clone.opt_info = self.opt_info
        clone.dual_loss = self.dual_loss
        clone.alpha = np.copy(self.alpha) if self.alpha is not None else None
        clone.support_vectors = np.copy(self.support_vectors) \
            if self.support_vectors_labels is not None else None
        clone.support_vectors_labels = np.copy(self.support_vectors_labels) \
            if self.support_vectors_labels is not None else None
        clone.support_vectors_alpha = np.copy(self.support_vectors_alpha) \
            if self.support_vectors_alpha is not None else None
        return clone

    def train(self, data_matrix, labels):
        """
        Train the SVM model computing the model parameters defining the decision boundary hyperplane
        :param data_matrix: 2-D Numpy array having one train sample per column
        :param labels: 1-D Numpy array having one label per train sample; labels must be only 0 and 1
        """
        data_matrix = super().train(data_matrix, labels)

        if self.unique_labels != [0, 1]:
            print("Error: SVM classifier works only with binary classification tasks with labels 0,1")
            exit(1)

        # * solve the dual problem, finding the Lagrange multipliers 'alpha' *

        # preprocess labels, mapping them from (0,1) to (-1,1)
        labels = 2*labels - 1

        num_train_samples = data_matrix.shape[1]
        num_samples_class_1 = data_matrix[:, labels == 1].shape[1]

        self.empirical_prior = float(num_samples_class_1) / float(num_train_samples)

        # compute matrix 'H' from extended features
        self.H = np.zeros((num_train_samples, num_train_samples))

        for i in range(num_train_samples):
            x_i = data_matrix[:,i]
            for j in range(num_train_samples):
                x_j = data_matrix[:,j]
                self.H[i,j] = labels[i] * labels[j] * (self.kernel.dot(x_i, x_j) + self.k**2)

        # * train model and save model parameters using preprocessed data, labels and H matrix *
        self.internal_train(data_matrix, labels, num_train_samples)

    def internal_train(self, expanded_data_matrix, labels, num_train_samples):
        """
        Internal function used by all SVM classifiers to train themselves, after having preprocessed train
        data, train labels and computed H matrix
        """
        # 1 - define dual problem objective function to minimize
        def soft_margin_SVM_objective_function(alpha):
            """
            Objective function of the (soft-margin) SVM dual problem with extended features
            (i.e. with only box constraints) to be minimized
            :param alpha: 1-D Numpy array 'alpha' containing all the Lagrange multipliers to be computed
            :return: the value of the objective function, for the specified alpha
            """
            alpha = vcol(alpha)             # reshape to 2-D column array
            return 1.0 / 2.0 * np.matmul(np.matmul(alpha.T, self.H), alpha) - alpha.sum()

        # 2 - define also the gradient of the objective function
        def grad_soft_margin_SVM_objective_function(alpha):
            """
            Gradient of the previous objective function
            :param alpha: 1-D Numpy array 'alpha' containing all the Lagrange multipliers to be computed
            :return: a 1-D Numpy array which is the gradient of the objective function for the specified alpha
            """
            alpha = vcol(alpha)     # rehape to a 2-D column array
            return np.matmul(self.H, alpha).flatten() - 1

        # 3 - define bounds for box constraints and starting point x0 of BFGS minimization algorithm

        if self.rebalancing_true_prior_probability is None:
            # use the same C constant for the constraints
            bounds = [(0, self.c) for _ in range(num_train_samples)]
        else:
            # compute two different constants C for the two classes
            r_pi_T = self.rebalancing_true_prior_probability
            c_0 = self.c * (1.0 - r_pi_T) / (1.0 - self.empirical_prior)
            c_1 = self.c * r_pi_T / self.empirical_prior

            # define different constraints for samples of different classes (class rebalancing)
            bounds = [(0, c_1 if class_label == 1 else c_0) for class_label in labels]

        x0 = np.zeros(num_train_samples)

        # 4 - minimize (dual problem) objective function
        opt_alpha, obj_func_min_value, self.opt_info = fmin_l_bfgs_b(
            soft_margin_SVM_objective_function,
            x0,
            fprime=grad_soft_margin_SVM_objective_function,
            bounds=bounds,
            factr=1.0
        )

        # 5 - save (dual) model parameters
        self.alpha = opt_alpha
        self.support_vectors = expanded_data_matrix[:, self.alpha > 0]
        self.support_vectors_labels = labels[self.alpha > 0]
        self.support_vectors_alpha = self.alpha[self.alpha > 0]

        # dual loss (opposite of the min value)
        self.dual_loss = -obj_func_min_value

    def predict(self, data_matrix, true_prior_probability=None, error_costs=None):
        """
        Predict labels of test samples using the trained SVM model: computing the score of
        each sample and assign label '1' if score > 0, '0' otherwise

        Note: here for SVM we don't need either application prior probabilities and error costs to make predictions
        :param data_matrix: 2-D Numpy array having one test sample per column
        :param true_prior_probability: (ignored by this classifier)
        :param error_costs: (ignored by this classifier)
        """
        # preprocess test samples
        data_matrix = super().predict(data_matrix)

        self.num_test_samples = data_matrix.shape[1]

        # Computing (test) samples scores using the dual formulation, by means of 'alpha' values
        self.test_scores = np.zeros(self.num_test_samples)

        for t in range(self.num_test_samples):
            x_t = data_matrix[:, t]
            score_sum = 0.0

            # compute score of sample x_t
            for i in range(len(self.support_vectors_alpha)):
                x_i = self.support_vectors[:, i]
                alpha_i = self.support_vectors_alpha[i]
                z_i = self.support_vectors_labels[i]
                score_sum += alpha_i * z_i * (self.kernel.dot(x_i, x_t) + self.k**2)

            self.test_scores[t] = score_sum        # save score

        # calibrate scores
        self.test_scores = self.calibrate(self.test_scores)

        if len(self.score_calibrators) == 0:
            # we are not calibrating SVM scores
            threshold = 0
        else:
            # we are calibrating SVM scores (typically with PW LR)
            # -> scores are thresholded with working point threshold

            # compute the threshold
            if true_prior_probability is None:
                true_prior_probability = 0.5

            if error_costs is None:
                error_costs = (1.0, 1.0)

            pi_T = true_prior_probability  # application prior
            C_fn, C_fp = error_costs

            threshold = -np.log(pi_T * C_fn) + np.log((1.0 - pi_T) * C_fp)

        super()._internal_predict(threshold)

    def __str__(self):
        return "%s Support Vector Machine Classifier (c=%s, k=%s%s)%s%s" % (
            self.kernel,
            self.c,
            self.k,
            (", %s" % ", ".join([str(x) for x in self.preprocessors])) if len(self.preprocessors) > 0 else "",
            " - class rebalancing (pi_1 = %f)" % self.rebalancing_true_prior_probability
            if self.rebalancing_true_prior_probability is not None else "",
            " with score calibration (pi=%f)"
            % self.score_calibrators[0].application_true_prior if len(self.score_calibrators) > 0 else ""
        )
