import sys

import numpy as np
from src.utilities.arrays import vrow, vcol
from src.utilities.gaussian import GMM_log_joint_densities
from scipy.special import logsumexp
from src.utilities.statistics import mean_and_covariance_of, isPowerOfTwo, log2


class GmmLbgEstimator:
    """
    An object implementing the LBG algorithm for the estimation of Gaussian Mixture Models
    """
    def __init__(
        self, max_components,
        diagonal_covariance=False, tied_covariance=False,
        min_eigenvalue=None, starting_Gaussian_params=None,
        min_increment=10**(-6), displacement_alpha=0.1
    ):
        """
        Create a new GMM LBG estimator, specifying the max number of components for the GMM distribution and
        the other LBG parameters.
        :param max_components: the maximum number of components to be estimated by the LBG algorithm
        :param diagonal_covariance: (default: False) indicate if the covariance matrices of the
         GMM components have to be constrained to diagonal matrices
        :param tied_covariance: (default: False) indicate if it has to be estimated the same covariance
         matrix for all the GMM components
        :param min_eigenvalue: (default: None) the minimum value of the covariance matrices' eigenvalues; if provided,
         all the eigenvalues of the covariance matrices will be forced to be greater or equal than this value
        :param starting_Gaussian_params: a tuple of Gaussian parameters (mean, covariance)
         - 'mean' -> 2-D Numpy column array
         - 'covariance' -> 2-D Numpy array
         NOTE: if None, it is initialized with the empirical mean and the empirical covariance of the dataset
        :param min_increment: (default: 10^-6) the threshold that defines the stopping criterion at each iteration:
         the EM algorithm stops when the average log-likelihood increases by a value **lower** than this threshold
        :param displacement_alpha: (default: 0.1) factor to scale the displacement vector
         before each split of the LBG algorithm
        """
        self.max_components = max_components

        # LBG algorithm params
        self.starting_Gaussian_params = starting_Gaussian_params
        self.min_increment = min_increment
        self.displacement_alpha = displacement_alpha

        # constraints on the components' covariance matrices
        self.min_eigenvalue = min_eigenvalue
        self.diagonal_covariance = diagonal_covariance
        self.tied_covariance = tied_covariance

        # model parameters
        self.all_estimated_GMM_params_and_likelihoods = None    # set of GMM parameters (and corresponding likelihoods)
                                                                # of the estimated distribution for all the
                                                                # LBG algorithm iterations

        self.last_data_ref = None       # fundamental to understand if data is the same as before,
                                        # and the current GMM params are still valid for the new provided dataset

    def estimate(self, original_data_matrix_ref, preprocessed_data_matrix, num_components):
        """
        Estimate the GMM distribution of the provided data, using the specified number of components, and
        considering the provided reference to the original data (if it is not changed, use cached results, instead
        of applying again the LBG algorithm)
        :param original_data_matrix_ref: reference to the data object which is unchanged between similar models
        :param preprocessed_data_matrix: actual preprocessed data whose GMM distribution has to be estimated
        :param num_components: number of GMM subcomponents to be estimated
        :return: the parameters of the requested estimated GMM model
        """
        if not (num_components > 0 and isPowerOfTwo(num_components)):
            print("Error: number of GMM components must be a power of 2", file=sys.stderr)
            exit(1)

        if num_components > self.max_components:
            print("Error: number of GMM components (%d) exceeded for the current GmmLbgEstimator. Max: %d"
                  % (num_components, self.max_components), file=sys.stderr)
            exit(2)

        # check if the provided data is the same as before (results are already cached)
        if self.last_data_ref is None or self.last_data_ref is not original_data_matrix_ref:
            # estimate again GMM distribution with LBG algorithm
            self.all_estimated_GMM_params_and_likelihoods = gmm_LBG_estimation(
                preprocessed_data_matrix,
                self.max_components,
                diagonal_covariance=self.diagonal_covariance,
                tied_covariance=self.tied_covariance,
                min_eigenvalue=self.min_eigenvalue,
                starting_Gaussian_params=self.starting_Gaussian_params,
                min_increment=self.min_increment,
                displacement_alpha=self.displacement_alpha
            )
            # save new data ref
            self.last_data_ref = original_data_matrix_ref

        powerOfTwo = round(log2(num_components))
        estimated_gmm_components, _ = self.all_estimated_GMM_params_and_likelihoods[powerOfTwo]
        return estimated_gmm_components


# * LBG algorithm *

def gmm_LBG_estimation(data_matrix, max_components, starting_Gaussian_params=None,
                       min_increment=10**(-6), displacement_alpha=0.1, min_eigenvalue=None,
                       diagonal_covariance=False, tied_covariance=False):
    """
    Estimate the Gaussian Mixture Model (GMM) distribution of a set of samples using the
    LBG algorithm, specifying the parameters of the starting Gaussian density, the maximum
    number of components to be reached in the iterations and the stopping criterion. At each iteration,
    the EM algorithm is applied to estimate the GMM parameters with G components; then the process is
    repeated with 2G components, and so on until max_components is reached
    :param data_matrix: 2-D Numpy array having one column for each sample
    :param max_components: an integer number of maximum components to be reached before stopping the algorithm;
     NOTE: it must be a power of 2; if it is not, it will be rounded to the one immediately
     lower than the specified number
    :param starting_Gaussian_params: a tuple of Gaussian parameters (mean, covariance)
     - 'mean' -> 2-D Numpy column array
     - 'covariance' -> 2-D Numpy array
     NOTE: if None, it is initialized with the empirical mean and the empirical covariance of the dataset
    :param min_increment: (default: 10^-6) the threshold that defines the stopping criterion at each iteration:
     the EM algorithm stops when the average log-likelihood increases by a value **lower** than this threshold
    :param displacement_alpha: (default: 0.1) factor to scale the displacement vector before each split
    :param min_eigenvalue: (default: None) the minimum value of the covariance matrices' eigenvalues; if provided,
     all the eigenvalues of the covariance matrices will be forced to be greater than this value
    :param diagonal_covariance: (default: False) indicate if the covariance matrices of the GMM components have to
     be constrained to diagonal matrices
    :param tied_covariance: (default: False) indicate if it has to be estimated the same covariance matrix for
     all the GMM components
    :return: a list containing, for each iteration:
     - the estimated GMM parameters, as an iterable of (weight, mean, covariance_matrix)
     - the corresponding value of average log-likelihood for the samples
    """
    print("* Starting LBG algorithm... *")

    if starting_Gaussian_params is None:
        starting_Gaussian_params = mean_and_covariance_of(data_matrix)

    all_gmm_params_and_likelihoods = []     # result

    mu, sigma = starting_Gaussian_params
    starting_GMM_params = [(1.0, mu, sigma)]

    while len(starting_GMM_params) <= max_components:
        print("#%d " % len(starting_GMM_params), end="")

        # 1 - estimate the GMM params using the current ones as starting point
        estimated_GMM_params, log_likelihood = gmm_EM_estimation(data_matrix, starting_GMM_params,
                                                                 min_increment, min_eigenvalue,
                                                                 diagonal_covariance, tied_covariance)
        # save result
        all_gmm_params_and_likelihoods.append((estimated_GMM_params, log_likelihood))

        new_starting_GMM_params = []

        # 2 - split each tuple of Gaussian parameters into two new tuples
        for weight, mean, covariance_matrix in estimated_GMM_params:
            # compute the displacement vector
            U, s, Vh = np.linalg.svd(covariance_matrix)
            leading_eigenvector = U[:, 0:1]
            leading_eigenvalue = s[0]
            displacement = leading_eigenvector * np.sqrt(leading_eigenvalue) * displacement_alpha

            # splitting the component in two components halving the weight and displacing the mean
            new_starting_GMM_params.append((weight/2.0, mean - displacement, covariance_matrix))
            new_starting_GMM_params.append((weight/2.0, mean + displacement, covariance_matrix))

        # 3 - update parameters and repeat until max num of components is reached
        starting_GMM_params = new_starting_GMM_params

    print("Done!")
    return all_gmm_params_and_likelihoods


# * Expectation Maximization (EM) algorithm *

def gmm_EM_estimation(data_matrix, starting_GMM_params, min_increment, min_eigenvalue=None,
                      diagonal_covariance=False, tied_covariance=False):
    """
    Estimate the Gaussian Mixture Model (GMM) distribution of a set of samples using the
    Expectation Maximization (EM) algorithm, specifying a starting set of GMM parameters
    and the stopping criterion
    :param data_matrix: 2-D Numpy array having one column for each sample
    :param starting_GMM_params: iterable of GMM parameters: each element must be a tuple (weight, mean, covariance)
     - 'weight' -> scalar
     - 'mean' -> 2-D Numpy column array
     - 'covariance' -> 2-D Numpy array
    :param min_increment: the threshold that defines the stopping criterion: the algorithm stops when
    the average log-likelihood increases by a value **lower** than this threshold
    :param min_eigenvalue: (default: None) the minimum value of the covariance matrices' eigenvalues; if provided,
     all the eigenvalues of the covariance matrices will be forced to be greater than or equal to this value
    :param diagonal_covariance: (default: False) indicate if the covariance matrices of the GMM components have to
     be constrained to be diagonal matrices
    :param tied_covariance: (default: False) indicate if it has to be estimated the same covariance matrix for
     all the GMM components
    :return: the estimated GMM parameters (in the same format of the input starting parameters) and the corresponding
     value of average log-likelihood for the samples
    """
    # modify covariance matrices with the specified constraints
    constrain_covariances(starting_GMM_params, min_eigenvalue, diagonal_covariance)

    num_samples = data_matrix.shape[1]

    previous_log_likelihood = -np.inf
    previous_GMM_params = starting_GMM_params

    while True:
        # 1. E-step - compute the previous step *log-likelihood* and the new samples' *responsibilities*
        log_likelihood, responsibilities = E_step(data_matrix, previous_GMM_params)

        # check log-likelihood becomes higher, otherwise there is an error
        log_likelihood_increment = log_likelihood - previous_log_likelihood
        # if log_likelihood_increment < 0:
        #     print("The log-likelihood in the EM algorithm cannot become smaller", file=sys.stderr)
        #     return None

        # 2. check stopping criterion (avg log-likelihood lower than min_increment)
        if log_likelihood_increment < num_samples * min_increment:
            avg_log_likelihood = log_likelihood / num_samples
            return previous_GMM_params, avg_log_likelihood

        # 3. M-step - compute new GMM parameters
        new_GMM_params = M_step(data_matrix, responsibilities, tied_covariance)

        # 4. (if specified) modify covariance matrices according to the specified constraints
        constrain_covariances(new_GMM_params, min_eigenvalue, diagonal_covariance)

        # update GMM parameters and log-likelihood value
        previous_log_likelihood = log_likelihood
        previous_GMM_params = new_GMM_params


def E_step(data_matrix, GMM_params):
    """
    Execute the E-step of the EM algorithm computing the responsibilities of each sample for each class,
     and compute the previous step log likelihood
    :param data_matrix: 2-D Numpy array with one sample per column
    :param GMM_params: iterable of GMM parameters, each element must be a tuple (weight, mean, covariance)
    :return: the value of the samples' log-likelihood of the previous step, and a 2-D Numpy array
     containing the new step responsibilities: (i,j) is the responsibility of sample 'j' for the component 'i'
    """
    # compute log joint densities and log marginal densities
    components_log_joint_densities = GMM_log_joint_densities(data_matrix, GMM_params)
    log_marginal_densities = logsumexp(components_log_joint_densities, axis=0)

    # compute samples' GMM log-likelihood
    log_likelihood = log_marginal_densities.sum()

    # compute responsibilities
    log_posterior_components_probabilities = components_log_joint_densities - vrow(log_marginal_densities)
    responsibilities = np.exp(log_posterior_components_probabilities)

    return log_likelihood, responsibilities


def M_step(data_matrix, responsibilities, tied_covariance):
    """
    Execute the M-step of the EM algorithm estimating the new GMM parameters
    :param data_matrix: 2-D Numpy array with one sample per column
    :param responsibilities: 2-D Numpy array containing the current responsibilities: (i,j) is the
     responsibility of sample 'j' for the component 'i'
    :param tied_covariance: indicate if it has to be estimated the same covariance matrix for all the GMM components
    :return: an iterable of tuples containing the new GMM parameters; each tuple contains the parameters of a Gaussian
     component (weight, mean, covariance)
    """
    num_dimensions = data_matrix.shape[0]
    num_components = responsibilities.shape[0]
    num_samples = data_matrix.shape[1]

    # * compute statistics *

    # Zero order statistics
    z = responsibilities.sum(axis=1)    # 1-D Numpy array

    # First order statistics
    f = np.empty((num_dimensions, num_components))      # 2-D Numpy array
    for g in range(num_components):
        weights = responsibilities[g, :]
        f[:, g] = (data_matrix * vrow(weights)).sum(axis=1)

    # Second order statistics
    s = np.empty((num_dimensions, num_dimensions, num_components))      # 3-D Numpy array
    for g in range(num_components):
        samples_covariances = np.empty((num_dimensions, num_dimensions, num_samples))

        for i in range(num_samples):
            x_i = data_matrix[:, i]
            samples_covariances[:, :, i] = responsibilities[g, i] * np.matmul(vcol(x_i), vrow(x_i))

        # compute sum of matrices and save second order statistic
        s[:, :, g] = samples_covariances.sum(axis=2)

    # use statistics to compute new GMM parameters
    weights = z / z.sum()   # 1-D Numpy array
    means = f / vrow(z)     # 2-D Numpy array (one mean per column)
    covariances = [s[:,:,g]/z[g] - np.matmul(vcol(means[:,g]), vrow(means[:,g])) for g in range(num_components)]  # list

    if tied_covariance:
        # * estimate a unique covariance matrix for all the GMM components *
        unique_covariance = np.zeros((num_dimensions, num_dimensions))

        # compute the sum of the components' covariances weighted by the zero order statistics
        for g in range(num_components):
            unique_covariance += z[g] * covariances[g]

        # divide by the number of samples
        unique_covariance /= float(num_samples)

        # replace each covariance matrix with the new unique covariance matrix
        covariances = [unique_covariance] * num_components

    # return a tuple of (weight, mean, covariance_matrix) for each Gaussian component
    # - weight -> must be a scalar
    # - mean -> must be a 2-D Numpy **column** array
    # - covariance -> must be a 2-D Numpy array
    return [(weights[g], means[:, g:g+1], covariances[g]) for g in range(num_components)]


def constrain_covariances(GMM_params, min_eigenvalue, diagonal_covariance):
    """
    Modify in place each covariance matrix in a set of GMM params, such that each one's eigenvalues
     are greater than a given threshold
    :param GMM_params: iterable of tuples, one for each Gaussian component,
     each one having (weight, mean, covariance_matrix)
    :param min_eigenvalue: min threshold to be applied to each covariance matrix's eigenvalues
    :param diagonal_covariance: indicate if the covariance matrices of the GMM components have to
     be constrained to diagonal matrices
    :return: the same GMM parameters, in the same format, with the modified covariance matrices
    """
    num_components = len(GMM_params)

    # modify each component's covariance matrix
    for i in range(num_components):
        covariance_matrix = GMM_params[i][2]

        if diagonal_covariance:
            covariance_matrix = covariance_matrix * np.eye(covariance_matrix.shape[0])

        if min_eigenvalue is not None:
            # compute Singular Value Decomposition
            eigenvectors, eigenvalues, _ = np.linalg.svd(covariance_matrix)

            # constraint each eigenvalue to be greater than the min threshold
            eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue

            # re-build the covariance matrix using the new eigenvalues but the same eigenvectors
            covariance_matrix = np.matmul(eigenvectors, vcol(eigenvalues) * eigenvectors.T)

        # substitute the old covariance matrix with the new one
        GMM_params[i] = (GMM_params[i][0], GMM_params[i][1], covariance_matrix)
