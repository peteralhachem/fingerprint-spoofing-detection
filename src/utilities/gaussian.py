import numpy as np
from scipy.special import logsumexp

from src.utilities.arrays import vcol
from src.utilities.statistics import mean_and_covariance_of, mean_and_features_variances_of


# * Multivariate Gaussian Density *

def log_MGD(x, mu, sigma, diagonal_matrix=False):
    """
    Compute the natural logarithm of the Multivariate Gaussian Density (MGD) value
    for a set of samples, specifying the mean and the covariance matrix of the distribution
    :param x: 2-D Numpy array containing the samples on the columns (one sample per column)
    :param mu: 2-D Numpy (column) array, representing the mean of the MGD function
    :param sigma: 2-D Numpy array, representing the covariance matrix of the MGD function;
    if diagonal_matrix is True, this must be a 1-D Numpy array of the features variances
    :param diagonal_matrix: (optional) if this is True, sigma must be just the 1-D diagonal
    of the covariance matrix (1-D Numpy array) and the computation is optimized
    :return: a 1-D Numpy array containing the ordered values of the log MGD for the samples
    """
    num_dimensions = x.shape[0]
    pi = np.pi
    x_centered = x - mu

    if not diagonal_matrix:
        _, log_determinant = np.linalg.slogdet(sigma)
        precision_matrix = np.linalg.inv(sigma)

        # compute the log density for all the samples leveraging broadcasting
        log_density_values = -float(num_dimensions)/2.0 * np.log(2.0*pi)      \
                             -1.0/2.0 * log_determinant                       \
                             -1.0/2.0 * (np.matmul(x_centered.T, precision_matrix) * x_centered.T).sum(axis=1)
    else:
        sigma = vcol(sigma)          # (num_dim, 1)
        log_sigma = np.log(sigma)

        # compute the log density for all the samples' features leveraging broadcasting
        log_density_values = (-1.0/2.0 * np.log(2.0*pi)
                              -1.0/2.0 * log_sigma
                              -1.0/2.0 * (x_centered**2 / sigma)).sum(axis=0)

    return log_density_values    # return a 1-D array


def MGD(x, mu, sigma, diagonal_matrix=False):
    """
    Compute the Multivariate Gaussian Density (MGD) value for a set of samples,
    specifying the mean and the covariance matrix of the distribution
    :param x: 2-D numpy array containing the samples on the columns (one sample per column)
    :param mu: 2-D numpy (column) array, representing the mean of the MGD function
    :param sigma: 2-D numpy array, representing the covariance matrix of the MGD function;
    if diagonal_matrix is True, this must be a 1-D Numpy array of the features variances
    :param diagonal_matrix: (optional) if this is True, sigma must be just the 1-D diagonal
    of the covariance matrix (1-D Numpy array) and the computation is optimized
    :return: a 1-D numpy array containing the ordered values of the MGD for the samples
    """
    log_values = log_MGD(x, mu, sigma, diagonal_matrix)
    return np.exp(log_values)


# * Gaussian Mixture Models *

def GMM_log_joint_densities(x, components_params):
    """
    Compute the log joint densities of a set of samples having a Gaussian Mixture Model (GMM)
     distribution and its components
    :param x: 2-D Numpy array containing the set of samples, one per column
    :param components_params: iterable of tuples, one per GMM component, each having (in order)
     the **weight** (scalar), the **mean** (2-D Numpy column array) and **the covariance matrix** (2-D Numpy array)
     of the component Gaussian distribution
    :return: a 2-D Numpy array containing the values of log joint densities, having one row for each
     component and one column for each sample: (i,j) is the joint density of sample 'j' and component 'i'
    """
    components_conditional_log_likelihoods = []
    components_weights = []

    for weight, mean, covariance_matrix in components_params:
        # compute the log-density of the Gaussian distribution for all the samples
        component_log_densities = log_MGD(x, mean, covariance_matrix)
        # save densities and weights
        components_conditional_log_likelihoods.append(component_log_densities)
        components_weights.append(weight)

    # create a matrix having in (i,j) the conditional log-likelihood of the j-th sample for i-th component
    components_conditional_log_likelihoods = np.vstack(components_conditional_log_likelihoods)

    # create a Numpy column array of weights
    components_weights = vcol(np.array(components_weights))

    # compute the joint densities of each sample and each component
    components_joint_densities = components_conditional_log_likelihoods + np.log(components_weights)

    return components_joint_densities


def log_GMM(x, components_params):
    """
    Compute the log-density of a set of samples having a Gaussian Mixture Model (GMM) distribution
    :param x: 2-D Numpy array containing the set of samples, one per column
    :param components_params: iterable of tuples, one for each GMM component, each having (in order)
     the **weight**, the **mean** and **the covariance matrix** of the component Gaussian distribution
    :return: a 1-D Numpy array containing the values of log-density, one for each sample
    """
    # compute the joint densities of each sample and each component
    components_log_joint_densities = GMM_log_joint_densities(x, components_params)

    # compute the log marginal densities over the components, to obtain the values of the GMM density
    log_marginal_densities = logsumexp(components_log_joint_densities, axis=0)

    return log_marginal_densities


# * Density Estimation *
def maximum_likelihood_density_estimation_for(data, naive_bayes_assumption=False):
    """
    Estimate a Multivariate Gaussian density distribution for the provided data
    using the Maximum Likelihood optimization method
    :param data 2-D Numpy array containing one data sample per column
    :param naive_bayes_assumption (optional) specify if using naive bayes assumption for density estimation
    :return: mean and covariance matrix of the estimated Multivariate Gaussian density,
    respectively as a 1-D Numpy array and a 2-D Numpy array; if naive bayes assumption is True,
    the second value is a 1-D Numpy array with the features variances
    """
    return mean_and_covariance_of(data) if not naive_bayes_assumption else mean_and_features_variances_of(data)
