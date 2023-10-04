import math

from src.utilities.arrays import vcol
import numpy as np


def mean_and_covariance_of(data_matrix):
    """
    Compute mean and covariance matrix of some data samples
    :param data_matrix: 2-D numpy array containing one sample for each column
    :return: 2-D Numpy (column) array for the mean, and 2-D numpy array for the covariance matrix
    """
    # compute mean vector
    mu = data_matrix.mean(axis=1)
    mu = vcol(mu)   # reshape to 2-D array

    # compute covariance matrix
    n = data_matrix.shape[1]        # num of samples
    centered_data_matrix = data_matrix - mu
    covariance_matrix = 1.0/float(n) * np.matmul(centered_data_matrix, centered_data_matrix.T)

    return mu, covariance_matrix

def mean_and_features_variances_of(data_matrix):
    """
    Compute mean and feature variances of some data samples
    :param data_matrix: 2-D Numpy array containing one sample for each column
    :return: 2-D Numpy (column) array for the mean, and 1-D Numpy array for the features variances
    """
    # compute mean vector
    mu = data_matrix.mean(axis=1)
    mu = vcol(mu)  # reshape to 2-D array

    # compute features variances
    n = data_matrix.shape[1]
    centered_data_matrix = data_matrix - mu
    features_variances = 1.0/float(n) * (centered_data_matrix**2).sum(axis=1)

    return mu, features_variances

# effective prior - prior log odds - working point conversion

def effective_prior_of_working_point(pi_T, C_fn, C_fp):
    pi_T = float(pi_T)
    C_fn = float(C_fn)
    C_fp = float(C_fp)

    return pi_T*C_fn / (pi_T*C_fn + (1.0 - pi_T)*C_fp)

def prior_log_odd_of_working_point(w):
    pi_T, C_fn, C_fp = w
    effective_prior = effective_prior_of_working_point(pi_T, C_fn, C_fp)
    return np.log(effective_prior / (1.0 - effective_prior))


# others


# Function to compute Log base 2
def log2(x):
    return math.log10(x) / math.log10(2)


# Function to check if x is power of 2
def isPowerOfTwo(n):
    return math.ceil(log2(n)) == math.floor(log2(n))