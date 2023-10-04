import numpy as np

# file containing kernel functions

class Kernel:
    def dot(self, x1, x2):
        """
        Compute the dot product between vectors x1 and x2 in the kernel's expanded features space
        :param x1: 1-D Numpy array
        :param x2: 1-D Numpy array
        :return: a scalar representing the dot product in the kernel's space
        """
        pass

    def __str__(self):
        pass


class LinearKernel(Kernel):
    """
    Linear Kernel computing dot product in linear space, that is the classic dot product
    """
    def dot(self, x1, x2):
        return np.dot(x1, x2)

    def __str__(self):
        return "Linear"


class PolynomialKernel(Kernel):
    """
    Polynomial Kernel computing dot product as (x1*x2 + c)^d
    """
    def __init__(self, degree, c=1.0):
        """
        Create a new Polynomial Kernel of the specified degree and a custom hyperparameter 'c'
        :param degree: integer degree of the polynomial space (hyperparameter)
        :param c: (default: 1.0) constant hyperparameter added to the linear dot product
        """
        # hyperparameters
        self.degree = degree
        self.c = c

    def dot(self, x1, x2):
        """
        Compute dot product in the polynomial space
        :return: (x1*x2 + c)^degree
        """
        return (np.dot(x1,x2) + self.c)**self.degree

    def __str__(self):
        return "Poly (d=%d, c=%.1f)" % (self.degree, self.c)


class RadialBasisFunctionKernel(Kernel):
    """
    Radial Basis Function (RBF) Kernel computing dot products as exp(-gamma * |x1-x2|^2)
    """
    def __init__(self, gamma=1.0):
        """
        Create a new RBF kernel with the specified hyperparameter 'gamma'
        :param gamma: hyperparameter weighting the distance between vectors
        """
        self.gamma = gamma

    def dot(self, x1, x2):
        """
        Compute dot product in an infinite dimensional space, according to the RBF kernel
        :return: exp(-gamma * |x1-x2|^2)
        """
        return np.exp(-self.gamma * np.linalg.norm(x1-x2)**2)

    def __str__(self):
        return "RBF (gamma=%s)" % self.gamma
