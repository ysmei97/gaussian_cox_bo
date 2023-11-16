import numpy as np
import math
from scipy.special import eval_hermite as hermite


class RBFKernel:
    def __init__(self, theta0=1, theta1=0.01, theta2=0.01):
        """
        :param theta0: regularization
        :param theta1: bandwidth: 1 / (2 * sigma^2)
        :param theta2: length-scale: from the Gaussian measure nu = N(0, l^2 * I) 1 / (2 * l^2)
        """
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2

    def rbf_scalar(self, x1, x2):
        """
        :param x1: input grid/observation point
        :param x2: input grid/observation point
        :return: kernel function value k(x_1, x_2)
        """
        return self.theta0 * np.exp(-self.theta1 * np.sum((x1 - x2)**2))

    def rbf_dist(self, d):
        """
        :param d: input pairwise distance
        :return: kernel function value k(x_1, x_2)
        """
        return self.theta0 * np.exp(-self.theta1 * d **2)

    def rbf_vector(self, x, y):
        """
        :param x: input grid/observation vector
        :param y: input grid/observation vector
        :return: kernel matrix [K]_ij = k(x_i, y_j)
        """
        assert x.shape[0] == 1
        assert y.shape[0] == 1
        return self.theta0 * np.exp(-self.theta1 * (np.repeat(x ** 2, y.shape[1], axis=0).T + np.repeat(y ** 2, x.shape[1], axis=0) - 2 * x.T @ y))  # shape of x.shape[1] * y.shape[1]

    def rbf_eigenvalue(self, k):
        """
        :param k: order of (physicist's) Hermite polynomial
        :return: eigenvalue of the order k
        """
        a = 0.5 * self.theta1
        b = self.theta2
        c = np.sqrt(a ** 2 + 2 * a * b)
        A = a + b + c
        B = b / A
        eta = np.sqrt(2 * a / A) * B ** k
        return eta

    def rbf_eigenfunction(self, k, x):
        """
        :param k: order of (physicist's) Hermite polynomial
        :param x: input grid/observation point
        :return: eigenfunction of the order k
        """
        a = 0.5 * self.theta1
        b = self.theta2
        c = np.sqrt(a ** 2 + 2 * a * b)
        psi = 1 / np.sqrt(np.sqrt(a / c) * 2 ** k * math.factorial(k)) * np.exp(-(c - a) * x ** 2) * hermite(k, np.sqrt(2 * c) * x)
        return psi
