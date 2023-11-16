import numpy as np
from scipy.stats import t


class StudentT:
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, kappa: float = 1, mu: float = 0):
        """
        StudentT distribution except normal distribution is replaced with the student T distribution
        https://en.wikipedia.org/wiki/Normal-gamma_distribution
        :param alpha: alpha in gamma distribution prior
        :param beta: beta in gamma distribution prior
        :param kappa: variance from normal distribution
        :param mu: mean from normal distribution
        """
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data: np.array):
        """
        Return the pdf function of the t distribution
        :param data: the datapoints to be evaluated (shape: 1 x D vector)
        :return:
        """
        return t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa)),
        )

    def update_theta(self, data: np.array, **kwargs):
        """
        Performs a bayesian update on the prior parameters, given data
        :param data: datapoints to be evaluated (shape: 1 x D vector)
        :param kwargs:
        :return:
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
