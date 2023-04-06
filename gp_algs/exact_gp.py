import numpy as np
from time import time
from .gp_base import GaussianProcesses
from scipy.optimize import minimize


class ExactGP(GaussianProcesses):
    def __init__(self):
        super().__init__()

    def init_trainer(self, initial_l, initial_sigma):
        self.width_l = initial_l
        self.sigma = initial_sigma

    def train(self):
        self.points_x = np.vstack(self.points_x)
        start_time = time()
        initial_theta = np.array([self.width_l, self.sigma])
        sol = minimize(self._neg_log_likelihood_func, x0=initial_theta,
                       # method='SLSQP',
                       method='L-BFGS-B'
                       )
        elapsed_time = time() - start_time
        self.width_l, self.sigma = sol.x
        self.update_matrices()
        return elapsed_time

    def _neg_log_likelihood_func(self, theta):
        """
        Get the negative log likelihood function

        :param theta:
        :return:
        """
        self.width_l, self.sigma = theta
        self.update_matrices()
        (sign, log_det_c) = np.linalg.slogdet(self.matrix_c)
        log_det_c = sign * log_det_c
        return 0.5 * log_det_c + 0.5 * np.transpose(self.points_y) @ self.inverse_c @ self.points_y
        # return log_det_c + 0.5 * np.dot(self.points_y, np.dot(self.inverse_c, self.points_y))

    def _neg_ll_derivative(self):
        """
        Jacobian matrix of the log likelihood function

        :return:
        """
        raise NotImplementedError
