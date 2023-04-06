"""
This code is modified based on
    https://github.com/plgreenLIRU/Gaussian-Process-SparseGP-Tutorial
"""

import numpy as np
from .gp_base import GaussianProcesses


class SparseGP(GaussianProcesses):
    def __init__(self):
        # input params
        super().__init__()
        self.inducing_points_x = None
        self.inducing_points_y = None
        self.inducing_num = None

        self.matrix_k_uf = None
        self.matrix_k_fu = None
        self.matrix_k_uu = None
        self.inv_matrix_k_uu = None

    def train(self, *args):
        raise NotImplementedError

    def init_trainer(self, initial_l, initial_sigma, inducing_num):
        self.width_l = initial_l
        self.sigma = initial_sigma
        self.inducing_num = inducing_num

    def update_matrices(self):
        self.matrix_k_uu = self.kernel(self.get_dis(self.inducing_points_x), self.width_l)  # Find K_uu
        self.inv_matrix_k_uu = np.linalg.inv(self.matrix_k_uu)
        self.matrix_k_fu = self.kernel(self.get_dis(self.points_x, self.inducing_points_x), self.width_l)  # Find K_fu
        self.matrix_k_uf = np.transpose(self.matrix_k_fu)  # Find K_MN
        self.matrix_c = 1 / (self.sigma ** 2) * self.matrix_k_uf @ self.matrix_k_fu + self.matrix_k_uu
        self.inverse_c = np.linalg.inv(self.matrix_c)

    def predict(self, predict_x):
        if np.size(self.inducing_points_x[0]) == 1:
            inducing_distances = (self.inducing_points_x - predict_x) ** 2
        else:
            inducing_distances = np.sum((self.inducing_points_x - predict_x) ** 2, 1)

        matrix_k_us = self.kernel(inducing_distances, self.width_l)
        matrix_k_su = np.transpose(matrix_k_us)
        matrix_q_star_star = matrix_k_su @ self.inv_matrix_k_uu @ matrix_k_us

        matrix_k_us = self.kernel(inducing_distances, self.width_l)

        predict_y_mean = 1 / (self.sigma ** 2) * matrix_k_su @ self.inverse_c
        predict_y_mean = predict_y_mean @ self.matrix_k_uf @ self.points_y
        predict_y_var = 1 + self.sigma ** 2 - matrix_q_star_star
        predict_y_var += matrix_k_su @ self.inverse_c @ matrix_k_us
        predict_y_std = np.sqrt(predict_y_var)
        return predict_y_mean, predict_y_std

