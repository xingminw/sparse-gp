"""
This code is modified based on
    https://github.com/plgreenLIRU/Gaussian-Process-SparseGP-Tutorial
"""

import time
import random
import numpy as np
from scipy.optimize import minimize
from .sparse_gp import SparseGP
from tqdm import tqdm


class SparseVariationalGP(SparseGP):
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

    def train(self, candidate_num):
        start_time = time.time()                    # Note the time when training began
        theta = np.array([self.width_l, self.sigma])  # GP hyperparameters

        # Loop over all the candidate sparse points
        new_selected_x, new_selected_y, lb_best = None, None, None
        for r in range(candidate_num):

            # Randomly select the locations of the candidate sparse points
            indices = random.sample(range(self.points_num), self.inducing_num)
            candidate_x = self.points_x[indices]

            # Define arguments to be passed to 'NegLowerBound' and
            # evaluate the lower bound for candidate sparse points
            a = candidate_x
            lb = - self._get_neg_lower_bound(theta, a)

            if r == 0:  # For the first set of candidate points
                new_selected_x = candidate_x  # Define optimum sparse inputs found so far
                new_selected_y = self.points_y[indices]  # Define optimum sparse outputs found so far
                lb_best = lb  # Store maximum lower bound found so far
            else:
                if lb > lb_best:  # If lower bound is largest we've seen
                    new_selected_x = candidate_x  # Define optimum sparse inputs found so far
                    new_selected_y = self.points_y[indices]  # Define optimum sparse outputs found so far
                    lb_best = lb  # Store maximum lower bound found so far

        # Update hyperparameters using scipy.minimize
        # Arguments needed as input to 'minimize' function
        a = new_selected_x

        b1 = (1e-3, 3)  # Bounds on length scale
        b2 = (1e-3, 1)  # Bounds on noise standard deviation
        bnds = (b1, b2)

        # Search for optimum hyperparameters
        sol = minimize(self._get_neg_lower_bound, x0=theta, args=(a,),
                       method='SLSQP', bounds=bnds)
        theta = sol.x

        # Find final gram matrix (and its inverse)
        self.inducing_points_x = new_selected_x
        self.inducing_points_y = new_selected_y
        self.width_l, self.sigma = theta  # Extract hyperparameters
        self.update_matrices()

        elapsed_time = time.time() - start_time     # Time taken for training
        return lb_best, elapsed_time

    def train_greedy(self, candidate_num):
        start_time = time.time()  # Note the time when training began

        # Loop over all the candidate sparse points
        selected_x, selected_y, active_indices, lb_best = [], [], [], None
        for _ in tqdm(range(self.inducing_num)):
            # Randomly select the locations of the candidate sparse points
            indices = random.sample(range(self.points_num), candidate_num + self.inducing_num)
            remaining_indices = []
            for idx in indices:
                if not (idx in active_indices):
                    remaining_indices.append(idx)

            theta = np.array([self.width_l, self.sigma])  # GP hyperparameters

            new_selected_x, new_selected_y, new_selected_index = None, None, None
            for idx in remaining_indices:
                new_x, new_y = self.points_x[idx], self.points_y[idx]
                # Define arguments to be passed to 'NegLowerBound' and
                # evaluate the lower bound for candidate sparse points
                a = np.vstack(selected_x + [new_x])
                lb = - self._get_neg_lower_bound(theta, a)
                if new_selected_x is None:
                    new_selected_x = new_x
                    new_selected_y = new_y
                    new_selected_index = idx
                    lb_best = lb
                else:
                    if lb > lb_best:
                        new_selected_x = new_x
                        new_selected_y = new_y
                        new_selected_index = idx
                        lb_best = lb
            selected_x += [new_selected_x]
            selected_y += [new_selected_y]
            active_indices += [new_selected_index]
            # print("Current active indices", active_indices)

            # Update hyperparameters using scipy.minimize
            # Arguments needed as input to 'minimize' function
            a = selected_x

            # Search for optimum hyperparameters
            sol = minimize(self._get_neg_lower_bound, x0=theta, args=(a,),
                           bounds=((1e-3, 3), (1e-3, 3)),
                           method='L-BFGS-B')
            theta = sol.x
            self.width_l, self.sigma = theta

        # Find final gram matrix (and its inverse)
        self.inducing_points_x = np.vstack(selected_x)
        self.inducing_points_y = np.vstack(selected_y)
        self.update_matrices()

        elapsed_time = time.time() - start_time  # Time taken for training
        return lb_best, elapsed_time

    def _get_neg_lower_bound(self, theta, a):
        """

        :param theta:
        :param a:
        :return:
        """

        raw_points_x, raw_points_y = self.points_x, self.points_y
        selected_points_x = a
        width_l, sigma = theta

        # get matrix K_uu
        matrix_k_uu = self.kernel(self.get_dis(selected_points_x), width_l)         # Find K_uu

        # get the svd of matrix K_uu
        svd_k_uu_left, svd_k_uu_sigma, _ = np.linalg.svd(matrix_k_uu)
        inv_k_uu_sigma = np.diag(1.0 / svd_k_uu_sigma)

        # find the inverse of matrix K_uu
        inv_matrix_k_uu = svd_k_uu_left @ inv_k_uu_sigma @ svd_k_uu_left.T

        # Get matrix K_fu and K_uf
        matrix_k_fu = self.kernel(self.get_dis(raw_points_x, selected_points_x), width_l)  # Find K_fu
        matrix_k_uf = np.transpose(matrix_k_fu)  # Find K_MN

        # Matrix Q_ff = K_fu * invK_uu * K_uf
        matrix_q_ff = matrix_k_fu @ inv_matrix_k_uu @ matrix_k_uf

        # B is an array containing only diagonal elements of K_uu - A.
        # Note we assume diagonal elements of A are always equal to 1.
        matrix_b = np.zeros(self.points_num)
        for i in range(self.points_num):
            matrix_b[i] = 1 - matrix_q_ff[i, i]

        # Calculate the (negative) lower bound
        matrix_c = matrix_q_ff + np.eye(self.points_num) * sigma ** 2

        # compute the inverse of matrix C, method 1: svd decomposition
        # matrix_q_left = matrix_k_fu @ svd_k_uu_left @ np.sqrt(inv_k_uu_sigma)
        # svd_q_left, svd_q_sigma, _ = np.linalg.svd(matrix_q_left)
        #
        # enlarged_q_sigma = np.hstack([np.square(svd_q_sigma), np.zeros(self.points_num - self.inducing_num)])
        # inv_matrix_c_diagonal = 1.0 / (enlarged_q_sigma + sigma ** 2 * np.ones(self.points_num))
        # inv_matrix_c_diagonal = np.diag(inv_matrix_c_diagonal)
        # inv_matrix_c = svd_q_left @ inv_matrix_c_diagonal @ svd_q_left.T

        # method 2: matrix inverse lemma
        matrix_q_left = matrix_k_fu @ svd_k_uu_left @ np.sqrt(inv_k_uu_sigma)
        q_right_inv_mat = np.linalg.inv(sigma ** 2 * np.eye(self.inducing_num) + matrix_q_left.T @ matrix_q_left)
        inv_matrix_c = np.eye(self.points_num) - matrix_q_left @ q_right_inv_mat @ matrix_q_left.T
        inv_matrix_c /= sigma ** 2

        # method 3: directly perform inverse
        # inv_matrix_c = np.linalg.inv(matrix_c)

        sign, log_det_c = np.linalg.slogdet(matrix_c)
        log_det_c = sign * log_det_c
        nlb = 0.5 * log_det_c + 0.5 * raw_points_y.T @ inv_matrix_c @ raw_points_y

        # todo: this can be simply removed to get the projected process approximation
        nlb += (1 / (2 * sigma ** 2) * np.sum(matrix_b))
        return np.ravel(nlb)


def compare_matrix(mat1, mat2):
    print("Diff:", np.linalg.norm(mat1 - mat2))
