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
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


class SparseVariationalGP(SparseGP):
    def __init__(self, variational: bool=True):
        # input params
        super().__init__()
        self.inducing_points_x = None
        self.inducing_points_y = None
        self.inducing_num = None

        self.matrix_k_uf = None
        self.matrix_k_fu = None
        self.matrix_k_uu = None
        self.inv_matrix_k_uu = None
        self.variational = variational

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

        # selection_time = time.time()
        # print("Seleciton time:", np.round(selection_time - start_time, 3))

        # Update hyperparameters using scipy.minimize
        # Arguments needed as input to 'minimize' function
        a = new_selected_x

        b1 = (1e-3, 3)  # Bounds on length scale
        b2 = (1e-3, 1)  # Bounds on noise standard deviation
        bnds = (b1, b2)

        # Search for optimum hyperparameters
        sol = minimize(self._get_neg_lower_bound, x0=theta, args=(a,),
                       method='SLSQP',
                       bounds=bnds)
        theta = sol.x
        # print("Opt params time:", np.round(time.time() - selection_time, 3))

        # Find final gram matrix (and its inverse)
        self.inducing_points_x = new_selected_x
        self.inducing_points_y = new_selected_y
        self.width_l, self.sigma = theta  # Extract hyperparameters
        self.update_matrices()

        elapsed_time = time.time() - start_time     # Time taken for training
        return lb_best, elapsed_time

    def train_greedy(self, candidate_num, stepwise_output=False):
        start_time = time.time()  # Note the time when training began
        rmse = []  # For stepwise training
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
            if stepwise_output:  # Incrementally increase number of inducing points and compute RMSE save results/plot
                self.inducing_points_x = np.vstack(selected_x)
                self.inducing_points_y = np.vstack(selected_y)
                self.update_matrices()
                predict_x = np.linspace(0, 10, 200)
                predict_y_mean = []
                predict_y_std = []
                for x in predict_x:
                    ym, ys = self.predict(x)
                    predict_y_mean.append(ym[0][0])
                    predict_y_std.append(ys[0][0])
                predict_y_mean = np.array(predict_y_mean)
                predict_y_std = np.array(predict_y_std)

                predict_y_points = []
                for x in self.points_x:
                    ym, ys = self.predict(x)
                    predict_y_points.append(ym[0][0])
                predict_y_points = np.array(predict_y_points)
                rmse.append(mean_squared_error(self.points_y, predict_y_points, squared=False))


                fig = plt.figure()
                plt.grid()
                plt.plot(self.points_x, self.points_y, '.', color='blue', label='Full dataset', zorder=0)
                plt.plot(self.inducing_points_x, self.inducing_points_y, 'o',
                         markeredgecolor='black', markerfacecolor='red',
                         markeredgewidth=1.5, markersize=10, label='Sparse dataset', zorder=10)
                plt.plot(predict_x, predict_y_mean, 'black', label='Sparse GP', linewidth=3, zorder=5)
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.title("Inducing Points: {}".format(len(selected_x)))
                plt.xlim([0,10])
                plt.ylim([-2,2])

        # Find final gram matrix (and its inverse)
        self.inducing_points_x = np.vstack(selected_x)
        self.inducing_points_y = np.vstack(selected_y)
        self.update_matrices()

        elapsed_time = time.time() - start_time  # Time taken for training
        if stepwise_output:
            return lb_best, elapsed_time, rmse
        else:
            return lb_best, elapsed_time

    def _get_neg_lower_bound(self, theta, a):
        """

        :param theta:
        :param a:
        :return:
        """

        raw_points_x, raw_points_y = self.points_x, self.points_y
        selected_points_x = a
        current_inducing_num = len(selected_points_x)
        width_l, sigma = theta

        # get matrix K_uu
        matrix_k_uu = self.kernel(self.get_dis(selected_points_x), width_l)         # Find K_uu
        inv_matrix_k_uu = np.linalg.inv(matrix_k_uu)

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

        # inv matrix, apply matrix inversion lemma: (QQ.T+s^2*I)^-1 = s^-2[I-*Q(s^2*I+Q.T Q)^-1Q.T]
        inv_matrix_left = np.linalg.cholesky(inv_matrix_k_uu)
        matrix_q_left = matrix_k_fu @ inv_matrix_left
        matrix_q_smaller = matrix_q_left.T @ matrix_q_left
        q_right_inv_mat = np.linalg.inv(sigma ** 2 * np.eye(current_inducing_num) + matrix_q_smaller)
        inv_matrix_c = np.eye(self.points_num) - matrix_q_left @ q_right_inv_mat @ matrix_q_left.T
        inv_matrix_c /= sigma ** 2

        # dete matrix, det(Im + AB) = det(In + BA) and det(aI_m) = a^m
        log_det_diff = np.log(1 / sigma) * 2 * self.points_num
        equiv_matrix_c = matrix_q_smaller / (sigma ** 2) + np.eye(current_inducing_num)
        sign, log_det_c = np.linalg.slogdet(equiv_matrix_c)
        log_det_c = log_det_c * sign - log_det_diff

        # # slow method to perform inverse (direct inversion, very slow)
        # matrix_c = matrix_q_ff + np.eye(self.points_num) * sigma ** 2
        # inv_matrix_c = np.linalg.inv(matrix_c)
        # #
        # # slow method to calculate the determinant (direct calculating logdet, very slow)
        # sign, log_det_c = np.linalg.slogdet(matrix_c)
        # log_det_c = sign * log_det_c

        nlb = 0.5 * log_det_c + 0.5 * raw_points_y.T @ inv_matrix_c @ raw_points_y
        if self.variational:
            nlb += (1 / (2 * sigma ** 2) * np.sum(matrix_b))
        return np.ravel(nlb)


def compare_matrix(mat1, mat2):
    print("Diff:", np.linalg.norm(mat1 - mat2))
