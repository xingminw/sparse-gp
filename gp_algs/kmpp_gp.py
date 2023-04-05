"""
This code is modified based on
    https://github.com/plgreenLIRU/Gaussian-Process-SparseGP-Tutorial
"""

import time
import random
import numpy as np
from scipy.optimize import minimize
from .gp_base import GaussianProcesses
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum, row_norms

class SparseKmppGP(GaussianProcesses):
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

    def init_trainer(self, initial_l, initial_sigma, inducing_num):
        self.width_l = initial_l
        self.sigma = initial_sigma
        self.inducing_num = inducing_num

    def k_meansplusplus(self, X, Y, n_clusters, random_state=np.random.RandomState,
                        n_local_trials=None):  # Copied from sklearn library
        """Init n_clusters seeds according to k-means++, adapted from sklearn
        Parameters
        ----------
        X : array or sparse matrix, shape (n_samples, n_features)
            The data to pick seeds for. To avoid memory copy, the input data
            should be double precision (dtype=np.float64).
        n_clusters : integer
            The number of seeds to choose
        random_state : int, RandomState instance
            The generator used to initialize the centers. Use an int to make the
            randomness deterministic.
            See :term:`Glossary <random_state>`.
        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.
        Notes
        -----
        Selects initial cluster centers for k-mean clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007
        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        # Calculated w/o initialization speed up trick in sklearn
        x_squared_norms = row_norms(X, squared=True)
        n_samples, n_features = X.shape

        centers = np.empty((n_clusters, n_features), dtype=X.dtype)
        centersy = np.empty((n_clusters, n_features), dtype=X.dtype)

        assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        # Pick first center randomly
        center_id = np.random.randint(n_samples)
        if sp.issparse(X):
            centers[0] = X[center_id].toarray()
            centersy[0] = Y[center_id].toarray()

        else:
            centers[0] = X[center_id]
            centersy[0] = Y[center_id]


        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = euclidean_distances(
            centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
            squared=True)
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = np.random.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                            rand_vals)
            # XXX: numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                    out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

            # update closest distances squared and potential for each candidate
            np.minimum(closest_dist_sq, distance_to_candidates,
                       out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            if sp.issparse(X):
                centers[c] = X[best_candidate].toarray()
                centersy[c] = Y[best_candidate].toarray()

            else:
                centers[c] = X[best_candidate]
                centersy[c] = Y[best_candidate]


        return centers, centersy

    def train(self, candidate_num):
        start_time = time.time()                    # Note the time when training began
        theta = np.array([self.width_l, self.sigma])  # GP hyperparameters

        # Loop over all the candidate sparse points
        new_selected_x, new_selected_y = self.k_meansplusplus(self.points_x, self.points_y, n_clusters=self.inducing_num)


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
        return elapsed_time

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

    def update_matrices(self):
        self.matrix_k_uu = self.kernel(self.get_dis(self.inducing_points_x), self.width_l)  # Find K_uu
        self.inv_matrix_k_uu = np.linalg.inv(self.matrix_k_uu)
        self.matrix_k_fu = self.kernel(self.get_dis(self.points_x, self.inducing_points_x), self.width_l)  # Find K_fu
        self.matrix_k_uf = np.transpose(self.matrix_k_fu)  # Find K_MN
        self.matrix_c = 1 / (self.sigma ** 2) * np.dot(self.matrix_k_uf, self.matrix_k_fu) + self.matrix_k_uu
        self.inverse_c = np.linalg.inv(self.matrix_c)

    def predict(self, predict_x):
        if np.size(self.inducing_points_x[0]) == 1:
            inducing_distances = (self.inducing_points_x - predict_x) ** 2
        else:
            inducing_distances = np.sum((self.inducing_points_x - predict_x) ** 2, 1)

        matrix_k_us = self.kernel(inducing_distances, self.width_l)
        matrix_k_su = np.transpose(matrix_k_us)
        matrix_q_star_star = np.dot(np.dot(matrix_k_su, self.inv_matrix_k_uu), matrix_k_us)

        matrix_k_us = self.kernel(inducing_distances, self.width_l)

        predict_y_mean = 1 / (self.sigma ** 2) * np.dot(matrix_k_su, self.inverse_c)
        predict_y_mean = np.dot(np.dot(predict_y_mean, self.matrix_k_uf), self.points_y)
        predict_y_var = 1 + self.sigma ** 2 - matrix_q_star_star
        predict_y_var += np.dot(np.dot(matrix_k_su, self.inverse_c), matrix_k_us)
        predict_y_std = np.sqrt(predict_y_var)
        return predict_y_mean, predict_y_std

    def _get_neg_lower_bound(self, theta, a):
        """

        :param theta:
        :param a:
        :return:
        """

        raw_points_x, raw_points_y = self.points_x, self.points_y
        selected_points_x = a       # Extract arguments
        width_l, sigma = theta                                  # Extract hyperparameters
        matrix_k_uu = self.kernel(self.get_dis(selected_points_x), width_l)         # Find K_uu
        inverse_matrix_k_uu = np.linalg.inv(matrix_k_uu)                            # Find inverse of K_uu
        matrix_k_fu = self.kernel(self.get_dis(raw_points_x, selected_points_x), width_l)  # Find K_fu
        matrix_k_uf = np.transpose(matrix_k_fu)  # Find K_MN

        # We define A = K_fu * invK_uu * K_uf
        matrix_q_ff = np.dot(matrix_k_fu, np.dot(inverse_matrix_k_uu, matrix_k_uf))

        # B is an array containing only diagonal elements of K_uu - A.
        # Note we assume diagonal elements of A are always equal to 1.
        matrix_b = np.zeros(self.points_num)
        for i in range(self.points_num):
            matrix_b[i] = 1 - matrix_q_ff[i, i]

        # Calculate the (negative) lower bound
        matrix_c = matrix_q_ff + np.eye(self.points_num) * sigma ** 2
        sign, log_det_c = np.linalg.slogdet(matrix_c)
        log_det_c = sign * log_det_c
        nlb = -(-0.5 * log_det_c - 0.5 * np.dot(raw_points_y.T, np.dot(np.linalg.inv(matrix_c), raw_points_y))
                - 1 / (2 * sigma ** 2) * np.sum(matrix_b))
        return np.ravel(nlb)
