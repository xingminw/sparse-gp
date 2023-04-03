import numpy as np
from scipy import spatial as spt


class GaussianProcesses(object):
    def __init__(self):
        # gp params
        self.width_l = None
        self.sigma = None

        self.points_x = None
        self.points_y = None
        self.points_num = None
        self.matrix_k = None
        self.matrix_c = None
        self.inverse_c = None

    def load_data(self, points_x, points_y, points_num=None):
        self.points_x = np.vstack(points_x)
        self.points_y = np.vstack(points_y)
        if points_num is None:
            self.points_num = len(self.points_x)
        else:
            self.points_num = points_num

    def init_trainer(self, *args):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError

    def predict(self, predict_x):
        """

        :param predict_x: single given x
        :return:
        """
        if np.size(self.points_x[0]) == 1:
            squared_distances = (self.points_x - predict_x) ** 2
        else:
            squared_distances = np.sum((self.points_x - predict_x) ** 2, 1)
        k = self.kernel(squared_distances, self.width_l)
        c = 1 + self.sigma ** 2  # Always true for this particular kernel
        predict_y_mean = np.dot(k.T, np.dot(self.inverse_c, self.points_y))
        predict_y_std = np.sqrt(c - np.dot(k.T, np.dot(self.inverse_c, k)))
        return predict_y_mean, predict_y_std

    def update_matrices(self):
        squared_dis = self.get_dis(self.points_x)
        self.matrix_k = self.kernel(squared_dis, self.width_l)                      # Gram matrix
        self.matrix_c = self.matrix_k + self.sigma ** 2 * np.eye(self.points_num)   # C matrix
        self.inverse_c = np.linalg.inv(self.matrix_c)                               # Find inverse of C

    @staticmethod
    def get_dis(points_x, points_x_prime=None):
        """
        get the distance matrix

        :param points_x:
        :param points_x_prime:
        :return:
        """
        # points_x = np.vstack(points_x)
        if points_x_prime is not None:
            points_x_prime = np.vstack(points_x_prime)
            return spt.distance.cdist(points_x, points_x_prime, metric='sqeuclidean')
        else:
            return spt.distance.cdist(points_x, points_x, metric='sqeuclidean')

    @staticmethod
    def kernel(squared_distances, width_l):
        """
        Kernel function for the GPs

        :param squared_distances:
        :param width_l:
        :return:
        """
        return np.exp(-1 / (2 * width_l ** 2) * squared_distances)

