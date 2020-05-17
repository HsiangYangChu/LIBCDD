import numpy as np
import math
import time

from scipy.stats import norm
from .base_distribution_detector import BaseDistributionDetector


class LSDDCDT(BaseDistributionDetector):

    def __init__(self, train_size=400, window_size=200, u_s=0.02, u_w=0.01, u_c=0.001, bootstrap_num=2000):
        super().__init__()
        if window_size * 2 > train_size:
            raise ValueError("window_size * 2 > train_size.")
        self.n = window_size
        self.m = bootstrap_num
        self.n_t = train_size
        self.u_s = u_s
        self.u_w = u_w
        self.u_c = u_c

        self.p_w = None
        self.p_c = None
        self.t_s = None
        self.t_w = None
        self.t_c = None
        self.sigma = None
        self.lambd = None
        self.window_reference = [None for _ in range(self.n)]
        self.window_slide = [None for _ in range(self.n)]
        self.window_train = [None for _ in range(self.n_t)]
        self.win_ref_i = None
        self.win_sli_i = None
        self.win_tra_i = None
        self.i = None
        self.warning_num = None

        self.reset()

    def reset(self):
        super().reset()
        self.t_s = None
        self.t_w = None
        self.t_c = None
        self.sigma = None
        self.lambd = None

        self.win_ref_i = 0
        self.win_sli_i = 0
        self.win_tra_i = 0
        self.i = 0
        self.warning_num = 0

    def add_element(self, input_value):

        if self.in_concept_change:
            self.reset()

        input_value = np.asarray(input_value)

        if input_value.ndim != 1:
            raise ValueError("X should has one dimension")

        # return

        if self.win_tra_i < self.n_t:
            self.window_train[self.win_tra_i] = input_value
            self.win_tra_i += 1
            if self.win_tra_i == self.n_t:
                self.training()
            return

        if self.win_ref_i < self.n:
            self.window_reference[self.win_ref_i] = input_value
            self.win_ref_i += 1
            return

        self.i += 1

        if self.i < self.n:
            self.window_slide[self.win_sli_i] = input_value
            self.win_sli_i = (self.win_sli_i + 1) % self.n
            return

        # slide window_slide
        self.window_slide[self.win_sli_i] = input_value
        self.win_sli_i = (self.win_sli_i + 1) % self.n

        # calculate d^2
        d = self.get_d(np.asarray(self.window_reference), np.asarray(self.window_slide))

        print(str(self.i) + ": " + str(d))

        if d > self.t_w or self.in_warning_zone:
            self.in_warning_zone = True
            self.warning_num += 1
            if d > self.t_c:
                self.in_concept_change = True

            if d < self.t_s or self.warning_num >= self.n:
                self.in_warning_zone = False
                self.warning_num = 0
                self.reservoir_sampling(input_value)
        else:
            self.reservoir_sampling(input_value)
            self.warning_num = 0

    def training(self):
        self.get_sigma()
        self.get_lambda()
        if self.lambd is None:
            self.lambd = 1.0
        self.bootstrapping()
        print("Ts: " + str(self.t_s))
        print("Tw: " + str(self.t_w))
        print("Tc: " + str(self.t_c))
        return

    def reservoir_sampling(self, input_value):
        r = np.random.randint(0, self.n + self.i + 1)
        if r < self.n - 1:
            self.window_reference[r] = input_value
        return

    def get_d(self, X1, X2):
        H, h = self.get_H_and_h(X1, X2)
        r = H.shape[0]
        theta = np.linalg.inv(H + np.eye(r) * self.lambd).dot(h)
        d = theta.T.dot(h) * 2 - theta.T.dot(H).dot(theta)
        return d[0][0]

    def get_H_and_h(self, X1, X2):
        r1, c1 = X1.shape
        r2, c2 = X2.shape
        if c1 != c2:
            raise ValueError("c1 != c2.")
        r = r1 + r2
        X = np.append(X1, X2, axis=0)
        H, h = [], []
        for i in range(r):
            get_H_i_j_vec = np.vectorize(self.get_H_i_j, signature='(n),(n)->()')
            H.append(get_H_i_j_vec(X, X[i]))
        H = np.asarray(H)

        for i in range(r):
            # get_distance_vec = np.vectorize(self.get_distance, signature='(n),(n)->()')
            # h.append(np.mean(get_distance_vec(X1, X[i])) - np.mean(get_distance_vec(X2, X[i])))
            get_h_i_vec = np.vectorize(self.get_h_i, signature='(n),(n)->()')
            h.append(np.mean(get_h_i_vec(X1, X[i])) - np.mean(get_h_i_vec(X2, X[i])))
        h = np.asarray(h).reshape((r, 1))
        return H, h

    def get_H_i_j(self, ci, cj):
        c = len(ci)                         #这个地方可能会有些问题
        tmp = math.pow(math.pi * pow(self.sigma, 2), c * 0.5) * \
              math.exp(-self.get_distance(ci, cj) / 4 / pow(self.sigma, 2))
        return tmp

    def get_h_i(self, ci, cj):
        return math.exp(-self.get_distance(ci, cj) / 2 / pow(self.sigma, 2))

    def get_distance(self, instance_one, instance_two):
        one = np.array(instance_one).flatten()
        two = np.array(instance_two).flatten()
        return np.sqrt(np.sum(np.power(np.subtract(one, two), [2 for _ in range(one.size)])))

    def get_sigma(self):
        sum = 0
        for xi in self.window_train:
            for xj in self.window_train:
                sum += self.get_distance(xi, xj)
        self.sigma = sum / pow(self.n_t, 2)

    def bootstrapping(self):
        array = np.array(self.window_train)
        sample_result_arr = []
        for i in range(self.m):
            index_arr = np.random.randint(0, self.n_t, size=self.n)
            data_sample1 = array[index_arr]
            index_arr = np.random.randint(0, self.n_t, size=self.n)
            data_sample2 = array[index_arr]
            sample_result = self.get_d(data_sample1, data_sample2)
            sample_result_arr.append(sample_result)

        k_s = int(self.m * (1 - self.u_s))
        k_w = int(self.m * (1 - self.u_w))
        k_c = int(self.m * (1 - self.u_c))

        auc_sample_arr_sorted = sorted(sample_result_arr)
        self.t_s = auc_sample_arr_sorted[k_s]
        self.t_w = auc_sample_arr_sorted[k_w]
        self.t_c = auc_sample_arr_sorted[k_c]

    def get_lambda(self):
        num, RD0 = 20, 0.25
        _lambdas = np.flipud(np.logspace(-2, 1, 20))
        for _lambda in _lambdas:
            array = np.array(self.window_train)
            ave_RD = 0
            for i in range(num):
                index_arr = np.random.randint(0, self.n_t, size=self.n)
                data_sample1 = array[index_arr]
                index_arr = np.random.randint(0, self.n_t, size=self.n)
                data_sample2 = array[index_arr]
                ave_RD += self.get_RD(data_sample1, data_sample2, _lambda)
            ave_RD /= num
            if ave_RD < RD0:
                self.lambd = _lambda
                return
        return

    def get_RD(self, X1, X2, _lambda):
        H, h = self.get_H_and_h(X1, X2)
        r = H.shape[0]
        aux = np.linalg.inv(H + np.eye(r) * _lambda)
        RD = h.T.dot(aux.dot(aux)).dot(h)[0][0] / (h.T.dot(aux).dot(h)[0][0]+1e-10)
        return RD * _lambda



