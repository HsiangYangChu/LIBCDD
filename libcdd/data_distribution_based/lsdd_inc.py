import numpy as np
import math


from .base_distribution_detector import BaseDistributionDetector


class LSDDINC(BaseDistributionDetector):

    def __init__(self, train_size=400, window1_size=200, window2_size=200, u=0.01, bootstrap_num=2000):
        super().__init__()
        self.n = window1_size
        self.m = window2_size
        self.bootstrap_num = bootstrap_num
        self.n_t = train_size
        self.u = u

        self.sigma = None
        self.lambd = None
        self.window_slide = [None for _ in range(self.m)]
        self.window_train = [None for _ in range(self.n_t)]
        self.centers = None
        self.win_sli_i = None
        self.win_tra_i = None
        self.i = None
        self.t1 = None
        self.Hl = None
        self.h = None

        self.reset()

    def reset(self):
        super().reset()
        self.sigma = None
        self.lambd = None
        self.t1 = None
        self.Hl = None
        self.h = None
        self.centers = None

        self.win_sli_i = 0
        self.win_tra_i = 0
        self.i = 0

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

        if self.i < self.m:
            self.window_slide[self.win_sli_i] = input_value
            self.win_sli_i = (self.win_sli_i + 1) % self.m
            self.i += 1
            if self.i == self.m:
                self.get_Hl_and_h()
                self.centers = np.append(np.asarray(self.window_train), np.asarray(self.window_slide), axis=0)
            return

        # slide window_slide
        self.h -= self.get_dia(x_new=input_value, x_old=self.window_slide[self.win_sli_i])
        d = self.h.T.dot(self.Hl).dot(self.h)
        if d > self.t1:
            self.in_concept_change = True

        self.window_slide[self.win_sli_i] = input_value
        self.win_sli_i = (self.win_sli_i + 1) % self.n

    def get_dia(self, x_new, x_old):
        get_h_i_vec = np.vectorize(self.get_h_i, signature='(n),(n)->()')
        dia = get_h_i_vec(self.centers, x_new)-get_h_i_vec(self.centers, x_old)
        dia = 1.0 / self.m * dia
        return dia.reshape((self.centers.shape[0], 1))

    def training(self):
        self.get_sigma()
        self.get_lambda
        if self.lambd is None:
            self.lambd = 1.0
        self.bootstrapping()
        return

    def get_Hl_and_h(self):
        H, self.h = self.get_H_and_h(np.asarray(self.window_train), np.asarray(self.window_slide))
        r = H.shape[0]
        aux = np.linalg.inv(H + np.eye(r) * self.lambd)
        self.Hl = aux*2 - aux.T.dot(H).dot(aux)

    def get_sigma(self):
        sum = 0
        for xi in self.window_train:
            for xj in self.window_train:
                sum += self.get_distance(xi, xj)
        self.sigma = sum / pow(self.n_t, 2)

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
            get_h_i_vec = np.vectorize(self.get_h_i, signature='(n),(n)->()')
            h.append(np.mean(get_h_i_vec(X1, X[i])) - np.mean(get_h_i_vec(X2, X[i])))
        h = np.asarray(h).reshape((r, 1))
        return H, h

    def get_H_i_j(self, ci, cj):
        c = len(ci)
        tmp = math.pow(math.pi * pow(self.sigma, 2), c * 0.5) * \
              math.exp(-self.get_distance(ci, cj) / 4 / pow(self.sigma, 2))
        return tmp

    def get_h_i(self, ci, cj):
        return math.exp(-self.get_distance(ci, cj) / 2 / pow(self.sigma, 2))

    def get_distance(self, instance_one, instance_two):
        one = np.array(instance_one).flatten()
        two = np.array(instance_two).flatten()
        return np.sqrt(np.sum(np.power(np.subtract(one, two), [2 for _ in range(one.size)])))

    def get_lambda(self):
        num, RD0 = 20, 0.2
        _lambdas = np.flipud(np.logspace(-2, 1, 20))
        for _lambda in _lambdas:
            array = np.array(self.window_train)
            ave_RD = 0
            for i in range(num):
                index_arr = np.random.randint(0, self.n_t, size=self.n)
                data_sample1 = array[index_arr]
                index_arr = np.random.randint(0, self.n_t, size=self.m)
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

    def bootstrappig(self):
        array = np.array(self.window_train)
        sample_result_arr = []
        for i in range(self.bootstrap_num):
            index_arr = np.random.randint(0, self.n_t, size=self.n)
            data_sample1 = array[index_arr]
            index_arr = np.random.randint(0, self.n_t, size=self.m)
            data_sample2 = array[index_arr]
            sample_result = self.get_d(data_sample1, data_sample2)
            sample_result_arr.append(sample_result)

        i = int(self.bootstrap_num * (1 - self.u))
        auc_sample_arr_sorted = sorted(sample_result_arr)
        t = auc_sample_arr_sorted[i]
        eh0 = np.mean(np.asarray(sample_result_arr))
        k = (1.0/self.n_t+1.0/self.m)/(1.0/self.n+1.0/self.m)-1
        self.t1 = k * eh0 + t

    def get_d(self, X1, X2):
        H, h = self.get_H_and_h(X1, X2)
        r = H.shape[0]
        theta = np.linalg.inv(H + np.eye(r) * self.lambd).dot(h)
        d = theta.T.dot(h) * 2 - theta.T.dot(H).dot(theta)
        return d[0][0]