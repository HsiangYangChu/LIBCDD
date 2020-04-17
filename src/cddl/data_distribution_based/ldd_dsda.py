import numpy as np
import math
import scipy.stats as stats

from skmultiflow.distribution_detection.base_distribution_detector import BaseDistributionDetector
from skmultiflow.bayes import NaiveBayes
from skmultiflow.utils.utils import *
from sklearn.neighbors import KDTree


class LDDDSDA(BaseDistributionDetector):

    def __init__(self, batch_size=100, train_size=100, rho=0.1, alpha=0.05, base_learner=NaiveBayes()):
        super().__init__()
        self.w = batch_size
        self.l = base_learner
        self.n = train_size
        self.alpha = alpha
        self.rho = rho
        self.trained = False

        self.d_train_X, self.d_train_y = [], []
        self.d_buffer_X, self.d_buffer_y = [], []
        self.reset()

    def reset(self):
        super().reset()

    def add_element(self, X, y):

        if self.in_concept_change:
            self.reset()

        X, y = np.asarray(X), np.asarray(y)

        # if X.ndim != 1 or y.ndim != 1:
        #     raise ValueError("input_value should has one dimension")

        if (not self.trained) and len(self.d_train_X) < self.n:
            self.d_train_X.append(X)
            self.d_train_y.append(y)
            if len(self.d_train_X) == self.n:
                self.l.partial_fit(np.asarray(self.d_train_X), np.asarray(self.d_train_y))
                self.trained = True
            return

        if len(self.d_train_X) < self.w:
            self.d_train_X.append(X)
            self.d_train_y.append(y)
            return

        self.d_buffer_X.append(X)
        self.d_buffer_y.append(y)

        if len(self.d_buffer_X) < self.w:
            return

        self.d_train_X, self.d_train_y = self.ldd_dis(np.asarray(self.d_train_X),
                                                      np.asarray(self.d_train_y),
                                                      np.asarray(self.d_buffer_X),
                                                      np.asarray(self.d_buffer_y))
        self.l = NaiveBayes()
        self.l.fit(self.d_train_X, self.d_train_y)

        self.d_train_X = self.d_train_X.tolist()
        self.d_train_y = self.d_train_y.tolist()
        print(len(self.d_train_X))
        self.d_buffer_X = []
        self.d_buffer_y = []

        return

    def predict(self, X):
        return self.l.predict(X)

    def ldd_dis(self, d1_X, d1_y, d2_X, d2_y):
        d = np.append(d1_X, d2_X, axis=0)
        d_y = np.append(d1_y, d2_y, axis=0)
        d1_dec, d1_sta, d1_inc = [], [], []
        d2_dec, d2_sta, d2_inc = [], [], []

        kdtree = KDTree(d)
        d_knn = []
        for i in range(d.shape[0]):
            d_knn.append(set(kdtree.query(X=d[i:i+1],
                                          k=int(d.shape[0] * self.rho),
                                          return_distance=False)[0]))

        indexes = np.arange(d.shape[0])
        np.random.shuffle(indexes)
        _d1 = set(indexes[:d1_X.shape[0]])
        _d2 = set(indexes[d1_X.shape[0]:])
        deltas = []
        for i in range(d.shape[0]):
            x1 = len(d_knn[indexes[i]] & _d1)
            x2 = len(d_knn[indexes[i]] & _d2)
            if i < d1_X.shape[0]:
                deltas.append(x2 / x1 - 1)
            else:
                deltas.append(x1 / x2 - 1)

        delta_std = np.std(deltas, ddof=1)
        theta_dec = stats.norm.ppf(1 - self.alpha, 0, delta_std)
        theta_inc = stats.norm.ppf(self.alpha, 0, delta_std)

        _d1 = set(np.arange(d1_X.shape[0]))
        _d2 = set(np.arange(d1_X.shape[0], d.shape[0]))
        for i in range(d.shape[0]):
            x1 = len(d_knn[i] & _d1)
            x2 = len(d_knn[i] & _d2)
            if i < d1_X.shape[0]:
                delta = x2 / x1 - 1
                if delta < theta_dec:
                    d1_dec.append(i)
                elif delta > theta_inc:
                    d1_inc.append(i)
                else:
                    d1_sta.append(i)
            else:
                delta = x1 / x2 - 1
                if delta < theta_dec:
                    d2_dec.append(i)
                elif delta > theta_inc:
                    d2_inc.append(i)
                else:
                    d2_sta.append(i)

        if len(d1_dec) == 0 and len(d2_inc) == 0:
            return d1_X, d1_y

        self.in_concept_change = True

        aux = []
        if len(d2_dec) != 0:
            aux.append(len(d1_inc) / len(d2_dec))
        if len(d2_sta) != 0:
            aux.append(len(d1_sta) / len(d2_sta))
        if len(d2_inc) != 0:
            aux.append(len(d1_dec) / len(d2_inc))
        k = min(aux)

        d2_dec += d1_inc[:int(k * len(d2_dec))]
        d2_sta += d1_sta[:int(k * len(d2_sta))]
        d2_inc += d1_dec[:int(k * len(d2_inc))]

        aux_indexes = d2_inc + d2_sta + d2_dec

        r = self.w / len(aux_indexes)

        d2_dec = d2_dec[:int(len(d2_dec)*r)]
        d2_sta = d2_sta[:int(len(d2_sta)*r)]
        d2_inc = d1_inc[:int(len(d2_inc)*r)]

        aux_indexes = d2_inc + d2_sta + d2_dec

        return d[aux_indexes], d_y[aux_indexes]
