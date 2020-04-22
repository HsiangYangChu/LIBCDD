import numpy as np
import math
import random

from .base_distribution_detector import BaseDistributionDetector


class RD(BaseDistributionDetector):

    def __init__(self, window_size=200, n=5000, p=0.99, type="rd1", sample_size=800):
        super().__init__()
        self.window_size = window_size
        self.window_ref = [None for _ in range(self.window_size)]
        self.window_sli = [None for _ in range(self.window_size)]
        self.n = n
        self.p = p
        if type != "rd1" and type != "rd2":
            raise ValueError("RD should be rd1 or rd2")
        self.type = type
        self.sample_size = sample_size

        self.win_ref_i = None
        self.win_sli_i = None
        self.i = None
        self.alpha = None
        self.rd = None
        self.sample()
        self.reset()

    def sample(self):
        samples = []
        size = int(self.n / 2)
        for i in range(self.sample_size):
            X1 = np.random.uniform(0, 1, size)
            X2 = np.random.uniform(0, 1, size)
            samples.append(self.get_rd(X1.tolist(), X2.tolist()))
        samples = sorted(samples)
        self.alpha = samples[int(self.p * self.sample_size)]
        return

    def reset(self):
        super().reset()
        self.win_ref_i = 0
        self.win_sli_i = 0
        self.i = 0
        self.rd = -1

    def add_element(self, input_value):

        if self.in_concept_change:
            self.reset()

        if self.win_ref_i < self.window_size:
            self.window_ref[self.win_ref_i] = input_value
            self.win_ref_i = self.win_ref_i + 1
            return

        self.window_sli[self.win_sli_i] = input_value
        self.win_sli_i = (self.win_sli_i + 1) % self.window_size
        self.i += 1

        if self.i < self.window_size:
            return

        self.rd = self.get_rd(self.window_ref, self.window_sli)
        # print(self.alpha)
        print(self.rd)

        if self.rd > self.alpha:
            self.in_concept_change = True

    def get_rd(self, X1, X2):
        X = X1 + X2
        for i in range(len(X)):
            if i < len(X1):
                X[i] = [X[i], 1 / len(X1)]
            else:
                X[i] = [X[i], -1 / len(X2)]
        X = sorted(X, key=(lambda x: x[0]))
        pre = [0 for _ in range(len(X) + 1)]
        for i in range(1, len(X) + 1):
            pre[i] = pre[i - 1] + X[i-1][1]
        rd = 0
        for l in range(1, len(X) + 1):
            for r in range(l, len(X) + 1):
                num = abs(pre[r] - pre[l - 1])
                aux = (r - l + 1) / len(X)
                dem = math.sqrt(min(aux, 1 - aux))
                if self.type == "rd2":
                    dem = math.sqrt(aux * (1 - aux))
                if dem == 0:
                    continue
                rd = max(rd, num / dem)
        return rd
