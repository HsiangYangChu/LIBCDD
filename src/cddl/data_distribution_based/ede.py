import numpy as np

from src.cddl.data_distribution_based.base_distribution_detector import BaseDistributionDetector


class EDE(BaseDistributionDetector):

    def __init__(self, window_size=100, alpha=0.05, sample_size=500):
        super().__init__()
        self.window_size = window_size
        self.window_ref = [None for _ in range(self.window_size)]
        self.window_sli = [None for _ in range(self.window_size)]
        self.sample_size = sample_size
        self.alpha = alpha

        self.win_ref_i = None
        self.win_sli_i = None
        self.i = None
        self.t = None
        self.w = None
        self.reset()

    def sampling(self):
        return

    def reset(self):
        super().reset()
        self.win_ref_i = 0
        self.win_sli_i = 0
        self.i = 0
        self.w = -1

    def add_element(self, input_value):

        if self.in_concept_change:
            self.reset()

        input_value = np.asarray(input_value)

        if input_value.ndim != 1:
            raise ValueError("X should has one dimension")

        if self.win_ref_i < self.window_size:
            self.window_ref[self.win_ref_i] = input_value
            self.win_ref_i = self.win_ref_i + 1
            return

        self.window_sli[self.win_sli_i] = input_value
        self.win_sli_i = (self.win_sli_i + 1) % self.window_size
        self.i += 1

        if self.i < self.window_size:
            return

        self.w = self.get_w(self.window_ref, self.window_sli)

        if self.w > self.t:
            self.in_concept_change = True

    def get_w(self, X1, X2):

        return 0

