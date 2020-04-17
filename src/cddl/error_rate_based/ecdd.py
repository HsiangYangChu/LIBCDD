import numpy as math
import math

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class ECDD(BaseDriftDetector):


    def __init__(self, min_num_instances=30, warning_level=0.5, _lambda=0.2):
        super().__init__()
        self.sample_count = None
        self.miss_prob = None
        self.miss_std = None
        self.miss_sum = None
        self.z_t = None
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self._lambda = _lambda
        self.reset()

    def reset(self):
        super().reset()
        self.sample_count = 1.0
        self.miss_prob = 0.0
        self.miss_std = 0.0
        self.miss_sum = 0.0
        self.z_t = 0.0

    def add_element(self, prediction):
        if self.in_concept_change:
            self.reset()

        self.miss_sum += prediction
        self.miss_prob = self.miss_sum/self.sample_count
        self.miss_std = math.sqrt( self.miss_prob * (1.0 - self.miss_prob)* self._lambda * (1.0 - math.pow(1.0 - self._lambda, 2.0 * self.sample_count)) / (2.0 - self._lambda))
        self.sample_count += 1
        self.z_t += self._lambda * (prediction - self.z_t)

        L_t = 3.97 - 6.56 * self.miss_prob + 48.73 * math.pow(self.miss_prob, 3) - 330.13 * math.pow(self.miss_prob, 5) + 848.18 * math.pow(self.miss_prob, 7)

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.z_t > self.miss_prob + L_t * self.miss_std:
            self.in_concept_change = True

        elif self.z_t > self.miss_prob + self.warning_level * L_t * self.miss_std:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False