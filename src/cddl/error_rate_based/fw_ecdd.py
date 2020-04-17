import numpy as math
import math

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class FWECDD(BaseDriftDetector):

    def __init__(self, min_num_instances=30, min_fw_size = 15, warning_level=2.0, _lambda=0.2, fw_rate=0.8):
        super().__init__()
        self.sample_count = None
        self.miss_sum = None
        self.miss_prob = None
        self.fw_miss_num = None
        self.fw_miss_prob = None
        # self.f_m_s = None
        # self.f_z_t = None
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.min_fw_size = min_fw_size
        self.fw_rate = fw_rate
        self.pre_hist = []
        self._lambda = _lambda
        self.z_t = None
        self.reset()

    def reset(self):
        super().reset()
        self.sample_count = 1
        self.miss_sum = 0
        self.miss_prob = 0
        # self.m_s = 0
        self.z_t = 0

        self.pre_hist.clear()

    def add_element(self, prediction):
        self.pre_hist.append(prediction)

        if self.in_concept_change:
            self.reset()

        self.miss_sum += prediction
        self.miss_prob = self.miss_sum / self.sample_count
        # self.m_s = math.sqrt(self.miss_prob * (1.0 - self.miss_prob) * self._lambda * (1.0 - math.pow(1.0 - self._lambda, 2.0 * self.sample_count)) / (2.0 - self._lambda))
        self.sample_count += 1

        self.fw_miss_prob = 0
        self.fw_miss_num = 1
        tmp = len(self.pre_hist)*self.fw_rate
        for i in range(len(self.pre_hist)):
            if tmp >= self.min_fw_size and i <= tmp:
                self.fw_miss_prob += self.pre_hist[i]*i/tmp
                self.fw_miss_num += i / tmp
                # self.f_z_t += self._lambda*(self.pre_hist[i]*i/tmp-self.f_z_t)
            else:
                self.fw_miss_prob += self.pre_hist[i]
                self.fw_miss_num += 1
                # self.f_z_t += self._lambda*(self.pre_hist[i]-self.f_z_t)

        self.fw_miss_prob /= self.fw_miss_num
        self.f_m_s = math.sqrt(self.fw_miss_prob*(1-self.fw_miss_prob)/self.fw_miss_num)

        self.z_t += self._lambda * (prediction - self.z_t)


        L_t = 3.97 - 6.56 * self.fw_miss_prob + 48.73 * math.pow(self.fw_miss_prob, 3) - 330.13 * math.pow(self.fw_miss_prob, 5) + 848.18 * math.pow(self.fw_miss_prob, 7)

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.z_t > self.fw_miss_prob + L_t * self.f_m_s:
            self.in_concept_change = True

        elif self.z_t > self.fw_miss_prob + self.warning_level * L_t * self.f_m_s:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False