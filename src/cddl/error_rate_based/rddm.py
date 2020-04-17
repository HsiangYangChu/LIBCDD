from math import *

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class RDDM(BaseDriftDetector):

    def __init__(self, min_num_instances=129, warning_level=1.773, drift_level=2.258, max_size_concept=40000, min_size_stable_concept=7000, warn_limit=1400):
        super().__init__()
        self.min_num_instances = min_num_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.max_size_concept = max_size_concept
        self.min_size_stable_concept = min_size_stable_concept
        self.warn_limit = warn_limit
        self.stored_predictions = [0 for i in range(int(self.min_size_stable_concept))]
        self.num_stored_instances = 0
        self.first_pos = 0
        self.last_pos = -1
        self.last_warn_pos = -1
        self.last_warn_inst = -1
        self.inst_num = 0
        self.rddm_drift = False
        self.in_concept_change = False
        self.miss_num = 1
        self.miss_prob = 1
        self.miss_std = 0

        self.reset()

        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.miss_prob_sd_min = float("inf")

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        # super().reset()
        self.miss_num = 1
        self.miss_prob = 1
        self.miss_std = 0
        if self.in_concept_change:
            self.miss_prob_min = float("inf")
            self.miss_sd_min = float("inf")
            self.miss_prob_sd_min = float("inf")

    def add_element(self, prediction):

        # if self.in_concept_change:
        #     self.reset()

        if self.rddm_drift:
            self.reset()
            if self.last_warn_pos != -1:
                self.first_pos = self.last_warn_pos
                self.num_stored_instances = self.last_pos - self.first_pos + 1
                if self.num_stored_instances <= 0:
                    self.num_stored_instances += self.min_size_stable_concept

            pos = self.first_pos
            for i in range(self.num_stored_instances):
                self.miss_prob = self.miss_prob + (self.stored_predictions[pos]-self.miss_prob) / self.miss_num
                self.miss_std = sqrt(self.miss_prob * (1 - self.miss_prob) / self.miss_num)
                if self.in_concept_change and self.miss_num > self.min_num_instances and self.miss_prob + self.miss_std < self.miss_prob_sd_min:
                    self.miss_prob_min = self.miss_prob
                    self.miss_sd_min = self.miss_std
                    self.miss_prob_sd_min = self.miss_prob + self.miss_std
                self.miss_num += 1
                pos = (pos + 1) % self.min_size_stable_concept

            self.last_warn_pos = -1
            self.last_warn_inst = -1
            self.rddm_drift = False
            self.in_concept_change = False

        self.last_pos = (self.last_pos + 1) % self.min_size_stable_concept
        self.stored_predictions[self.last_pos] = prediction
        if self.num_stored_instances < self.min_size_stable_concept:
            self.num_stored_instances += 1
        else:
            self.first_pos = (self.first_pos + 1) % self.min_size_stable_concept
            if self.last_warn_pos == self.last_pos:
                self.last_warn_pos = -1

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / self.miss_num
        self.miss_std = sqrt(self.miss_prob * (1 - self.miss_prob) / self.miss_num)

        self.inst_num += 1
        self.miss_num += 1
        self.estimation = self.miss_prob
        self.in_warning_zone = False

        if self.miss_num <= self.min_num_instances:
            return

        if self.miss_prob + self.miss_std < self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.drift_level * self.miss_sd_min:
            self.in_concept_change = True
            self.rddm_drift = True
            if self.last_warn_inst == -1:
                self.first_pos = self.last_pos    # DDM Drift without previous warning
                self.min_num_instances = 1
            return

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            # Warning level for warn_limit consecutive instances will force drifts
            if self.last_warn_inst != -1 and self.last_warn_inst + self.warn_limit <= self.inst_num:
                self.in_concept_change = True
                self.rddm_drift = True
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
                self.last_warn_pos = -1
                self.last_warn_inst = -1
                return
            # Warning Zone
            self.in_warning_zone = True
            if self.last_warn_inst == -1:
                self.last_warn_inst = self.inst_num
                self.last_warn_pos = self.last_pos
        else:
            self.last_warn_inst = -1
            self.last_warn_pos = -1

        if self.miss_num > self.max_size_concept and not self.in_warning_zone:
            self.rddm_drift = True