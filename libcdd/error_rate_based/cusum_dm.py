import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class CusumDM(BaseDriftDetector):

    def __init__(self, min_num_instances=30, delta=0.005, _lambda=50):
        super().__init__()
        self.sample_count = None
        self.miss_prob = None
        self.miss_sum = None
        self.min_instances = min_num_instances
        self.delta = delta
        self._lambda = _lambda
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 0.0
        self.miss_sum = 0.0

    def add_element(self, prediction):
        """ Add a new element to the statistics

        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).

        Notes
        -----
        After calling this method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.

        """
        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_sum = max(0, self.miss_sum + prediction - self.miss_prob - self.delta)
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.miss_sum > self._lambda:
            self.in_concept_change = True
