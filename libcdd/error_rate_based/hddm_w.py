from math import *

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class HDDM_W(BaseDriftDetector):

    class SampleInfo:
        def __init__(self):
            self.EWMA_estimator = -1.0
            self.independent_bounded_condition_sum = None

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, lambda_option=0.050, two_side_option=True):
        super().__init__()
        super().reset()
        self.total = self.SampleInfo()
        self.sample1_decr_monitor = self.SampleInfo()
        self.sample1_incr_monitor = self.SampleInfo()
        self.sample2_decr_monitor = self.SampleInfo()
        self.sample2_incr_monitor = self.SampleInfo()
        self.incr_cutpoint = float("inf")
        self.decr_cutpoint = float("inf")
        self.width = 0
        self.delay = 0
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_option = lambda_option
        self.two_side_option = two_side_option

    def add_element(self, prediction):
        """ Add a new element to the statistics
        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).
        Notes
        -----
        After calling self method, to verify if change was detected or if
        the learner is in the warning zone, one should call the super method
        detected_change, which returns True if concept drift was detected and
        False otherwise.
        """
        aux_decay_rate = 1.0 - self.lambda_option
        self.width += 1
        if self.total.EWMA_estimator < 0:
            self.total.EWMA_estimator = prediction
            self.total.independent_bounded_condition_sum = 1
        else:
            self.total.EWMA_estimator = self.lambda_option * prediction + aux_decay_rate * self.total.EWMA_estimator
            self.total.independent_bounded_condition_sum = \
                self.lambda_option * self.lambda_option \
                + aux_decay_rate * aux_decay_rate * self.total.independent_bounded_condition_sum

        self._update_incr_statistics(prediction, self.drift_confidence)
        if self._monitor_mean_incr(self.drift_confidence):
            self.reset()
            self.in_concept_change = True
            self.in_warning_zone = False
            return
        elif self._monitor_mean_incr(self.warning_confidence):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

        self._update_decr_statistics(prediction, self.drift_confidence)
        if self.two_side_option and self._monitor_mean_decr(self.drift_confidence):
            self.reset()
        self.estimation = self.total.EWMA_estimator

    def _detect_mean_increment(self, sample1, sample2, confidence):
        if sample1.EWMA_estimator < 0 or sample2.EWMA_estimator < 0:
            return False

        bound = sqrt((sample1.independent_bounded_condition_sum
                      + sample2.independent_bounded_condition_sum) * log(1 / confidence) / 2)
        return sample2.EWMA_estimator - sample1.EWMA_estimator > bound

    def _monitor_mean_incr(self, confidence):
        return self._detect_mean_increment(self.sample1_incr_monitor, self.sample2_incr_monitor, confidence)

    def _monitor_mean_decr(self, confidence):
        return self._detect_mean_increment(self.sample2_decr_monitor, self.sample1_decr_monitor, confidence)

    def _update_incr_statistics(self, value, confidence):
        aux_decay = 1.0 - self.lambda_option
        epsilon = sqrt(self.total.independent_bounded_condition_sum * log(1.0 / confidence) / 2)

        if self.total.EWMA_estimator + epsilon < self.incr_cutpoint:
            self.incr_cutpoint = self.total.EWMA_estimator + epsilon
            self.sample1_incr_monitor.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_incr_monitor.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_incr_monitor = self.SampleInfo()
            self.delay = 0
        else:
            self.delay += 1
            if self.sample2_incr_monitor.EWMA_estimator < 0:
                self.sample2_incr_monitor.EWMA_estimator = value
                self.sample2_incr_monitor.independent_bounded_condition_sum = 1
            else:
                self.sample2_incr_monitor.EWMA_estimator = \
                    self.lambda_option * value + aux_decay * self.sample2_incr_monitor.EWMA_estimator
                self.sample2_incr_monitor.independent_bounded_condition_sum = \
                    self.lambda_option * self.lambda_option + \
                    aux_decay * aux_decay * self.sample2_incr_monitor.independent_bounded_condition_sum

    def _update_decr_statistics(self, value, confidence):
        aux_decay = 1.0 - self.lambda_option
        epsilon = sqrt(self.total.independent_bounded_condition_sum * log(1.0 / confidence) / 2)

        if self.total.EWMA_estimator - epsilon > self.decr_cutpoint:
            self.decr_cutpoint = self.total.EWMA_estimator - epsilon
            self.sample1_decr_monitor.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_decr_monitor.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_decr_monitor = self.SampleInfo()
        else:
            if self.sample2_decr_monitor.EWMA_estimator < 0:
                self.sample2_decr_monitor.EWMA_estimator = value
                self.sample2_decr_monitor.independent_bounded_condition_sum = 1
            else:
                self.sample2_decr_monitor.EWMA_estimator = \
                    self.lambda_option * value + aux_decay * self.sample2_decr_monitor.EWMA_estimator
                self.sample2_decr_monitor.independent_bounded_condition_sum = \
                    self.lambda_option * self.lambda_option \
                    + aux_decay * aux_decay * self.sample2_decr_monitor.independent_bounded_condition_sum

    def reset(self):
        """ reset
        Resets the change detector parameters.
        """
        super().reset()
        self.total = self.SampleInfo()
        self.sample1_decr_monitor = self.SampleInfo()
        self.sample1_incr_monitor = self.SampleInfo()
        self.sample2_decr_monitor = self.SampleInfo()
        self.sample2_incr_monitor = self.SampleInfo()
        self.incr_cutpoint = float("inf")
        self.decr_cutpoint = float("inf")
        self.width = 0
        self.delay = 0