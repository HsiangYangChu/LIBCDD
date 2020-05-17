import numpy as math
import math

from scipy.stats import norm
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class STEPD(BaseDriftDetector):

    def __init__(self, window_size=30, alpha_warning=0.05, alpha_dirft=0.003):
        super().__init__()
        self.window_size = window_size
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_dirft

        self.stored_pred = [0 for i in range(int(self.window_size))]
        self.first_pos = None
        self.last_pos = None

        self.ro = None
        self.rr = None
        self.wo = None
        self.wr = None
        self.no = None
        self.nr = None
        self.p = None
        self.Z = None
        self.size_inve_sum = None

        self.reset()

    def reset(self):
        super().reset()
        self.first_pos = 0
        self.last_pos = -1
        self.wo = 0.0
        self.wr = 0.0
        self.no = 0
        self.nr = 0

    def add_element(self, prediction):
        if self.in_concept_change:
            self.reset()

        if self.nr == self.window_size:                     #Recent window is full.
            self.wo += self.stored_pred[self.first_pos]     #Oldest prediction in recent window
            self.no += 1                                    #is moved to older window
            self.wr -= self.stored_pred[self.first_pos]
            self.first_pos += 1
            if self.first_pos == self.window_size:
                self.first_pos = 0
        else:   #Recent window grows
            self.nr += 1

        self.last_pos += 1  #Add prediction at the end of recent window
        if self.last_pos == self.window_size:
            self.last_pos = 0
        self.stored_pred[self.last_pos] = prediction
        self.wr += prediction

        self.in_warning_zone = False
        self.in_concept_change = False
        self.delay = 0

        if self.no >= self.window_size:     #The same as: (no + nr) >= 2 * window_size
            self.ro = self.no - self.wo     #Number of correct predictions are calculated
            self.rr = self.nr - self.wr
            self.size_inve_sum = 1.0/self.no + 1.0/self.nr
            self.p = (self.ro+self.rr)/(self.no+self.nr)
            self.Z = abs(self.ro/self.no-self.rr/self.nr)
            self.Z = self.Z - self.size_inve_sum * 0.5
            self.Z = self.Z/( math.sqrt(self.p*(1.0-self.p)*self.size_inve_sum)+1e-18 )
            self.Z = norm.ppf(abs(self.Z))
            self.Z = 2 * (1 - self.Z);

            if self.Z < self.alpha_drift:
                self.in_concept_change = True
            elif self.Z < self.alpha_warning:
                self.in_warning_zone = True
            else:
                self.in_warning_zone = False
