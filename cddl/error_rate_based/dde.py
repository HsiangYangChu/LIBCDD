from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

from cddl.error_rate_based import *

class DDE(BaseDriftDetector):

    def __init__(self, outlier=100, detectors="HDDM_A,HDDM_W,DDM", min_drift_weight=1):
        super().__init__()
        self.outlier = None
        self.result = []
        self.warning_level = None
        self.drift_level = None
        self.inst_number = None
        self.index = None
        self.min_drift_weight = None
        self.change_detector_pool = []
        # self.ddmstring = []
        self.value_list = detectors
        self.ensemble()
        self.result = [0 for i in range(len(self.change_detector_pool))]
        self.min_drift_weight = min_drift_weight
        self.outlier = outlier
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.ensemble()
        self.inst_number = 0
        for i in range(len(self.result)):
            self.result[i] = 0

    def add_element(self, prediction):

        if self.in_concept_change:
            self.reset()
        self.inst_number += 1
        self.drift_level = 0
        self.warning_level = 0

        for i in range(len(self.change_detector_pool)):
            if self.result[i] < 1:  # not in drift
                self.change_detector_pool[i].add_element(prediction)

                if self.change_detector_pool[i].in_concept_change:
                    self.result[i] = self.inst_number
                    self.drift_level += 1
                else:
                    if self.change_detector_pool[i].in_warning_zone:
                        self.warning_level += 1
            else:  # in drift
                if self.result[i] + self.outlier < self.inst_number:
                    self.result[i] = 0
                else:
                    self.drift_level += 1

            if (self.drift_level >= self.min_drift_weight):
                break
        if self.warning_level + self.drift_level < self.min_drift_weight:
            self.in_warning_zone = False
        else:
            if self.drift_level < self.min_drift_weight:
                self.in_warning_zone = True
            else:
                # self.reset()
                self.in_concept_change = True

    def ensemble(self):
        if self.value_list != "":
            self.split = self.value_list.split(",")
            self.change_detector_pool = []
            # self.ddmstring = []
            if len(self.split) > 0:
                for i in range(len(self.split)):
                    # if self.split[i].index("(") > -1:
                    #     self.split[i] = self.split[i].substring(self.split[i].index("(")+1)
                    #     self.split[i] = self.split[i].substring(0, self.split[i].index(")"))

                    self.change_detector_pool.append(globals()[self.split[i]]())
                    # self.ddmstring.append(self.split[i])
            else:
                self.change_detector_pool.append(globals()[self.value_list]())
                # self.ddmstring.append(self.value_list)
