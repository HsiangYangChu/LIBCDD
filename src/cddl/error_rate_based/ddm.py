from skmultiflow.drift_detection.ddm import DDM

class DDM(DDM):
    """
    Come from skmultiflow

    """

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0):
        super().__init__(min_num_instances, warning_level, out_control_level)