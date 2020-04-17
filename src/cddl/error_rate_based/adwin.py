from skmultiflow.drift_detection.adwin import ADWIN


class ADWIN(ADWIN):

    """
    come from skmultiflow

    """
    def __init__(self, delta=.002):
        super().__init__(delta)

