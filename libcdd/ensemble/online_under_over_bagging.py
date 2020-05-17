from skmultiflow.lazy import KNNAdwin
from skmultiflow.meta.online_under_over_bagging import OnlineUnderOverBagging


class OnlineUnderOverBagging(OnlineUnderOverBagging):

    """
    come form skmultiflow

    """

    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, sampling_rate=2, drift_detection=True,
                 random_state=None):
        super().__init__(base_estimator, n_estimators, sampling_rate, drift_detection,
                 random_state)
