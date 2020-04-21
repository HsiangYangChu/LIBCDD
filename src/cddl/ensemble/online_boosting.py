from skmultiflow.lazy import KNNAdwin
from skmultiflow.meta.online_boosting import OnlineBoosting


class OnlineBoosting(OnlineBoosting):

    """
    come form skmultiflow

    """

    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, drift_detection=True, random_state=None):
        super().__init__(base_estimator, n_estimators, drift_detection, random_state)
