from skmultiflow.lazy import KNNAdwin
from skmultiflow.meta.online_rus_boost import OnlineRUSBoost

class OnlineRUSBoost(OnlineRUSBoost):

    """
    come from skmultiflow

    """

    def __init__(self,
                 base_estimator=KNNAdwin(),
                 n_estimators=10,
                 sampling_rate=3,
                 algorithm=1,
                 drift_detection=True,
                 random_state=None):

        super().__init__(base_estimator,
                 n_estimators,
                 sampling_rate,
                 algorithm,
                 drift_detection,
                 random_state)
