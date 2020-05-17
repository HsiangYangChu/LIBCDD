from skmultiflow.lazy import KNNAdwin
from skmultiflow.meta.online_csb2 import OnlineCSB2


class OnlineCSB2(OnlineCSB2):

    """
    come form skmultiflow

    """

    def __init__(self,
                 base_estimator=KNNAdwin(),
                 n_estimators=10,
                 cost_positive=1,
                 cost_negative=0.1,
                 drift_detection=True,
                 random_state=None):
        super().__init__(base_estimator,
                         n_estimators,
                         cost_positive,
                         cost_negative,
                         drift_detection,
                         random_state)