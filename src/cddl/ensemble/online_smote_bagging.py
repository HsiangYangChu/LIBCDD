from skmultiflow.lazy import KNNAdwin
from skmultiflow.meta.online_smote_bagging import OnlineSMOTEBagging


class OnlineSMOTEBagging(OnlineSMOTEBagging):

    """
    come from skmultiflow
    """

    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, sampling_rate=1, drift_detection=True,
                 random_state=None):
        super().__init__(base_estimator, n_estimators, sampling_rate, drift_detection,
                 random_state)
