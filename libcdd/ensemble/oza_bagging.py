from skmultiflow.lazy import KNNAdwin

from skmultiflow.meta.oza_bagging import OzaBagging


class OzaBagging(OzaBagging):

    """
    come from skmultiflow

    """
    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, random_state=None):
        super().__init__(base_estimator, n_estimators, random_state)

