from skmultiflow.lazy import KNNAdwin

from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin


class OzaBaggingAdwin(OzaBaggingAdwin):

    """
    come form skmultiflow

    """

    def __init__(self, base_estimator=KNNAdwin(), n_estimators=10, random_state=None):
        super().__init__(base_estimator, n_estimators, random_state)

