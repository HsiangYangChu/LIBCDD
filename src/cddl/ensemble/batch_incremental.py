from sklearn.tree import DecisionTreeClassifier
from skmultiflow.meta.batch_incremental import BatchIncremental


class BatchIncremental(BatchIncremental):

    """
    come from skmultiflow
    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), window_size=100, n_estimators=100):
        super().__init__(base_estimator, window_size, n_estimators)
