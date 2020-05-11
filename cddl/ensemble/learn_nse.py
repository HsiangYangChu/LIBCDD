from sklearn.tree import DecisionTreeClassifier
from skmultiflow.meta.learn_nse import LearnNSE


class LearnNSE(LearnNSE):
    """
    come from skmultiflow

    """

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 window_size=250,
                 slope=0.5,
                 crossing_point=10,
                 n_estimators=15,
                 pruning=None):
        super().__init__(base_estimator,
                         window_size,
                         slope,
                         crossing_point,
                         n_estimators,
                         pruning)
