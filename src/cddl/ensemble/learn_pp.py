from sklearn.tree import DecisionTreeClassifier
from skmultiflow.meta.learn_pp import LearnPP


class LearnPP(LearnPP):
    """
    come from skmultiflow

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(),
                 error_threshold=0.5,
                 n_estimators=30,
                 n_ensembles=10,
                 window_size=100,
                 random_state=None):
        super().__init__(base_estimator,
                         error_threshold,
                         n_estimators,
                         n_ensembles,
                         window_size,
                         random_state)
