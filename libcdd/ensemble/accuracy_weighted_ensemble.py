from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.accuracy_weighted_ensemble import AccuracyWeightedEnsemble


class AccuracyWeightedEnsemble(AccuracyWeightedEnsemble):
    """
    come from skmultiflow
    """

    def __init__(self, n_estimators=10, n_kept_estimators=30,
                 base_estimator=NaiveBayes(), window_size=200, n_splits=5):
        """ Create a new ensemble"""

        super().__init__(n_estimators, n_kept_estimators,
                 base_estimator, window_size, n_splits)
