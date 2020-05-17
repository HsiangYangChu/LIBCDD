from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.dynamic_weighted_majority import DynamicWeightedMajority


class DynamicWeightedMajority(DynamicWeightedMajority):

    """
    come from skmultiflow
    """

    def __init__(self, n_estimators=5, base_estimator=NaiveBayes(),
                 period=50, beta=0.5, theta=0.01):
        """
        Creates a new instance of DynamicWeightedMajority.
        """
        super().__init__(n_estimators, base_estimator,
                 period, beta, theta)