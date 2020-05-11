from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta.additive_expert_ensemble import AdditiveExpertEnsemble


class AdditiveExpertEnsemble(AdditiveExpertEnsemble):
    """
    come from skmultiflow
    """

    def __init__(self, n_estimators=5, base_estimator=NaiveBayes(), beta=0.8,
                 gamma=0.1, pruning='weakest'):
        """
        Creates a new instance of AdditiveExpertEnsemble.
        """
        super().__init__(n_estimators, base_estimator, beta,
                         gamma, pruning)
