from sklearn.linear_model import SGDClassifier

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin


class MultiOutputLearner(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin):
    """
    come form skmultiflow

    """

    def __init__(self, base_estimator=SGDClassifier(max_iter=100)):
        super().__init__(base_estimator)
