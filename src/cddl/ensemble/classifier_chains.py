from sklearn.linear_model import LogisticRegression
from skmultiflow.meta.classifier_chains import ClassifierChain


class ClassifierChain(ClassifierChain):

    """
    come from skmultiflow
    """

    def __init__(self, base_estimator=LogisticRegression(), order=None, random_state=None):
        super().__init__(base_estimator, order, random_state)
