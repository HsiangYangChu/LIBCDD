from skmultiflow.meta.classifier_chains import *


class ClassifierChain(ClassifierChain):

    def __init__(self, base_estimator=LogisticRegression(), order=None, random_state=None):
        super().__init__(base_estimator, order, random_state)

class ProbabilisticClassifierChain(ProbabilisticClassifierChain):

    def __init__(self, base_estimator=LogisticRegression(), order=None, random_state=None):
        super().__init__(base_estimator, order, random_state)

class MonteCarloClassifierChain(MonteCarloClassifierChain):

    def __init__(self, base_estimator=LogisticRegression(), M=10, random_state=None):
        super().__init__(base_estimator, M, random_state)
