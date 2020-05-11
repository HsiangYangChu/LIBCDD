from sklearn.linear_model import SGDRegressor

from skmultiflow.meta.regressor_chains import RegressorChain

class RegressorChain(RegressorChain):

    """
    come from skmultiflow

    """
    def __init__(self, base_estimator=SGDRegressor(), order=None, random_state=None):
        super().__init__(base_estimator, order, random_state)
