import copy as cp
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.utils.utils import *


class PairedLearners(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, stable_estimator=NaiveBayes(), reactive_estimator=NaiveBayes(), window_size=12, threshold=0.2):
        super().__init__()
        # default values

        self.c = None
        self.stable_base_estimator = stable_estimator
        self.reactive_base_estimator = reactive_estimator
        self.stable_estimator = None
        self.reactive_estimator = None
        self.t = None
        self.classes = None
        self.w = window_size
        self.theta = math.floor(self.w * threshold)
        self.instances_X = None
        self.instances_y = None
        self.change_detected = None
        self.number_of_errors = None
        self.__configure()

    def __configure(self):

        self.classes = None
        self.change_detected = 0
        self.number_of_errors = 0
        self.t = 0
        self.c = [0 for i in range(self.w)]
        self.instances_X = [None for _ in range(self.w)]
        self.instances_y = [None for _ in range(self.w)]
        self.stable_estimator = cp.deepcopy(self.stable_base_estimator)
        self.reactive_estimator = cp.deepcopy(self.reactive_base_estimator)

    def reset(self):
        self.__configure()
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):

        if classes is None and self.classes is None:
            raise ValueError("The first partial_fit call should pass all the classes.")
        if classes is not None and self.classes is None:
            self.classes = classes
        elif classes is not None and self.classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError(
                    "The classes passed to the partial_fit function differ from those passed in an earlier moment.")
        r, c = get_dimensions(X)
        for i in range(r):
            self.__partial_fit(np.asarray([X[i]]), np.asarray([y[i]]))

        return self

    def __partial_fit(self, X, y):

        self.instances_X[self.t] = X
        self.instances_y[self.t] = y

        self.stable_prediction = self.stable_estimator.predict(X)[0] == y[0]
        self.reactive_prediction = self.reactive_estimator.predict(X)[0] == y[0]

        self.number_of_errors -= self.c[self.t]
        if not self.stable_prediction and self.reactive_prediction:
            self.c[self.t] = 1
            self.number_of_errors += 1
        else:
            self.c[self.t] = 0

        if self.theta < self.number_of_errors:
            self.change_detected += 1
            self.stable_estimator = cp.deepcopy(self.reactive_estimator)
            for i in range(self.w):
                self.c[i] = 0
            self.number_of_errors = 0

        self.stable_estimator.partial_fit(X, y)
        self.reactive_estimator = cp.deepcopy(self.reactive_base_estimator)
        for i in range(self.w):
            if self.instances_X[i] is None:
                break
            self.reactive_estimator.partial_fit(self.instances_X[i], self.instances_y[i])

        self.t += 1
        if self.t == self.w:
            self.t = 0

    def predict(self, X):

        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        if proba is None:
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.asarray(predictions)

    def predict_proba(self, X):

        proba = []
        r, c = get_dimensions(X)

        try:
            partial_proba = self.reactive_estimator.predict_proba(X)
            if len(partial_proba[0]) > max(self.classes) + 1:
                raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

            if len(proba) < 1:
                for n in range(r):
                    proba.append([0.0 for _ in partial_proba[n]])

            for n in range(r):
                for l in range(len(partial_proba[n])):
                    try:
                        proba[n][l] += partial_proba[n][l]
                    except IndexError:
                        proba[n].append(partial_proba[n][l])

        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        # normalizing probabilities
        sum_proba = []
        for l in range(r):
            sum_proba.append(np.sum(proba[l]))
        aux = []
        for i in range(len(proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in proba[i]])
            else:
                aux.append(proba[i])
        return np.asarray(aux)
