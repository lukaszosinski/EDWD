import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NaiveBayesNominal:
    def __init__(self):
        self.classes = None
        self.classes_prob = {}
        self.model = dict()
        self.y_prior = []

    def fit(self, x: np.array, y: np.array):
        self.classes = set(y)
        self.compute_classes_prob(y)
        self.compute_conditional_prob(x, y)

        print(self.classes_prob)

    def compute_classes_prob(self, y: np.array):
        samples_amount = y.size
        for c in self.classes:
            class_prob = len([a for a in y if a == c]) / samples_amount
            self.classes_prob.update({c: class_prob})

    def compute_conditional_prob(self, x: np.array, y: np.array):
        for column in x:
            print(x[column])

    def predict_proba(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError




class NaiveBayesGaussian:
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
