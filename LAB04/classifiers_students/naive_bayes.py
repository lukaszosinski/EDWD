import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NaiveBayesNominal:
    def __init__(self):
        self.classes = {}
        self.classes_prob = {}
        self.model = {}
        self.y_prior = []

    def fit(self, input_data: np.array, y: np.array):
        self.y_prior = y
        classes_ = set(y)
        for c in classes_:
            class_samples_amount = len([a for a in y if a == c])
            self.classes.update({c: class_samples_amount})

        self.compute_classes_prob()
        self.compute_features_prob(input_data)

        print(self.model)

    def compute_classes_prob(self):
        samples_amount = sum(self.classes.values())
        for c in self.classes:
            class_prob = self.classes.get(c) / samples_amount
            self.classes_prob.update({c: class_prob})

    def compute_features_prob(self, input_data: np.array):
        number_of_columns = input_data.shape[1]
        for column in range(number_of_columns):
            feature_id = column + 1
            self.process_column([row[column] for row in input_data], feature_id)

    def process_column(self, data_column, feature_id):
        number_of_data = len(data_column)
        possible_options = set(data_column)

        for _class in self.classes:
            for option in possible_options:
                ref_tuple = (feature_id, option, _class)
                counter = 0
                for i in range(number_of_data):
                    query_tuple = (feature_id, data_column[i], self.y_prior[i])
                    if ref_tuple == query_tuple:
                        counter += 1

                tuple_prob = counter / self.classes.get(_class)
                self.model.update({ref_tuple: tuple_prob})

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
