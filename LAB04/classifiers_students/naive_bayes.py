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
        for class_ in classes_:
            class_samples_amount = len([a for a in y if a == class_])
            self.classes.update({class_: class_samples_amount})

        self.compute_classes_prob()
        self.compute_features_prob(input_data)

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

    def predict_proba(self, x: np.array):
        number_of_classes = len(self.classes.keys())
        computed_probs = np.append(number_of_classes, 0.0)
        for class_index in range(number_of_classes):
            prob = self.compute_prob_for_class(class_index, x)
            computed_probs[class_index] = prob

        return computed_probs

    def compute_prob_for_class(self, class_index, x):
        denom = 0.0
        input_features_vector = x
        all_classes = set(self.classes.keys())

        nom = self.method_name(class_index, input_features_vector)
        for class_index in all_classes:
            denom += self.method_name(class_index, input_features_vector)

        prob = nom / denom
        return prob

    def method_name(self, class_index, x):
        feature_id = 1
        result = 1.0
        for feature_value in x:
            query_tuple = (feature_id, feature_value, class_index)
            tuple_prob = self.model.get(query_tuple)
            result *= tuple_prob
            feature_id += 1
        result = result * self.classes_prob.get(class_index)

        return result

    def predict(self, x):
        input_vectors_number = x.shape[0]
        output_vector = np.zeros(input_vectors_number, int)
        for i in range(input_vectors_number):
            probs = self.predict_proba(x[i])
            output_vector[i] = self.get_class_index_with_max_prob(probs)
        return output_vector

    @staticmethod
    def get_class_index_with_max_prob(probs):
        index_to_return = -1
        max_prob = 0
        for index in range(len(probs)):
            if max_prob < probs[index]:
                max_prob = probs[index]
                index_to_return = index
        return index_to_return


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
