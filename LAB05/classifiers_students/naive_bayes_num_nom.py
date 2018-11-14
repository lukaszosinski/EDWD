import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NaiveBayesNominal:
    def __init__(self):
        self.classes = {}
        self.amount_of_samples_for_class = {}
        self.classes_probs = {}
        self.model = {}
        self.y_prior = []

    def fit(self, input_data: np.array, y: np.array):
        self.y_prior = y
        self.classes = set(y)
        for class_ in self.classes:
            class_samples_amount = len([a for a in y if a == class_])
            self.amount_of_samples_for_class.update({class_: class_samples_amount})

        self.__compute_classes_prob()
        self.__compute_features_prob(input_data)

    def __compute_classes_prob(self):
        samples_amount = sum(self.amount_of_samples_for_class.values())
        for class_ in self.classes:
            class_prob = self.amount_of_samples_for_class.get(class_) / samples_amount
            self.classes_probs.update({class_: class_prob})

    def __compute_features_prob(self, input_data: np.array):
        number_of_columns = input_data.shape[1]
        for column in range(number_of_columns):
            feature_id = column + 1
            column = [row[column] for row in input_data]
            self.__process_column(column, feature_id)

    def __process_column(self, data_column, feature_id):
        number_of_data = len(data_column)
        possible_feature_values = set(data_column)

        for class_ in self.classes:
            for feature_value in possible_feature_values:
                ref_tuple = (feature_id, feature_value, class_)
                counter = 0
                for i in range(number_of_data):
                    query_tuple = (feature_id, data_column[i], self.y_prior[i])
                    if ref_tuple == query_tuple:
                        counter += 1

                tuple_prob = counter / self.amount_of_samples_for_class.get(class_)
                self.model.update({ref_tuple: tuple_prob})

    def __predict_proba(self, x: np.array):
        computed_probs = np.zeros(len(self.classes), float)
        for class_ in self.classes:
            prob = self.__compute_prob_for_class(class_, x)
            computed_probs[class_] = prob

        return computed_probs

    def __compute_prob_for_class(self, class_, x):
        nominator = self.__compute_partial_prob_for_class(class_, x)
        denominator = 0.0
        for c in self.classes:
            denominator += self.__compute_partial_prob_for_class(c, x)
        prob = nominator / denominator

        return prob

    def __compute_partial_prob_for_class(self, class_, x):
        feature_id = 1
        result = 1.0
        for feature_value in x:
            query_tuple = (feature_id, feature_value, class_)
            tuple_prob = self.model.get(query_tuple)
            result *= tuple_prob
            feature_id += 1
        result = result * self.classes_probs.get(class_)

        return result

    def predict(self, x):
        input_vectors_number = x.shape[0]
        output_vector = np.zeros(input_vectors_number, int)
        for i in range(input_vectors_number):
            classes_probabilities = self.__predict_proba(x[i])
            output_vector[i] = self.__get_class_with_max_prob(classes_probabilities)
        return output_vector

    @staticmethod
    def __get_class_with_max_prob(classes_probabilities):
        index_to_return = -1
        max_prob = 0
        for class_index in range(len(classes_probabilities)):
            if max_prob < classes_probabilities[class_index]:
                max_prob = classes_probabilities[class_index]
                index_to_return = class_index
        return index_to_return


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        self.is_cat = is_cat
        self.classes = {}
        self.amount_of_samples_for_class = {}
        self.classes_probs = {}
        self.nominal_model = {}
        self.features_means = {}
        self.features_standard_devs = {}
        self.y_prior = []

    def fit(self, input_data: np.array, y: np.array):
        self.y_prior = y
        self.classes = set(y)
        for class_ in self.classes:
            class_samples_amount = len([a for a in y if a == class_])
            self.amount_of_samples_for_class.update({class_: class_samples_amount})

        self.__compute_classes_prob()
        self.__compute_features_prob(input_data)

    def __compute_classes_prob(self):
        samples_amount = sum(self.amount_of_samples_for_class.values())
        for class_ in self.classes:
            class_prob = self.amount_of_samples_for_class.get(class_) / samples_amount
            self.classes_probs.update({class_: class_prob})

    def __compute_features_prob(self, input_data: np.array):
        number_of_columns = input_data.shape[1]
        for column in range(number_of_columns):
            feature_id = column + 1
            column = [row[column] for row in input_data]
            if not (self.is_cat[feature_id - 1]):
                self.__process_column_nominal(column, feature_id)
            else:
                self.__process_column_numerical(column, feature_id)

    def __process_column_nominal(self, data_column, feature_id):
        number_of_data = len(data_column)
        possible_feature_values = set(data_column)

        for class_ in self.classes:
            for feature_value in possible_feature_values:
                ref_tuple = (feature_id, feature_value, class_)
                counter = 0
                for i in range(number_of_data):
                    query_tuple = (feature_id, data_column[i], self.y_prior[i])
                    if ref_tuple == query_tuple:
                        counter += 1

                tuple_prob = counter / self.amount_of_samples_for_class.get(class_)
                self.nominal_model.update({ref_tuple: tuple_prob})

    def __process_column_numerical(self, data_column, feature_id):
        number_of_data = len(data_column)

        for class_ in self.classes:
            mean = 0
            standard_deviation = 0.0
            for i in range(number_of_data):
                if self.y_prior[i] == class_:
                    mean += data_column[i]
                    standard_deviation += data_column[i] ** 2

            mean /= self.amount_of_samples_for_class.get(class_)

            standard_deviation = (standard_deviation - self.amount_of_samples_for_class.get(class_)*mean ** 2) / \
                                 (self.amount_of_samples_for_class.get(class_) - 1)
            standard_deviation = math.sqrt(standard_deviation)
            tup = (feature_id, class_)
            self.features_means.update({tup: mean})
            self.features_standard_devs.update({tup: standard_deviation})

    def __predict_proba(self, x: np.array):
        computed_probs = np.zeros(len(self.classes), float)
        for class_ in self.classes:
            prob = self.__compute_prob_for_class(class_, x)
            computed_probs[class_] = prob

        return computed_probs

    def __compute_prob_for_class(self, class_, x):
        nominator = self.__compute_partial_prob_for_class(class_, x)
        denominator = 0.0
        for c in self.classes:
            denominator += self.__compute_partial_prob_for_class(c, x)
        prob = nominator / denominator

        return prob

    def __compute_partial_prob_for_class(self, class_, x):
        feature_id = 1
        result = 1.0
        for feature_value in x:
            if not (self.is_cat[feature_id - 1]):
                query_tuple = (feature_id, feature_value, class_)
                tuple_prob = self.nominal_model.get(query_tuple)
                result *= tuple_prob
            else:
                result *= self.calc_num_feature_prob(feature_id, feature_value, class_)
            feature_id += 1

        result = result * self.classes_probs.get(class_)

        return result

    def calc_num_feature_prob(self, feature_id, feature_value, class_):
        query_tuple = (feature_id, class_)
        return norm.pdf(feature_value, self.features_means.get(query_tuple),
                        self.features_standard_devs.get(query_tuple))

    def predict(self, x):
        input_vectors_number = x.shape[0]
        output_vector = np.zeros(input_vectors_number, int)
        for i in range(input_vectors_number):
            classes_probabilities = self.__predict_proba(x[i])
            output_vector[i] = self.__get_class_with_max_prob(classes_probabilities)
        return output_vector

    @staticmethod
    def __get_class_with_max_prob(classes_probabilities):
        index_to_return = -1
        max_prob = 0
        for class_index in range(len(classes_probabilities)):
            if max_prob < classes_probabilities[class_index]:
                max_prob = classes_probabilities[class_index]
                index_to_return = class_index
        return index_to_return


class NaiveBayesNumNom_ppp(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def predict_proba(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    # norm.pdf(x, average, std_dev)
