import pandas as pd

from LAB04.classifiers_students.naive_bayes import NaiveBayesNominal
from LAB04.input_mapper import map_input


training_set = pd.read_csv("sklearn-nb/data/grypa1-train.csv")
test_set = pd.read_csv("sklearn-nb/data/grypa1-test.csv")
training_set = map_input(training_set)
test_set = map_input(test_set)
training_set_x = training_set.loc[:, training_set.columns != 'grypa'].values
training_set_y = training_set.grypa.values
test_set = test_set.values
print(training_set_y)
nb = NaiveBayesNominal()
nb.fit(training_set_x, training_set_y)
print(nb.predict(training_set_x))