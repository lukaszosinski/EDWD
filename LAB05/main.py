import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score

from LAB05.classifiers_students.naive_bayes_num_nom import NaiveBayesNumNom
from LAB05.input_mapper import map_input

def test_model(m, X, y=None, print_data=True):
    ypred = m.predict(X)
    if print_data:
        if y is not None:
            print("ground truth:")
            print(y)
        print("predicted:")
        print(ypred)
    if y is not None:
        print("accuracy:")
        print(accuracy_score(y, ypred))

training_set = pd.read_csv("sklearn-nb/data/grypa1-train.csv")
test_set = pd.read_csv("sklearn-nb/data/grypa1-test.csv")
training_set = map_input(training_set)
test_set = map_input(test_set)
training_set_x = training_set.loc[:, training_set.columns != 'grypa'].values
training_set_y = training_set.grypa.values
test_set = test_set.values
nbnm = NaiveBayesNumNom([False, False, False, False])
nbnm.fit(training_set_x, training_set_y)
print(nbnm.predict(test_set))
print("************")
print(training_set_y)
print(nbnm.predict(training_set_x))

nbnm_2 = NaiveBayesNumNom([True, True, True, True])
# Load some data provided with the sklearn package http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

# Divide the dataset into training and test part 80:20 respectively
frac_train = 0.8
n_samples = X_iris.shape[0]
n_train = int(float(n_samples*frac_train))
X_iris_train = X_iris[:n_train, :]
y_iris_train = y_iris[:n_train]
X_iris_test = X_iris[n_train:, :]
y_iris_test = y_iris[n_train:]

nbnm_2.fit(X_iris_train, y_iris_train)

test_model(nbnm_2, X_iris_test, y_iris_test, False)
test_model(nbnm_2, X_iris_train, y_iris_train, False)
