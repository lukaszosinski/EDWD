import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from LAB04.classifiers_students.naive_bayes import NaiveBayesNominal
from LAB04.input_mapper import map_input


trainSet = pd.read_csv("sklearn-nb/data/grypa1-train.csv")
testSet = pd.read_csv("sklearn-nb/data/grypa1-test.csv")

trainSer = map_input(trainSet)
testSet = map_input(testSet)

trainSet_x = trainSet.loc[:, trainSet.columns != 'grypa'].values
trainSet_y = trainSet.grypa.values
testSet = testSet.values

nb = NaiveBayesNominal()
nb.fit(trainSet_x, trainSet_y)