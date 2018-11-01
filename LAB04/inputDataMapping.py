import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def mapper(x):
    if x == 'duzy':
        return 2
    if x == 'sredni':
        return 1
    if x == 'tak':
        return 1
    if x == 'nie':
        return 0
    else:
        return x


trainSet = pd.read_csv("sklearn-nb/data/grypa1-train.csv")
testSet = pd.read_csv("sklearn-nb/data/grypa1-test.csv")

for column in trainSet:
    trainSet[column] = trainSet[column].map(mapper)

for column in testSet:
    testSet[column] = testSet[column].map(mapper)

trainSet_x = trainSet.loc[:, trainSet.columns != 'grypa'].values
trainSet_y = trainSet.grypa.values
testSet = testSet.values

dt = DecisionTreeClassifier()
classifier = dt.fit(trainSet_x, trainSet_y)

pr_test = classifier.predict(testSet)
pr_train = classifier.predict(trainSet_x)
print(pr_test)
print(pr_train)

ac = accuracy_score(trainSet_y, pr_train)
print(ac)
