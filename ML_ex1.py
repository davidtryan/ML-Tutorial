"""
Created on Mon April 17 2017

@author: dtr5a
"""

#Reference:
#http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

##############################
## 0. Load required libraries
import os
import sys
import scipy
import numpy
import matplotlib
import sklearn
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


##############################
## 1. Load Data Sets (and format)
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pandas.read_csv(url, names=names)
from sklearn import datasets
dataset = datasets.load_iris()

# a) examine properties/gather basic information
dataset.keys()
dataset.target_names
dataset.target
dataset.DESCR
dataset.feature_names

# b) format data (Create a dataframe object to work with data)
dataset_class = pd.Series(dataset.target)
for i in range(len(dataset_class)):
    if dataset_class[i] == 0:
        dataset_class[i] = dataset.target_names[0]
    if dataset_class[i] == 1:
        dataset_class[i] = dataset.target_names[1]
    if dataset_class[i] == 2:
        dataset_class[i] = dataset.target_names[2]        

dataset = {dataset.feature_names[0]: dataset.data[:,0],
           dataset.feature_names[1]: dataset.data[:,1],
           dataset.feature_names[2]: dataset.data[:,2],
           dataset.feature_names[3]: dataset.data[:,3],
           'class': dataset_class}
#dataset = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
dataset = pd.DataFrame(dataset)


##############################
## 2. Explore/Summarize Data (gather basic information)

#shape
print(dataset.shape)

#snapshot
print(dataset.head(20))

#Statistical summary
print(dataset.describe())

#number of unique labels
print(len(numpy.unique(dataset['class'])))
    
#Class distribution
print(dataset.groupby('class').size())


##############################
## 3. Data Visualization (with matplotlib)

#box and whiskers plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

#multivariate plots
scatter_matrix(dataset)
plt.show()


##############################
## 3b. Data Preparation

#normalization
from sklearn.preprocessing import scale
dnorm = scale(dataset.iloc[:,1:5].values)
dataset.iloc[:1,5] = dnorm


##############################
## 4. Evaluate Algorithms

# a) Split out validation data set
array = dataset.values
X = array[:,1:5]
Y = array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# b) Test Harness (Cross Validation)
seed = 7
scoring = 'accuracy'

# c) Build models
#Algorithms to check
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print msg

# d) Compare models and select the best
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


##############################
## 5. Make Predications
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

cart = DecisionTreeClassifier()
cart.fit(X_train, Y_train)
predictions = cart.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

svm = SVC(gamma=0.001, C=100., kernel='linear')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Import GridSearchCV
from sklearn.grid_search import GridSearchCV
# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, Y_train)

# Print out the results 
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)

# Apply the classifier to the test data, and view the accuracy score
clf.score(X_validation, Y_validation)  

# Train and score a new classifier with the grid search parameters
svm2 = SVC(C=1., kernel='linear', gamma='auto')
svm2.fit(X_train, Y_train).score(X_validation, Y_validation)
print(accuracy_score(Y_validation, svm.predict(X_validation)))










