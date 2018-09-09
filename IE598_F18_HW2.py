#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:30:43 2018

@author: yingzhaocheng
"""
import sklearn
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

X = iris.data[:,[2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y,test_size=0.3, random_state=0,stratify=y)
print( X_train.shape, y_train.shape)

# Standardize the features
#'scaler = preprocessing.StandardScaler().fit(X_train)
#'X_train = scaler.transform(X_train)
#'X_test = scaler.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state = 33)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from mlxtend.plotting import plot_decision_regions
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined,y_combined,clf = tree )
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')
knn.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std,X_test_std))
plot_decision_regions(X_combined_std, y_combined, clf=knn)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()


from sklearn.metrics import accuracy_score
k_range = range(1,26)
scores = []
for k in k_range:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train,y_train)
    y_pred = knn_test.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
    
scores
plt.plot(k_range, scores)

print("My name is zhaocheng Ying")
print("My NetID is: zying4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    
    
    


















