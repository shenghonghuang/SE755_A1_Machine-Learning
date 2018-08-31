# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:27:20 2018

@author: HSH
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, matthews_corrcoef, zero_one_loss

data=pd.read_csv('occupancy_sensor_data.csv')
print(data.head(5))
columns=data.columns
print(columns)


features=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
target=['Occupancy']
x=data[features].values
y=data[target].values
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)

#===================Perceptron=========================
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
y_pred = ppn.fit(x_train, y_train).predict(x_test)
print('Perceptron:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))
#===================SVM=========================
from sklearn import svm
#调用SVC()
clf = svm.SVC()
y_pred = clf.fit(x_train, y_train).predict(x_test)
print('SVM:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))
#===================Decision Trees=========================
from sklearn import tree
clf = tree.DecisionTreeClassifier()
y_pred = clf.fit(x_train, y_train).predict(x_test)
print('Decision trees:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))
#=================== Nearest neighbour classifier=========================
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
y_pred = clf.fit(x_train, y_train).predict(x_test)
print('Nearest neighbour classifier:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))
#===================  Naïve Bayes classifier=========================
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
y_pred = clf.fit(x_train, y_train).predict(x_test)
print('Naive Bayes classifier:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))

