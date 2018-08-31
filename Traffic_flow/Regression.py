# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:27:20 2018

@author: HSH
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

traffic=pd.read_csv('traffic_flow_data.csv')
print(traffic.head(5))
columns=traffic.columns
# select the features column
features=['Segment_45(t)']
#target=features[-1]
target=['Segment23_(t+1)']

x=traffic[features].values
y=traffic[target].values
# Split the dataset into training dataset and testing dataset
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
#=======================Ordinary Regression============

lr = linear_model.LinearRegression().fit(x_train,y_train)
y_predict=lr.predict(x_test)
print(lr.score(x_test, y_test))
# print model
p1=plt.scatter(range(len(x_test)), y_test,  color='black')
# Prediction and draw the diagram
p2=plt.plot(range(len(x_test)), y_predict, color='red', linewidth=1)
plt.legend(["predict", "true"], loc='upper right')
plt.title('Ordinary Regression')
plt.show()
print('_________________###################____________________')
print('Explained variance regression score function: %.2f' % explained_variance_score(y_test, y_predict))
print('Mean absolute error regression loss: %.2f' % mean_absolute_error(y_test, y_predict))
print("Mean squared error regression loss: %.2f" % mean_squared_error(y_test, y_predict))
print('Mean squared logarithmic error regression loss: %.2f' % mean_squared_log_error(y_test, y_predict))
print('Median absolute error regression loss: %.2f' % median_absolute_error(y_test, y_predict))
print('R^2 (coefficient of determination) regression score function.: %.2f' % r2_score(y_test, y_predict))
print('******************************************************* ')


#========================== Ridge Regression=================

from sklearn.linear_model import Ridge
clf = Ridge(alpha=.5)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
# print model
p1=plt.scatter(range(len(x_test)), y_test,  color='black')
# Prediction and draw the diagram
p2=plt.plot(range(len(x_test)), y_predict, color='red', linewidth=1)
plt.legend(["predict", "true"], loc='upper right')
plt.title('Ridge Regression')
plt.show()
print('_________________###################____________________')
print('Explained variance regression score function: %.2f' % explained_variance_score(y_test, y_predict))
print('Mean absolute error regression loss: %.2f' % mean_absolute_error(y_test, y_predict))
print("Mean squared error regression loss: %.2f" % mean_squared_error(y_test, y_predict))
print('Mean squared logarithmic error regression loss: %.2f' % mean_squared_log_error(y_test, y_predict))
print('Median absolute error regression loss: %.2f' % median_absolute_error(y_test, y_predict))
print('R^2 (coefficient of determination) regression score function.: %.2f' % r2_score(y_test, y_predict))
print('******************************************************* ')
