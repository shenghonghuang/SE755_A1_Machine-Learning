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
#features=columns[0:-1]#除去最后一列，前面的做特征
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
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(y_test, y_predict))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(y_test, y_predict))
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
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(y_test, y_predict))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(y_test, y_predict))
print('******************************************************* ')
