# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:50:29 2018

@author: HSH
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('max_colwidth',200)

# Preprocessing dataset
worldcup=pd.read_csv('2018 worldcup.csv',index_col=0)
# Drop irrelevant column
worldcup.drop(['Date'],axis=1,inplace=True)
worldcup.describe()

# Convert words to number

# Calculate the number of locations
Location=worldcup['Location'].unique()
print('Location={0}'.format(len(Location)))
#
Phase=worldcup['Phase'].unique()
print('Phase={0}'.format(len(Phase)))

# Drop irrelevant columns for features
worldcup=worldcup.drop(['Location','Normal_Time'],axis=1)
print(worldcup.shape)

# Drop irrelevant columns for features
worldcup=worldcup.drop(['Team1','Team1_Continent','Team2','Team2_Continent'],axis=1)

# Drop irrelevant columns for features
worldcup=worldcup.drop(['Match_result','Team1_Pass_Accuracy(%)','Team2_Pass_Accuracy(%)','Team1_Ball_Possession(%)','Team2_Ball_Possession(%)',],axis=1)

print(worldcup.columns)
#利用pandas进行one-hot编码
worldcup=pd.get_dummies(worldcup,prefix='Phase')
print(worldcup.shape)

# Selection useful columns for regression
featrues=['Team1_Attempts', 'Team1_Corners', 'Team1_Distance_Covered',
          'Team2_Attempts', 'Team2_Corners', 'Team2_Distance_Covered']
target=['Total_Scores']
print(worldcup.head(5))
x=worldcup[featrues].values
y=worldcup[target].values

# Split the dataset into training dataset and testing dataset
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)

# Use SK-Learn generate models
#=======================Ordinary Regression============

lr = linear_model.LinearRegression().fit(x_train,y_train)
y_predict=lr.predict(x_test)
print(lr.score(x_test, y_test))
# Print the model
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
# Print the model
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