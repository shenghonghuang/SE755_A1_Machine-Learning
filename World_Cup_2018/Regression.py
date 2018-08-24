# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:50:29 2018

@author: HSH
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion
##################
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1 首先获取整个表的内容，去掉一列不需要的
worldcup=pd.read_csv('2018 worldcup.csv',index_col=0)
#match date is assumed to be irrelevant for the match results
worldcup.drop(['Date','Team1_Ball_Possession(%)'],axis=1,inplace=True)
worldcup.describe()

#2 提取Target和Features
# 2.1 提取前26列用作Features
w_features=worldcup.iloc[:,np.arange(26)].copy()
# 2.2 提取score作为regression的Target
w_scores=worldcup.iloc[:,26].copy()
# 2.3 提取result作为regression的Target
w_results=worldcup.iloc[:,27].copy()

# Coz w_features include numerical or categorical, this is to divided them
# Subset up the total feature set
# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#  w_features_num: numerical features
#  w_features_cat: categorical features 
# drop not for w_features_num use
w_features_num = w_features.drop(['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time'], axis=1,inplace=False)
# keep what w_features_cat need
w_features_cat=w_features[['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time']].copy()

# build a series of pipe, each has its unique funcionalities
# targeting at the numerical features
num_pipeline = Pipeline([
        # just split the data into numerical features data and categorical features data
        ('selector', DataFrameSelector(list(w_features_num))),
        # 用一个中间值来填充丢失的数据NAN
        ('imputer', Imputer(strategy="median")),
        # make each of the features has zero means under unit variance for the sake of the model training. Coz the models sometimes don't accept small value of features
        ('std_scaler', StandardScaler()),
    ])
# target at the category features
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_cat))),
        ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
    ])
    
# they are seperate pipeline by using the FeatureUnion to merge these 2 parallel pipeline into one
# each corresponding to a certain type pf data the numeral 
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

# use full pipeline fit and tranform method apply on your features
# w_features is orignal feature
# use fit_transform and give the index, by using this command all things will be solved. and will get a final prepare feature set
# each feature roughly has the same of arranging varing arranged. so they have very similar values here
# finally get the feature prepared, so this feature prepared will be further fed into the machine learning process, into testing and the training, and a further split the training into validation and the subtraining data
feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(w_features),index=np.arange(1,65))
worldcup_cleaned=pd.concat([feature_prepared,w_scores.to_frame(), w_results.to_frame()], axis=1)
dt_df = pd.DataFrame(feature_prepared,dtype=np.float64)

#回归部分#回归部分#回归部分#回归部分#回归部分#回归部分#回归部分#回归部分#回归部分
from sklearn import datasets

# 这段是我在学习建模时候用的，与作业csv文件无关
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, [2]]
#one target attribute
diabetes_y=diabetes.target


#######################################################version 1#############################################################
# Use only one feature
# 2 从总数据中提取所需要的数据
w_x=feature_prepared.iloc[:,2].copy()
#one target attribute
w_y=w_scores

# Split the data into training/testing sets
w_X_train, w_X_test,w_y_train,w_y_test= \
train_test_split(w_x,w_y,test_size=0.2,random_state=1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(w_X_train, w_y_train)

# Make predictions using the testing set
w_y_pred = regr.predict(w_X_test)
# Make predictions using the testing set
w_y_train_pred = regr.predict(w_X_train)

print(' ')
# The coefficients
print('Coefficients and Intercept are: ', regr.coef_,"   ",regr.intercept_,' respectively')
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(w_y_test, w_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(w_y_test, w_y_pred))
print('******************************************************* ')
print("Mean squared error for training data: %.2f"
      % mean_squared_error(w_y_train, w_y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' % r2_score(w_y_train, w_y_train_pred))

# Plot outputs
plt.scatter(w_X_test, w_y_test,  color='black')
plt.plot(w_X_test, w_y_pred, color='blue', linewidth=3)

plt.xticks(np.arange(start=-0.1,stop=0.2,step=0.06))
plt.yticks(np.arange(500,step=100))

plt.show()

#######################################################version 2#############################################################
# 这部分是我利用多个特征学习方法，可以通过建模
# 数据library内置，与作业无关，需要用作业的数据
# Use two features
diabetes_X = diabetes.data[:, [3,2]]

#one target attribute
diabetes_y=diabetes.target

# Split the data into training/testing sets
diabetes_X_train, diabetes_X_test,diabetes_y_train,diabetes_y_test= \
train_test_split(diabetes_X,diabetes_y,test_size=0.2,random_state=1)



# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients and Intercept are: ', regr.coef_,"   ",regr.intercept_,' respectively')
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

#Plot outputs
fig = plt.figure(1, figsize=(9, 6))
ax = Axes3D(fig)
#plot testing points
ax.scatter(diabetes_X_test[:,0], diabetes_X_test[:,1],diabetes_y_test,depthshade=False,c='black')


x = np.arange(-0.2, 0.2, 0.05)
N = x.size
a,b = np.meshgrid(x,x)
it = np.array([b.ravel(),a.ravel(),np.ones(N*N)]).T
w=np.append(regr.coef_,regr.intercept_)
result=it.dot(w)
#plot fitted hyperplane
ax.plot_surface(np.reshape(it[:,0],(N,N)), np.reshape(it[:,1],(N,N)), np.reshape(result,(N,N)),shade=False,color=(0.,1.,0.,0.2))
#set axis 
ax.set_title("Two features regression")
ax.set_xlabel("4th feature")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("3rd feature")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("target")
ax.w_zaxis.set_ticklabels(np.arange(500,step=100))
plt.show()