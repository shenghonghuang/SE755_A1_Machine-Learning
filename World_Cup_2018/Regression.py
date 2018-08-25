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

#1 首先获取整个表的内容，去掉一列不需要的
worldcup=pd.read_csv('2018 worldcup.csv',index_col=0)
#match date is assumed to be irrelevant for the match results
worldcup.drop(['Date'],axis=1,inplace=True)
worldcup.describe()

#将文字转换成数值型的

#先统计一下位置信息
Location=worldcup['Location'].unique()
print('Location={0}'.format(len(Location)))
#
Phase=worldcup['Phase'].unique()
print('Phase={0}'.format(len(Phase)))

# drop掉不相关信息
worldcup=worldcup.drop(['Location','Normal_Time'],axis=1)
print(worldcup.shape)

#继续筛选特征，去掉['Team1','Team1_Continent','Team2','Team2_Continent']
worldcup=worldcup.drop(['Team1','Team1_Continent','Team2','Team2_Continent'],axis=1)

#以上是我们抽取出来的通用的特征
#选取Score作为target,并忽略Win/loss/draw属性
#用不到的Match_result，Team1_Pass_Accuracy(%)，Team2_Pass_Accuracy(%)，Team1_Ball_Possession(%)，Team2_Ball_Possession(%)
worldcup=worldcup.drop(['Match_result','Team1_Pass_Accuracy(%)','Team2_Pass_Accuracy(%)','Team1_Ball_Possession(%)','Team2_Ball_Possession(%)',],axis=1)

print(worldcup.columns)
#利用pandas进行one-hot编码
worldcup=pd.get_dummies(worldcup,prefix='Phase')
print(worldcup.shape)

featrues=['Team1_Attempts', 'Team1_Corners', 'Team1_Offsides',
        'Team1_Distance_Covered',
       'Team1_Ball_Recovered', 'Team1_Yellow_Card', 'Team1_Red_Card',
       'Team1_Fouls', 'Team2_Attempts', 'Team2_Corners', 'Team2_Offsides',
       'Team2_Distance_Covered', 'Team2_Ball_Recovered', 'Team2_Yellow_Card',
       'Team2_Red_Card', 'Team2_Fouls',
       'Phase_Group', 'Phase_Knockout']
target=['Total_Scores']
print(worldcup.head(5))
x=worldcup[featrues].values
y=worldcup[target].values
#将数据分成训练集和测试集，用测试集评价模型
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
#使用sklearn库构建模型

#=======================Ordinary Regression============

lr = linear_model.LinearRegression().fit(x_train,y_train)
y_predict=lr.predict(x_test)
print(lr.score(x_test, y_test))
# print model
p1=plt.scatter(range(len(x_test)), y_test,  color='black')
#用predict预测，这里预测输入x对应的值，进行画线
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
p1=plt.scatter(range(len(x_test)), y_test,  color='black')
#用predict预测，这里预测输入x对应的值，进行画线
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
