# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:50:29 2018

@author: HSH
"""


import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
Phase=worldcup['Phase'].unique()
print('Phase={0}'.format(len(Phase)))
# Drop irrelevant column
worldcup=worldcup.drop(['Location','Normal_Time'],axis=1)
print(worldcup.shape)

# Drop irrelevant columns
worldcup=worldcup.drop(['Team1','Team1_Continent','Team1_Corners','Team1_Offsides','Team1_Ball_Possession(%)','Team1_Pass_Accuracy(%)','Team1_Distance_Covered','Team1_Ball_Recovered','Team1_Yellow_Card','Team1_Red_Card','Team1_Fouls','Team2','Team2_Continent','Team2_Corners','Team2_Offsides','Team2_Ball_Possession(%)','Team2_Pass_Accuracy(%)','Team2_Distance_Covered','Team2_Ball_Recovered','Team2_Yellow_Card','Team2_Red_Card','Team2_Fouls'],axis=1)

# Drop irrelevant column for features
worldcup=worldcup.drop(['Total_Scores'],axis=1)

#将target转换成数值型的label
le =preprocessing.LabelEncoder()
result=worldcup['Match_result'].unique()
le.fit(result)
worldcup['Match_result']=le.transform(worldcup['Match_result'].values)

worldcup=pd.get_dummies(worldcup,prefix=['Phase'])
print(worldcup.head(5))
print(worldcup.columns)

'''features=['Team1_Attempts', 'Team1_Corners', 'Team1_Offsides',
       'Team1_Ball_Possession(%)', 'Team1_Pass_Accuracy(%)',
       'Team1_Distance_Covered', 'Team1_Ball_Recovered', 'Team1_Yellow_Card',
       'Team1_Red_Card', 'Team1_Fouls', 'Team2_Attempts', 'Team2_Corners',
       'Team2_Offsides', 'Team2_Ball_Possession(%)', 'Team2_Pass_Accuracy(%)',
       'Team2_Distance_Covered', 'Team2_Ball_Recovered', 'Team2_Yellow_Card',
       'Team2_Red_Card', 'Team2_Fouls',  'Phase_Group',
       'Phase_Knockout']
'''

features=['Team1_Attempts', 'Team2_Attempts', 'Phase_Group','Phase_Knockout']
target=['Match_result']
#打乱顺序
worldcup = shuffle(worldcup)
x=worldcup[features].values
y=worldcup[target].values
#为了评价模型的性能，我们将数据分成训练集和测试集，用测试集评价模型
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
#===================Perceptron=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)  #y=w.x+b
multi_target_ppn = MultiOutputClassifier(ppn)  # 构建多输出多分类器
y_pred = multi_target_ppn.fit(x_train, y_train).predict(x_test)
print('Perceptron')
print(classification_report(y_test,y_pred))

#===================SVM=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
#调用SVC()
clf = svm.SVC()
multi_target_clf = MultiOutputClassifier(clf)  # 构建多输出多分类器
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('SVM:')
print(classification_report(y_test,y_pred))

#===================Decision Trees=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
multi_target_clf = MultiOutputClassifier(clf)  # 构建多输出多分类器
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('Decision Tree:')
print(classification_report(y_test,y_pred))

#=================== Nearest neighbour classifier=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
multi_target_clf = MultiOutputClassifier(clf)  # 构建多输出多分类器
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('Nearest Neighbour classifier:')
print(classification_report(y_test,y_pred))

#===================  Naïve Bayes classifier=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
multi_target_clf = MultiOutputClassifier(clf)  # 构建多输出多分类器
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('Naive Bayes classifier:')
print(classification_report(y_test,y_pred))

