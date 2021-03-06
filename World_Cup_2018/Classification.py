# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:50:29 2018

@author: HSH
"""


import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_similarity_score, matthews_corrcoef, zero_one_loss
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
worldcup=worldcup.drop(['Team1','Team1_Continent','Team1_Corners','Team1_Offsides','Team1_Ball_Possession(%)','Team1_Pass_Accuracy(%)','Team1_Distance_Covered','Team1_Ball_Recovered','Team2','Team2_Continent','Team2_Corners','Team2_Offsides','Team2_Ball_Possession(%)','Team2_Pass_Accuracy(%)','Team2_Distance_Covered','Team2_Ball_Recovered','Total_Scores'],axis=1)

#Convert target into number label
le =preprocessing.LabelEncoder()
result=worldcup['Match_result'].unique()
le.fit(result)
worldcup['Match_result']=le.transform(worldcup['Match_result'].values)

worldcup=pd.get_dummies(worldcup,prefix=['Phase'])
print(worldcup.head(5))
print(worldcup.columns)

features=['Team2_Yellow_Card','Team2_Red_Card', 'Team2_Fouls']
target=['Match_result']

worldcup = shuffle(worldcup)
x=worldcup[features].values
y=worldcup[target].values
# Split the dataset into training dataset and testing dataset
x_train, x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)
#===================Perceptron=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)  #y=w.x+b
multi_target_ppn = MultiOutputClassifier(ppn)
y_pred = multi_target_ppn.fit(x_train, y_train).predict(x_test)
print('Perceptron')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))

#===================SVM=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
#调用SVC()
clf = svm.SVC()
multi_target_clf = MultiOutputClassifier(clf)  # Create Multi Output Classifier
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('SVM:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))

#===================Decision Trees=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
multi_target_clf = MultiOutputClassifier(clf)  # Create Multi Output Classifier
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('Decision Tree:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))

#=================== Nearest neighbour classifier=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
multi_target_clf = MultiOutputClassifier(clf) # Create Multi Output Classifier
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('Nearest Neighbour classifier:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))

#===================  Naïve Bayes classifier=========================
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
multi_target_clf = MultiOutputClassifier(clf)  # Create Multi Output Classifier
y_pred = multi_target_clf.fit(x_train, y_train).predict(x_test)
print('Naive Bayes classifier:')
print(classification_report(y_test,y_pred))
print('Accuracy classification score: %.2f' % accuracy_score(y_test,y_pred))
print('Average Hamming loss: %.2f' % hamming_loss(y_test,y_pred))
print('Jaccard similarity coefficient score: %.2f' % jaccard_similarity_score(y_test,y_pred))
print('Matthews correlation coefficient (MCC): %.2f' % matthews_corrcoef(y_test,y_pred))
print('Zero-one classification loss: %.2f' % zero_one_loss(y_test,y_pred))