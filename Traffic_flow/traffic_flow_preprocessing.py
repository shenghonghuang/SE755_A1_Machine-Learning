# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:27:20 2018

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

traffic=pd.read_csv(r'D:\Projects\755\SE755_A1_Machine-Learning\Traffic_flow\traffic_flow_data.csv',index_col=0)
#match date is assumed to be irrelevant for the match results
traffic.drop(['Date','Team1_Ball_Possession(%)'],axis=1,inplace=True)
traffic.describe()

#traffic attributes
w_features=traffic.iloc[:,np.arange(26)].copy()
#traffic goal result
w_goals=traffic.iloc[:,26].copy()
#traffic match result
w_results=traffic.iloc[:,27].copy()


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
w_features_num = w_features.drop(['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time'], axis=1,inplace=False)
w_features_cat=w_features[['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time']].copy()


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_num))),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_cat))),
        ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
    ])



full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(w_features),index=np.arange(1,65))
traffic_cleaned=pd.concat([feature_prepared,w_goals.to_frame(), w_results.to_frame()], axis=1)
