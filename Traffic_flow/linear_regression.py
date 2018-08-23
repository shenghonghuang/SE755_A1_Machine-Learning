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

traffic=pd.read_csv('traffic_flow_data.csv',index_col=0)
traffic.describe()

#traffic attributes
t_features=traffic.iloc[:,np.arange(448)].copy()

t_target=traffic.iloc[:,449].copy()
