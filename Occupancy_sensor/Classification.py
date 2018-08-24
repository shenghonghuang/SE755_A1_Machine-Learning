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

occupancy=pd.read_csv(r'D:\Projects\755\SE755_A1_Machine-Learning\Occupancy_sensor\occupancy_sensor_data.csv',index_col=0)
occupancy.describe()

#attributes
o_features=occupancy.iloc[:,0:5].copy()
o_target=occupancy.iloc[:,5].copy()

for i in range(0,len(x)):
    for j in range(0,len(x[i][j])):
        thisdata=x[i][j]
        if (thisdata==0):
            x[i][j]=int(1)
        else:
            x[i][j]=int(0)
