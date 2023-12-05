#!/usr/bin/env python
# coding: utf-8

# Import all the libraries

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# custom scaler class

# In[2]:


class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# special class for new data

# In[3]:


class running_points_model():
    
    def __init__(self, model_file, scaler_file):
        
        with open('model', 'rb') as model_file, open ('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    def load_and_clean_data(self, data_self):
        pddf = pd.read_csv(data_file, delimeter = ",")
        self.pddf_with_predictions = pd.copy()
        pddf = pddf.drop(['Runnings', 'isNotOut'], axis = 1)
        
        points_run = pd.get_dummies(pddf['minutes', 'runningOver'])
        split_data = pdDF['runningScore'].str.split(',', expand=True, n=2)
        split_data.columns = ['Wickets', 'Runnings']
        pdDF = pd.concat([pdDF, split_data], axis=1).drop('runningScore', axis=1)
        
        pdDF['Runnings'] = pdDF['Runnings'].str.replace('runs', '')
        pdDF['Runnings'] = pdDF['Runnings'].str.replace('}', '')
        pdDF['Runnings'] = pdDF['Runnings'].str.replace(':', '')
        pdDF['Runnings'] = pdDF['Runnings'].str.replace("'", '')

        pdDF['Wickets'] = pdDF['Wickets'].str.replace('{', '')
        pdDF['Wickets'] = pdDF['Wickets'].str.replace('wickets', '')
        pdDF['Wickets'] = pdDF['Wickets'].str.replace(':', '')
        pdDF['Wickets'] = pdDF['Wickets'].str.replace("'", '')
        
        pdDF['Wickets'] = pd.to_numeric(pdDF['Wickets'], errors='coerce').fillna(0).astype(int)
        pdDF['Runnings'] = pd.to_numeric(pdDF['Runnings'], errors='coerce').fillna(0).astype(int)
        pdDF['strikeRate'] = pd.to_numeric(pdDF['strikeRate'], errors='coerce').fillna(0).astype(int)
        
        self.preprocessed_data = pddf.copy()
        self.data = self.scaler.transform(pddf)
        
    def predicted_probability(self):
        if (self.date is not None):
            pred = self.reg.predict_proba(self.data)[ : , 1]
            return pred
    
    def predicted_output_category(self):
        if (self.dat is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[ : , 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data


# LinkedIn Ismael (Ishmael T.) Ngobeni

# In[ ]:




