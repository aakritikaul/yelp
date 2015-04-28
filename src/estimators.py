'''
Roll an estimator for prediction.
----------
AUTHOR: CHIA YING LEE
DATE: 10 APRIL 2015
----------
'''

import numpy as np
import pandas as pd
from sklearn import base

'''
# Template for Estimator class 
class Estimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self):
        # initialization code

    def fit(self, X, y):
        return self

    def predict(self, X):
        return # prediction
    
    def score(self, X, y):
        return np.sum((y - self.predict(X))**2) # the score
'''

class City_Based_Estimator(base.BaseEstimator, base.RegressorMixin):
    def __init__(self):
        self.mean_rating_by_city_ = []
        
    def fit(self, X, y):
        data = pd.DataFrame({ 'city' : X['city'], 'stars' : y })
        self.mean_rating_by_city_ = data.groupby('city')['stars'].mean()
        return self
        
    def predict(self, X):
        return X['city'].apply(lambda x: self.mean_rating_by_city_[x])

    def score(self, X, y):
        return np.mean((y - self.predict(X))**2)
