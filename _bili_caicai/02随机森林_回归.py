from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boston = load_boston()

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
score = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error")

scoring_key = sklearn.metrics.SCORERS.keys()
print("查看可用的scoring方法", sorted(list(scoring_key)))

