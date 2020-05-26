from xgboost import XGBRegressor as XGBR
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime
from time import time
import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.datasets import load_breast_cancer
import sys
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as TTS
import pickle

data = load_boston()

X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)
reg.predict(Xtest)  # 传统接口predict
score = reg.score(Xtest, Ytest)  # 你能想出这里应该返回什么模型评估指标么？
MSE(Ytest, reg.predict(Xtest))

dtrain = xgb.DMatrix(Xtrain, Ytrain)
param = {'silent': True, 'obj': 'reg:linear', "subsample": 1, "eta": 0.05, "gamma": 20, "lambda": 3.5, "alpha": 0.2,
         "max_depth": 4, "colsample_bytree": 0.4, "colsample_bylevel": 0.6, "colsample_bynode": 1}
num_round = 180
bst = xgb.train(param, dtrain, num_round)
model_path = "./xgboostonboston.dat"
pickle.dump(bst, open(model_path, "wb"))

data = load_boston()
X = data.data
y = data.target
Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
dtest = xgb.DMatrix(Xtest, Ytest)

loaded_model = pickle.load(open(model_path, "rb"))
print("Loaded model from: xgboostonboston.dat")

ypreds = loaded_model.predict(dtest)

MSE(Ytest, ypreds)
r2_score(Ytest, ypreds)

bst = xgb.train(param, dtrain, num_round)
import joblib

joblib.dump(bst, "xgboost-boston.dat")
loaded_model = joblib.load("xgboost-boston.dat")
ypreds = loaded_model.predict(dtest)
MSE(Ytest, ypreds)
r2_score(Ytest, ypreds)

from xgboost import XGBRegressor as XGBR

bst = XGBR(n_estimators=200, eta=0.05, gamma=20, reg_lambda=3.5, reg_alpha=0.2, max_depth=4, colsample_bytree=0.4,
           colsample_bylevel=0.6).fit(Xtrain, Ytrain)
joblib.dump(bst, "xgboost-boston.dat")
loaded_model = joblib.load("xgboost-boston.dat")

ypreds = loaded_model.predict(Xtest)
MSE(Ytest, ypreds)
