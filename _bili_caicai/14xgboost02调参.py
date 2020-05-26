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

data = load_boston()

X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)
reg.predict(Xtest)  # 传统接口predict
score = reg.score(Xtest, Ytest)  # 你能想出这里应该返回什么模型评估指标么？
MSE(Ytest, reg.predict(Xtest))

dfull = xgb.DMatrix(X, y)

param1 = {'silent': True
          # 并非默认
    , 'obj': 'reg:linear'  # 并非默认
    , "subsample": 1, "max_depth": 6, "eta": 0.3, "gamma": 0, "lambda": 1, "alpha": 0, "colsample_bytree": 1,
          "colsample_bylevel": 1, "colsample_bynode": 1, "nfold": 5}
num_round = 200
time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
fig, ax = plt.subplots(1, figsize=(15,
                                   10))
# ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
ax.legend(fontsize="xx-large")
plt.show()

param1 = {'silent': True, 'obj': 'reg:linear', "subsample": 1, "max_depth": 6, "eta": 0.3, "gamma": 0, "lambda": 1,
          "alpha": 0, "colsample_bytree": 1, "colsample_bylevel": 1, "colsample_bynode": 1, "nfold": 5}
num_round = 200
time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
fig, ax = plt.subplots(1, figsize=(15, 8))
ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
param2 = {'silent': True, 'obj': 'reg:linear', "nfold": 5}
param3 = {'silent': True, 'obj': 'reg:linear', "nfold": 5}
time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
time0 = time()
cvresult3 = xgb.cv(param3, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
ax.plot(range(1, 201), cvresult2.iloc[:, 0], c="green", label="train,last")
ax.plot(range(1, 201), cvresult2.iloc[:, 2], c="blue", label="test,last")
ax.plot(range(1, 201), cvresult3.iloc[:, 0], c="gray", label="train,this")
ax.plot(range(1, 201), cvresult3.iloc[:, 2], c="pink", label="test,this")
ax.legend(fontsize="xx-large")
plt.show()

param1 = {'silent': True, 'obj': 'reg:linear', "subsample": 1, "max_depth": 6, "eta": 0.3, "gamma": 0, "lambda": 1,
          "alpha": 0, "colsample_bytree": 1, "colsample_bylevel": 1, "colsample_bynode": 1, "nfold": 5}
num_round = 200
time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
fig, ax = plt.subplots(1, figsize=(15, 8))
ax.set_ylim(top=5)
ax.grid()

ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
param2 = {'silent': True, 'obj': 'reg:linear', "subsample": 1, "eta": 0.05, "gamma": 20, "lambda": 3.5, "alpha": 0.2,
          "max_depth": 4, "colsample_bytree": 0.4, "colsample_bylevel": 0.6, "colsample_bynode": 1, "nfold": 5}

param3 = {'silent': True, 'obj': 'reg:linear', "max_depth": 2, "eta": 0.05, "gamma": 0, "lambda": 1, "alpha": 0,
          "colsample_bytree": 1, "colsample_bylevel": 0.4, "colsample_bynode": 1, "nfold": 5}
time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
ax.plot(range(1, 201), cvresult2.iloc[:, 0], c="green", label="train,final")
ax.plot(range(1, 201), cvresult2.iloc[:, 2], c="blue", label="test,final")
ax.legend(fontsize="xx-large")
plt.show()
