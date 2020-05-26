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
#
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# param = {"reg_alpha": np.arange(0, 5, 0.05), "reg_lambda": np.arange(0, 2, 0.05)}
# gscv = GridSearchCV(reg, param_grid=param, scoring="neg_mean_squared_error", cv=cv)
# time0 = time()
# gscv.fit(Xtrain, Ytrain)
# print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
# gscv.best_params_
# gscv.best_score_
# preds = gscv.predict(Xtest)
#
# r2_score(Ytest, preds)
# MSE(Ytest, preds)

axisx = np.arange(0, 5, 0.5)
rs = []
var = []
ge = []

# for i in axisx:
#     reg = XGBR(n_estimators=180, random_state=420, gamma=i)
#     result = CVS(reg, Xtrain, Ytrain, cv=cv)
#     rs.append(result.mean())
#     var.append(result.var())
#     ge.append((1 - result.mean()) ** 2 + result.var())
# print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
# print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
# print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
# rs = np.array(rs)
# var = np.array(var) * 0.1
# plt.figure(figsize=(20, 5))
# plt.plot(axisx, rs, c="black", label="XGB")
# plt.plot(axisx, rs + var, c="red", linestyle='-.')
# plt.plot(axisx, rs - var, c="red", linestyle='-.')
# plt.legend()
# plt.show()

dfull = xgb.DMatrix(X, y)
param1 = {'silent': True, 'obj': 'reg:linear', "gamma": 0}
num_round = 180
n_fold = 5
time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round, n_fold)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
plt.figure(figsize=(20, 5))
plt.grid()

plt.plot(range(1, 181), cvresult1.iloc[:, 0], c="red", label="train,gamma=0")
plt.plot(range(1, 181), cvresult1.iloc[:, 2], c="orange", label="test,gamma=0")
plt.legend()
plt.show()

param1 = {'silent': True, 'obj': 'reg:linear', "gamma": 0}
param2 = {'silent': True, 'obj': 'reg:linear', "gamma": 20}
num_round = 180
n_fold = 5
time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round, n_fold)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_round, n_fold)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
plt.figure(figsize=(20, 5))
plt.grid()
plt.plot(range(1, 181), cvresult1.iloc[:, 0], c="red", label="train,gamma=0")
plt.plot(range(1, 181), cvresult1.iloc[:, 2], c="orange", label="test,gamma=0")
plt.plot(range(1, 181), cvresult2.iloc[:, 0], c="green", label="train,gamma=20")
plt.plot(range(1, 181), cvresult2.iloc[:, 2], c="blue", label="test,gamma=20")
plt.legend()
plt.show()

data2 = load_breast_cancer()
x2 = data2.data
y2 = data2.target
dfull2 = xgb.DMatrix(x2, y2)
param1 = {'silent': True, 'obj': 'binary:logistic', "gamma": 0, "nfold": 5}
param2 = {'silent': True, 'obj': 'binary:logistic', "gamma": 2, "nfold": 5}
num_round = 100
time0 = time()
cvresult1 = xgb.cv(param1, dfull2, num_round, metrics=("error"))
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
time0 = time()
cvresult2 = xgb.cv(param2, dfull2, num_round, metrics=("error"))
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
plt.figure(figsize=(20, 5))
plt.grid()
plt.plot(range(1, 101), cvresult1.iloc[:, 0], c="red", label="train,gamma=0")
plt.plot(range(1, 101), cvresult1.iloc[:, 2], c="orange", label="test,gamma=0")
plt.plot(range(1, 101), cvresult2.iloc[:, 0], c="green", label="train,gamma=2")
plt.plot(range(1, 101), cvresult2.iloc[:, 2], c="blue", label="test,gamma=2")
plt.legend()
plt.show()
