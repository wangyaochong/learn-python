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


def regassess(reg, Xtrain, Ytrain, cv, scoring=["r2"], show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i], CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean()))
        score.append(CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean())
    return score


data = load_boston()

X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)
reg.predict(Xtest)  # 传统接口predict
score = reg.score(Xtest, Ytest)  # 你能想出这里应该返回什么模型评估指标么？
MSE(Ytest, reg.predict(Xtest))
reg.feature_importances_

reg = XGBR(n_estimators=100)
CVS(reg, Xtrain, Ytrain, cv=5).mean()
CVS(reg, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()
sorted(sklearn.metrics.SCORERS.keys())

rfr = RFR(n_estimators=100)
CVS(rfr, Xtrain, Ytrain, cv=5).mean()
CVS(rfr, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()
lr = LinearR()
CVS(lr, Xtrain, Ytrain, cv=5).mean()
CVS(lr, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()
reg = XGBR(n_estimators=10, silent=True)
CVS(reg, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()


def plot_learning_curve(estimator, title, X, y, ax=None,
                        # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, shuffle=True, cv=cv
                                                            # ,random_state=420
                                                            , n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Test score")
    ax.legend(loc="best")
    return ax


cv = KFold(n_splits=5, shuffle=True, random_state=42)
plot_learning_curve(XGBR(n_estimators=100, random_state=420), "XGB", Xtrain, Ytrain, ax=None, cv=cv)
plt.show()

axisx = range(10, 1010, 50)
rs = []

for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    rs.append(CVS(reg, Xtrain, Ytrain, cv=cv).mean())

print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="red", label="XGB")
plt.legend()
plt.show()

axisx = range(50, 1050, 50)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="red", label="XGB")
plt.legend()
plt.show()

axisx = range(100, 300, 10)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var) * 0.01
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="black", label="XGB")
plt.plot(axisx, rs + var, c="red", linestyle='-.')
plt.plot(axisx, rs - var, c="red", linestyle='-.')
plt.legend()
plt.show()
plt.figure(figsize=(20, 5))
plt.plot(axisx, ge, c="gray", linestyle='-.')
plt.show()

time0 = time()
print(XGBR(n_estimators=100, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)
time0 = time()
print(XGBR(n_estimators=660, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)
time0 = time()
print(XGBR(n_estimators=180, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

axisx = np.linspace(0, 1, 20)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=180, subsample=i, random_state=420)
    rs.append(CVS(reg, Xtrain, Ytrain, cv=cv).mean())

print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="green", label="XGB")
plt.legend()
plt.show()

axisx = np.linspace(0.05, 1, 20)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=180, subsample=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var)
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="black", label="XGB")
plt.plot(axisx, rs + var, c="red", linestyle='-.')
plt.plot(axisx, rs - var, c="red", linestyle='-.')
plt.legend()
plt.show()

axisx = np.linspace(0.75, 1, 25)
reg = XGBR(n_estimators=180, subsample=0.7708333333333334, random_state=420).fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)
MSE(Ytest, reg.predict(Xtest))

regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"])
regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"], show=False)
for i in [0, 0.2, 0.5, 1]:
    time0 = time()
    reg = XGBR(n_estimators=180, random_state=420, learning_rate=i)
    print("learning_rate = {}".format(i))
    regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"])
    print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
    print("\t")

axisx = np.arange(0.05, 1, 0.05)
rs = []
te = []
for i in axisx:
    reg = XGBR(n_estimators=180, random_state=420, learning_rate=i)
    score = regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"], show=False)
    test = reg.fit(Xtrain, Ytrain).score(Xtest, Ytest)
    rs.append(score[0])
    te.append(test)
print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20, 5))
plt.plot(axisx, te, c="gray", label="XGB")
plt.plot(axisx, rs, c="green", label="XGB")
plt.legend()
plt.show()

for booster in ["gbtree", "gblinear", "dart"]:
    reg = XGBR(n_estimators=180, learning_rate=0.1, random_state=420, booster=booster).fit(Xtrain, Ytrain)
    print(booster)
    print(reg.score(Xtest, Ytest))  # 自己找线性数据试试看"gblinear"的效果吧~

reg = XGBR(n_estimators=180, random_state=420).fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)
MSE(Ytest, reg.predict(Xtest))

dtrain = xgb.DMatrix(Xtrain, Ytrain)
dtest = xgb.DMatrix(Xtest, Ytest)
param = {'silent': True, 'objective': 'reg:linear', "eta": 0.1}
num_round = 180
bst = xgb.train(param, dtrain, num_round)
r2_score(Ytest, bst.predict(dtest))
MSE(Ytest, bst.predict(dtest))
