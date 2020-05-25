from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch  # 加利福尼亚房屋价值数据集
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
import sklearn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

housevalue = fch()
# 会需要下载，大家可以提前运行试试看
X = pd.DataFrame(housevalue.data)
# 放入DataFrame中便于查看
y = housevalue.target
X.shape
y.shape
X.head()
housevalue.feature_names
X.columns = housevalue.feature_names

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])
Xtrain.shape

reg = LR().fit(Xtrain, Ytrain)
coef = reg.coef_
intercept = reg.intercept_
map_coef = [*zip(Xtrain.columns, reg.coef_)]
yhat = reg.predict(Xtest)
yhat

print("mse", MSE(yhat, Ytest))

# score1 = cross_val_score(reg, X, y, cv=10, scoring="mean_squared_error")
score2 = cross_val_score(reg, X, y, cv=10, scoring="neg_mean_squared_error")

r2_score(yhat, Ytest)
r2 = reg.score(Xtest, Ytest)
print("r2", r2)

sorted(Ytest)
plt.plot(range(len(Ytest)), sorted(Ytest), c="black", label="Data")
plt.plot(range(len(yhat)), sorted(yhat), c="red", label="Predict")
plt.legend()
plt.show()

# 关于r2指标为什么是负数
rng = np.random.RandomState(42)
X = rng.randn(100, 80)
y = rng.randn(100)
score3 = cross_val_score(LR(), X, y, cv=5, scoring='r2')
