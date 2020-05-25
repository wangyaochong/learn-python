import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt

housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目", "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]
X.head()
Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

reg = LinearRegression().fit(Xtrain, Ytrain)

Ridge_ = Ridge(alpha=0).fit(Xtrain, Ytrain)

lasso_ = Lasso(alpha=0).fit(Xtrain, Ytrain)

Ridge_2 = Ridge(alpha=0.01).fit(Xtrain, Ytrain)

lasso_2 = Lasso(alpha=0.01).fit(Xtrain, Ytrain)

Ridge_3 = Ridge(alpha=10 ** 4).fit(Xtrain, Ytrain)
lasso_3 = Lasso(alpha=10 ** 4).fit(Xtrain, Ytrain)

plt.plot(range(1, 9), (reg.coef_ * 100).tolist(), color="red", label="LR")
plt.plot(range(1, 9), (Ridge_3.coef_ * 100).tolist(), color="orange", label="Ridge")
plt.plot(range(1, 9), (lasso_3.coef_ * 100).tolist(), color="k", label="Lasso")
plt.plot(range(1, 9), [0] * 8, color="grey", linestyle="--")
plt.xlabel('w')  # 横坐标是每一个特征所对应的系数
plt.legend()
plt.show()
