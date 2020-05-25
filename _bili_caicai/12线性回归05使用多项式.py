import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from sklearn.model_selection import cross_val_score as CVS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
import numpy as np

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)  # random.uniform，从输入的任意两个整数中取出size个随机数
# 生成y的思路：先使用NumPy中的函数生成一个sin函数图像，然后再人为添加噪音
y = np.sin(X) + rnd.normal(size=len(X)) / 3  # random.normal，生成size个服从正态分布的随机数
# 使用散点图观察建立的数据集是什么样子
plt.scatter(X, y, marker='o', c='k', s=20)
plt.show()
# 为后续建模做准备：sklearn只接受二维以上数组作为特征矩阵的输入 X.shape
X = X.reshape(-1, 1)

line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
enc = KBinsDiscretizer(n_bins=10, encode="onehot")
X_binned = enc.fit_transform(X)
line_binned = enc.transform(line)

LinearR_ = LinearRegression().fit(X_binned, y)
TreeR_ = DecisionTreeRegressor(random_state=0).fit(X_binned, y)

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)
y = np.sin(X) + rnd.normal(size=len(X)) / 3
X = X.reshape(-1, 1)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
LinearR = LinearRegression().fit(X, y)
scorer = LinearR.score(X, y)
scorer2 = LinearR.score(line, np.sin(line))

d = 3
poly = PF(degree=d)
X_ = poly.fit_transform(X)
line_ = PF(degree=d).fit_transform(line)
LinearR_ = LinearRegression().fit(X_, y)

scoreh = LinearR_.score(X_, y)
scoreh2 = LinearR_.score(line_, np.sin(line))

LinearR = LinearRegression().fit(X, y)
X_ = PF(degree=d).fit_transform(X)
LinearR_ = LinearRegression().fit(X_, y)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
line_ = PF(degree=d).fit_transform(line)

fig, ax1 = plt.subplots(1)
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green', label="linear regression")
ax1.plot(line, LinearR_.predict(line_), linewidth=2, color='red', label="Polynomial regression")
ax1.plot(X[:, 0], y, 'o', c='k')
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Linear Regression ordinary vs poly")
plt.tight_layout()
plt.show()
