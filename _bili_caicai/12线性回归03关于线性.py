import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)  # random.uniform，从输入的任意两个整数中取出size个随机数
# 生成y的思路：先使用NumPy中的函数生成一个sin函数图像，然后再人为添加噪音
y = np.sin(X) + rnd.normal(size=len(X)) / 3  # random.normal，生成size个服从正态分布的随机数
# 使用散点图观察建立的数据集是什么样子
plt.scatter(X, y, marker='o', c='k', s=20)
plt.show()
X = X.reshape(-1, 1)

LinearR = LinearRegression().fit(X, y)
TreeR = DecisionTreeRegressor(random_state=0
                              # , min_samples_split=3
                              # , min_samples_leaf=3
                              ).fit(X, y)
fig, ax1 = plt.subplots(1)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

ax1.plot(line, LinearR.predict(line), linewidth=2, color='green', label="linear regression")
ax1.plot(line, TreeR.predict(line), linewidth=2, color='red', label="decision tree")

ax1.plot(X[:, 0], y, 'o', c='k')
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")

ax1.set_title("Result before discretization")
plt.tight_layout()
plt.show()
