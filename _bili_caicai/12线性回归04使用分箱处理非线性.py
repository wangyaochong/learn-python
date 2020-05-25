import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from sklearn.model_selection import cross_val_score as CVS

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)  # random.uniform，从输入的任意两个整数中取出size个随机数
# 生成y的思路：先使用NumPy中的函数生成一个sin函数图像，然后再人为添加噪音
y = np.sin(X) + rnd.normal(size=len(X)) / 3  # random.normal，生成size个服从正态分布的随机数
# 使用散点图观察建立的数据集是什么样子
plt.scatter(X, y, marker='o', c='k', s=20)
plt.show()
# 为后续建模做准备：sklearn只接受二维以上数组作为特征矩阵的输入 X.shape
X = X.reshape(-1, 1)

LinearR = LinearRegression().fit(X, y)
TreeR = DecisionTreeRegressor(random_state=0).fit(X, y)
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

# enc = KBinsDiscretizer(n_bins=10  # 分几类？
#                        , encode="onehot")  # ordinal
# X_binned = enc.fit_transform(X)
# x_bin_pandas = pd.DataFrame(X_binned.toarray())

# LinearR_ = LinearRegression().fit(X_binned, y)
# # LinearR_.predict(line)  # line作为测试集
# line_binned = enc.transform(line)
# LinearR_.predict(line_binned)

enc = KBinsDiscretizer(n_bins=10, encode="onehot")
X_binned = enc.fit_transform(X)
line_binned = enc.transform(line)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True  # 让两张图共享y轴上的刻度
                               , figsize=(10, 4))

ax1.plot(line, LinearR.predict(line), linewidth=2, color='green', label="linear regression")
ax1.plot(line, TreeR.predict(line), linewidth=2, color='red', label="decision tree")
ax1.plot(X[:, 0], y, 'o', c='k')
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Result before discretization")
LinearR_ = LinearRegression().fit(X_binned, y)
TreeR_ = DecisionTreeRegressor(random_state=0).fit(X_binned, y)

ax2.plot(line
         # 横坐标
         , LinearR_.predict(line_binned)
         # 分箱后的特征矩阵的结果
         , linewidth=2, color='green', linestyle='-', label='linear regression')

ax2.plot(line, TreeR_.predict(line_binned), linewidth=2, color='red', linestyle=':', label='decision tree')
ax2.vlines(enc.bin_edges_[0]  # x轴
           , *plt.gca().get_ylim()
           # y轴的上限和下限
           , linewidth=1, alpha=.2)

ax2.plot(X[:, 0], y, 'o', c='k')
ax2.legend(loc="best")
ax2.set_xlabel("Input feature")
ax2.set_title("Result after discretization")
plt.tight_layout()
plt.show()

pred, score, var = [], [], []
binsrange = [2, 5, 10, 15, 20, 30]
for i in binsrange:
    enc = KBinsDiscretizer(n_bins=i, encode="onehot")
    X_binned = enc.fit_transform(X)
    line_binned = enc.transform(line)
    LinearR_ = LinearRegression()
    cvresult = CVS(LinearR_, X_binned, y, cv=5)
    score.append(cvresult.mean())
    var.append(cvresult.var())
    pred.append(LinearR_.fit(X_binned, y).score(line_binned, np.sin(line)))
plt.figure(figsize=(6, 5))
plt.plot(binsrange, pred, c="orange", label="test")
plt.plot(binsrange, score, c="k", label="full data")
plt.plot(binsrange, score + np.array(var) * 0.5, c="red", linestyle="--", label="var")
plt.plot(binsrange, score - np.array(var) * 0.5, c="red", linestyle="--")
plt.legend()
plt.show()
