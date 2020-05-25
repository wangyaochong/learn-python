import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib

# font = {"weight": "bold", "size": "20", "family": "KaiTi"}
# matplotlib.rc("font", **font)

# 不推荐使用岭迹图

X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
coefs = []

for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
# 绘图展示结果
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
# 将横坐标逆转
plt.xlabel('alpha')
plt.ylabel('w')
plt.axis('tight')
plt.show()
