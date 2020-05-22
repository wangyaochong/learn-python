import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 20)
y = []
for epsilon in x:
    E = np.array([sp.comb(25, i) * (epsilon ** i) * ((1 - epsilon) ** (25 - i)) for i in range(13, 26)]).sum()
    y.append(E)

plt.plot(x, y, label="estimators are different")
plt.plot(x, x, label="estimators are same")
plt.legend()
plt.show()

# 单个树的正确率至少要有50%，否则使用随机森林结果只会更差
# https://www.bilibili.com/video/BV1P7411P78r?p=19
