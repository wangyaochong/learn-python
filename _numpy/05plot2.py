import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import random

font = {"family": "KaiTi"}
matplotlib.rc("font", **font)

data = np.loadtxt("./USvideos.csv", skiprows=1, delimiter=",", dtype=float)
data = data[data[:, 2] < 500000]  # 过滤喜欢数量比500000小的
print(data)

comment = data[:, 4].astype(int)
like = data[:, 2].astype(int)
print(comment, max(comment), min(comment))

plt.scatter(like, comment)
plt.show()
