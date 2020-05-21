from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
import graphviz
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(1)
x = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

r1 = tree.DecisionTreeRegressor(max_depth=2)
r2 = tree.DecisionTreeRegressor(max_depth=5)

r1.fit(x, y)
r2.fit(x, y)

x_test = np.arange(0, 5, 0.01)[:, np.newaxis]  # 增维
y_1 = r1.predict(x_test)
y_2 = r2.predict(x_test)

plt.scatter(x, y)
plt.plot(x_test, y_1, label="max_depth=2")
plt.plot(x_test, y_2, label="max_depth=5")  # max_depth=5是过拟合的
plt.show()
