from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
from sklearn.decomposition import PCA
import matplotlib
import random
import pandas as pd

font = {"weight": "bold", "size": "20", "family": "KaiTi"}
matplotlib.rc("font", **font)

data = load_breast_cancer()
X = data.data
y = data.target

X_dr = PCA(2).fit_transform(X)
plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(X_dr[:, 0], X_dr[:, 1], c=y, s=1)
plt.title("两个成分的分布")
plt.show()  # 显示主成分分布

X.shape
np.unique(y)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# Kernel = ["linear", "poly", "rbf", "sigmoid"] #这个会需要比较长的运行时间
# for kernel in Kernel:
#     time0 = time()
#     clf = SVC(kernel=kernel, gamma="auto"  # , degree = 1 #degree是多项式核函数的次数
#               , cache_size=5000
#               ).fit(Xtrain, Ytrain)
#     print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
#     print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

Kernel = ["linear", "rbf", "sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf = SVC(kernel=kernel, gamma="auto"  # , degree = 1
              , cache_size=5000
              ).fit(Xtrain, Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

Kernel = ["linear", "poly", "rbf", "sigmoid"]
print("*" * 100)
for kernel in Kernel:
    time0 = time()
    clf = SVC(kernel=kernel, gamma="auto", degree=1,  # degree是多项式核函数的次数
              cache_size=5000
              ).fit(Xtrain, Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

print("统一量纲", "*" * 100)
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
data = pd.DataFrame(X)
data.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
Kernel = ["linear", "poly", "rbf", "sigmoid"]
for kernel in Kernel:
    time0 = time()
    clf = SVC(kernel=kernel, gamma="auto", degree=1, cache_size=5000
              ).fit(Xtrain, Ytrain)
    print("The accuracy under kernel %s is %f" % (kernel, clf.score(Xtest, Ytest)))
    print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
print("统一量纲后rbf准确率明显上升")

score = []
gamma_range = np.logspace(-10, 1,
                          50)  # 返回在对数刻度上均匀间隔的数字
for i in gamma_range:
    clf = SVC(kernel="rbf", gamma=i, cache_size=5000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
print(max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range, score)
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

time0 = time()
gamma_range = np.logspace(-10, 1, 20)
coef0_range = np.linspace(0, 5, 10)
param_grid = dict(gamma=gamma_range, coef0=coef0_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=420)
grid = GridSearchCV(SVC(kernel="poly", degree=1, cache_size=5000), param_grid=param_grid, cv=cv)
grid.fit(X, y)
print("The best parameters are %s, score = %0.5f" % (grid.best_params_, grid.best_score_))
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

# 调参C
score = []
C_range = np.linspace(0.01, 30, 50)
for i in C_range:
    clf = SVC(kernel="linear", C=i, cache_size=5000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()

# 换rbf
score = []
C_range = np.linspace(0.01, 30, 50)
for i in C_range:
    clf = SVC(kernel="rbf", C=i, gamma=0.012742749857031322, cache_size=5000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()
# 进一步细化
score = []
C_range = np.linspace(5, 7, 50)
for i in C_range:
    clf = SVC(kernel="rbf", C=i, gamma=0.012742749857031322, cache_size=5000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()
