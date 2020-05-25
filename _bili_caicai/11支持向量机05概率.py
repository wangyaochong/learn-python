import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as LogiR
import pandas as pd
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R

class_1 = 500  # 类别1有500个样本
class_2 = 50  # 类别2只有50个
centers = [[0.0, 0.0], [2.0, 2.0]]  # 设定两个类别的中心
clusters_std = [1.5, 0.5]  # 设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples=[class_1, class_2], centers=centers, cluster_std=clusters_std, random_state=0,
                  shuffle=False)
# 看看数据集长什么样
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", s=10)
plt.show()
# 其中红色点是少数类，紫色点是多数类
clf_proba = svm.SVC(kernel="linear", C=1.0, probability=True).fit(X, y)
proba = clf_proba.predict_proba(X)
clf_proba.predict_proba(X).shape
deci = clf_proba.decision_function(X)
clf_proba.decision_function(X).shape
