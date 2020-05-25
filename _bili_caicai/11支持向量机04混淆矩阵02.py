import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as LogiR
import pandas as pd
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R

class_1_ = 7
class_2_ = 4
centers_ = [[0.0, 0.0], [1, 1]]
clusters_std = [0.5, 1]
X_, y_ = make_blobs(n_samples=[class_1_, class_2_], centers=centers_, cluster_std=clusters_std, random_state=0,
                    shuffle=False)
plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap="rainbow", s=30)
plt.show()
clf_lo = LogiR().fit(X_, y_)
prob = clf_lo.predict_proba(X_)
# 将样本和概率放到一个DataFrame中
prob = pd.DataFrame(prob)
prob.columns = ["0", "1"]

for i in range(prob.shape[0]):
    if prob.loc[i, "1"] > 0.5:
        prob.loc[i, "pred"] = 1
    else:
        prob.loc[i, "pred"] = 0
prob["y_true"] = y_
prob = prob.sort_values(by="1", ascending=False)

cm = CM(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
# 试试看手动计算Precision和Recall?
precision = P(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
recall = R(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])

for i in range(prob.shape[0]):
    if prob.loc[i, "1"] > 0.4:
        prob.loc[i, "pred"] = 1
    else:
        prob.loc[i, "pred"] = 0
cm2 = CM(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
# 试试看手动计算Precision和Recall?
precision2 = P(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
recall2 = R(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])



