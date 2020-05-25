import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as LogiR
import pandas as pd
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R
from sklearn.metrics import confusion_matrix as CM, recall_score as R
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC

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
prob = clf_proba.predict_proba(X)
clf_proba.predict_proba(X).shape
deci = clf_proba.decision_function(X)
clf_proba.decision_function(X).shape

prob = pd.DataFrame(prob)
prob.columns = ["0", "1"]

for i in range(prob.shape[0]):
    if prob.loc[i, "1"] > 0.5:
        prob.loc[i, "pred"] = 1
    else:
        prob.loc[i, "pred"] = 0
prob["y_true"] = y

cm = CM(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
cm[1, 0] / cm[1, :].sum()
cm[0, 0] / cm[0, :].sum()
recall = []
FPR = []
probrange = np.linspace(clf_proba.predict_proba(X)
                        [:, 1].min(), clf_proba.predict_proba(X)[:, 1].max(), num=50, endpoint=False)
for i in probrange:
    y_predict = []
    for j in range(X.shape[0]):
        if clf_proba.predict_proba(X)[j, 1] > i:
            y_predict.append(1)
        else:
            y_predict.append(0)
    cm = CM(y, y_predict, labels=[1, 0])
    recall.append(cm[0, 0] / cm[0, :].sum())
    FPR.append(cm[1, 0] / cm[1, :].sum())
recall.sort()
FPR.sort()
plt.plot(FPR, recall, c="red")
plt.plot(probrange + 0.05, probrange + 0.05, c="black", linestyle="--")
plt.show()

FPR, recall_temp, thresholds = roc_curve(y, clf_proba.decision_function(X), pos_label=1)
area = AUC(y, clf_proba.decision_function(X))
plt.figure()
plt.plot(FPR, recall_temp, color='red', label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

maxindex = (recall_temp - FPR).tolist().index(max(recall_temp - FPR))
thresholds[maxindex]
# 我们可以在图像上来看看这个点在哪里
plt.scatter(FPR[maxindex], recall[maxindex], c="black", s=30)

# 把上述代码放入这段代码中：
plt.figure()
plt.plot(FPR, recall_temp, color='red', label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.scatter(FPR[maxindex], recall_temp[maxindex], c="black", s=30)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
