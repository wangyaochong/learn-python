import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import confusion_matrix as cm, recall_score as recall, roc_auc_score as auc

class_1 = 500
class_2 = 50

centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]

X, y = make_blobs(n_samples=[class_1, class_2], centers=centers, cluster_std=clusters_std, random_state=0,
                  shuffle=False)
Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

clf = XGBC().fit(Xtrain, Ytrain)
ypred = clf.predict(Xtest)
clf.score(Xtest, Ytest)
cm(Ytest, ypred, labels=[1, 0])
recall(Ytest, ypred)
auc(Ytest, clf.predict_proba(Xtest)[:, 1])

clf_ = XGBC(scale_pos_weight=10).fit(Xtrain, Ytrain)
ypred_ = clf_.predict(Xtest)
clf_.score(Xtest, Ytest)
cm(Ytest, ypred_, labels=[1, 0])
recall(Ytest, ypred_)
auc(Ytest, clf_.predict_proba(Xtest)[:, 1])

for i in [1, 5, 10, 20, 30]:
    clf_ = XGBC(scale_pos_weight=i).fit(Xtrain, Ytrain)
    ypred_ = clf_.predict(Xtest)
    print(i)
    print("\tAccuracy:{}".format(clf_.score(Xtest, Ytest)))
    print("\tRecall:{}".format(recall(Ytest, ypred_)))
    print("\tAUC:{}".format(auc(Ytest, clf_.predict_proba(Xtest)[:, 1])))
dtrain = xgb.DMatrix(Xtrain, Ytrain)
dtest = xgb.DMatrix(Xtest, Ytest)
param = {'silent': True, 'objective': 'binary:logistic', "eta": 0.1, "scale_pos_weight": 1}
num_round = 100
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

ypred = preds.copy()
ypred[preds > 0.5] = 1
ypred[ypred != 1] = 0

scale_pos_weight = [1, 5, 10]
names = ["negative vs positive: 1", "negative vs positive: 5", "negative vs positive: 10"]

from sklearn.metrics import accuracy_score as accuracy, recall_score as recall, roc_auc_score as auc

for name, i in zip(names, scale_pos_weight):
    param = {'silent': True, 'objective': 'binary:logistic', "eta": 0.1, "scale_pos_weight": i}
    clf = xgb.train(param, dtrain, num_round)
    preds = clf.predict(dtest)
    ypred = preds.copy()
    ypred[preds > 0.5] = 1
    ypred[ypred != 1] = 0
    print(name)
    print("\tAccuracy:{}".format(accuracy(Ytest, ypred)))
    print("\tRecall:{}".format(recall(Ytest, ypred)))
    print("\tAUC:{}".format(auc(Ytest, preds)))

for name, i in zip(names, scale_pos_weight):
    for thres in [0.3, 0.5, 0.7, 0.9]:
        param = {'silent': True, 'objective': 'binary:logistic', "eta": 0.1, "scale_pos_weight": i}
        clf = xgb.train(param, dtrain, num_round)
        preds = clf.predict(dtest)
        ypred = preds.copy()
        ypred[preds > thres] = 1
        ypred[ypred != 1] = 0
        print("{},thresholds:{}".format(name, thres))
        print("\tAccuracy:{}".format(accuracy(Ytest, ypred)))
        print("\tRecall:{}".format(recall(Ytest, ypred)))
        print("\tAUC:{}".format(auc(Ytest, preds)))
