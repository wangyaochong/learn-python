from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import BernoulliNB

class_1 = 500
class_2 = 500  # 两个类别分别设定500个样本
centers = [[0.0, 0.0], [2.0, 2.0]]  # 设定两个类别的中心
clusters_std = [0.5, 0.5]  # 设定两个类别的方差
X, y = make_blobs(n_samples=[class_1, class_2], centers=centers, cluster_std=clusters_std, random_state=0,
                  shuffle=False)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

mms = MinMaxScaler().fit(Xtrain)
Xtrain_ = mms.transform(Xtrain)
Xtest_ = mms.transform(Xtest)

mnb = MultinomialNB().fit(Xtrain_, Ytrain)
mnb.class_log_prior_
np.unique(Ytrain)(Ytrain == 1).sum() / Ytrain.shape[0]
mnb.class_log_prior_.shape

mnb.predict(Xtest_)
mnb.predict_proba(Xtest_)
mnb.score(Xtest_, Ytest)
brier_score_loss(Ytest, mnb.predict_proba(Xtest_)[:, 1], pos_label=1)

kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(Xtrain)
Xtrain_ = kbs.transform(Xtrain)
Xtest_ = kbs.transform(Xtest)
mnb = MultinomialNB().fit(Xtrain_, Ytrain)
mnb.score(Xtest_, Ytest)
brier_score_loss(Ytest, mnb.predict_proba(Xtest_)[:, 1], pos_label=1)

mms = MinMaxScaler().fit(Xtrain)
Xtrain_ = mms.transform(Xtrain)
Xtest_ = mms.transform(Xtest)

bnl_ = BernoulliNB().fit(Xtrain_, Ytrain)
bnl_.score(Xtest_, Ytest)
brier_score_loss(Ytest, bnl_.predict_proba(Xtest_)[:, 1], pos_label=1)

bnl = BernoulliNB(binarize=0.5).fit(Xtrain_, Ytrain)
bnl.score(Xtest_, Ytest)
brier_score_loss(Ytest, bnl.predict_proba(Xtest_)[:, 1], pos_label=1)
