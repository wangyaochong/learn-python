from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import brier_score_loss as BS, recall_score, roc_auc_score as AUC
from sklearn.naive_bayes import ComplementNB
from time import time
import datetime

class_1 = 50000  # 多数类为50000个样本
class_2 = 500  # 少数类为500个样本
centers = [[0.0, 0.0], [5.0, 5.0]]  # 设定两个类别的中心
clusters_std = [3,
                1]  # 设定两个类别的方差
X, y = make_blobs(n_samples=[class_1, class_2], centers=centers, cluster_std=clusters_std, random_state=0,
                  shuffle=False)
name = ["Multinomial", "Gaussian", "Bernoulli"]
models = [MultinomialNB(), GaussianNB(), BernoulliNB()]

for name, clf in zip(name, models):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3,
                                                    random_state=420)
    if name != "Gaussian":
        kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(Xtrain)
        Xtrain = kbs.transform(Xtrain)
        Xtest = kbs.transform(Xtest)
    clf.fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)
    proba = clf.predict_proba(Xtest)[:, 1]
    score = clf.score(Xtest, Ytest)
    print(name)
    print("\tBrier:{:.3f}".format(BS(Ytest, proba, pos_label=1)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\tRecall:{:.3f}".format(recall_score(Ytest, y_pred)))
    print("\tAUC:{:.3f}".format(AUC(Ytest, proba)))
print("*" * 100)
name = ["Multinomial", "Gaussian", "Bernoulli", "Complement"]
models = [MultinomialNB(), GaussianNB(), BernoulliNB(), ComplementNB()]

for name, clf in zip(name, models):
    times = time()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
    if name != "Gaussian":
        kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(Xtrain)
        Xtrain = kbs.transform(Xtrain)
        Xtest = kbs.transform(Xtest)
    clf.fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)
    proba = clf.predict_proba(Xtest)[:, 1]
    score = clf.score(Xtest, Ytest)
    print(name)
    print("\tBrier:{:.3f}".format(BS(Ytest, proba, pos_label=1)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\tRecall:{:.3f}".format(recall_score(Ytest, y_pred)))
    print("\tAUC:{:.3f}".format(AUC(Ytest, proba)))
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))
