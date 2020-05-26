from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import brier_score_loss as BS, recall_score, roc_auc_score as AUC
from sklearn.naive_bayes import ComplementNB
from time import time
import datetime
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import brier_score_loss as BS
from sklearn.calibration import CalibratedClassifierCV

sample = ["Machine learning is fascinating, it is wonderful", "Machine learning is a sensational techonology",
          "Elsa is a popular character"]

vec = CountVectorizer()
X = vec.fit_transform(sample)

CVresult = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

vec = TFIDF()
X = vec.fit_transform(sample)
TFIDFresult = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
CVresult.sum(axis=0) / CVresult.sum(axis=0).sum()
TFIDFresult.sum(axis=0) / TFIDFresult.sum(axis=0).sum()

data = fetch_20newsgroups()
categories = [
    "sci.space"  # 科学技术 - 太空
    , "rec.sport.hockey"  # 运动 - 曲棍球
    , "talk.politics.guns"  # 政治 - 枪支问题
    , "talk.politics.mideast"]  # 政治 - 中东问题

train = fetch_20newsgroups(subset="train", categories=categories)
test = fetch_20newsgroups(subset="test", categories=categories)

Xtrain = train.data
Xtest = test.data
Ytrain = train.target
Ytest = test.target
tfidf = TFIDF().fit(Xtrain)
Xtrain_ = tfidf.transform(Xtrain)
Xtest_ = tfidf.transform(Xtest)
tosee = pd.DataFrame(Xtrain_.toarray(), columns=tfidf.get_feature_names())

name = ["Multinomial", "Complement", "Bournulli"]
models = [MultinomialNB(), ComplementNB(), BernoulliNB()]

for name, clf in zip(name, models):
    clf.fit(Xtrain_, Ytrain)
    y_pred = clf.predict(Xtest_)
    proba = clf.predict_proba(Xtest_)
    score = clf.score(Xtest_, Ytest)
    print(name)
    Bscore = []
    for i in range(len(np.unique(Ytrain))):
        bs = BS(Ytest, proba[:, i], pos_label=i)
        Bscore.append(bs)
        print("\tBrier under {}:{:.3f}".format(train.target_names[i], bs))
    print("\tAverage Brier:{:.3f}".format(np.mean(Bscore)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\n")

name = ["Multinomial", "Multinomial + Isotonic", "Multinomial + Sigmoid", "Complement", "Complement + Isotonic",
        "Complement + Sigmoid", "Bernoulli", "Bernoulli + Isotonic", "Bernoulli + Sigmoid"]
models = [MultinomialNB(), CalibratedClassifierCV(MultinomialNB(), cv=2, method='isotonic'),
          CalibratedClassifierCV(MultinomialNB(), cv=2, method='sigmoid'), ComplementNB(),
          CalibratedClassifierCV(ComplementNB(), cv=2, method='isotonic'),
          CalibratedClassifierCV(ComplementNB(), cv=2, method='sigmoid'), BernoulliNB(),
          CalibratedClassifierCV(BernoulliNB(), cv=2, method='isotonic'),
          CalibratedClassifierCV(BernoulliNB(), cv=2, method='sigmoid')
          ]

for name, clf in zip(name, models):
    clf.fit(Xtrain_, Ytrain)
    y_pred = clf.predict(Xtest_)
    proba = clf.predict_proba(Xtest_)
    score = clf.score(Xtest_, Ytest)
    print(name)
    Bscore = []
    for i in range(len(np.unique(Ytrain))):
        bs = BS(Ytest, proba[:, i], pos_label=i)
        Bscore.append(bs)
        print("\tBrier under {}:{:.3f}".format(train.target_names[i], bs))
    print("\tAverage Brier:{:.3f}".format(np.mean(Bscore)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\n")
