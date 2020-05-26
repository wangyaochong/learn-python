import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import brier_score_loss
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd

# https://www.bilibili.com/video/BV1P7411P78r?p=209
digits = load_digits()
X, y = digits.data, digits.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

gnb = GaussianNB().fit(Xtrain, Ytrain)
acc_score = gnb.score(Xtest, Ytest)
Y_pred = gnb.predict(Xtest)
prob = gnb.predict_proba(Xtest)
cm = CM(Ytest, Y_pred)

h = .02

names = ["Multinomial", "Gaussian", "Bernoulli", "Complement"]
classifiers = [MultinomialNB(), GaussianNB(), BernoulliNB(), ComplementNB()]
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
datasets = [make_moons(noise=0.3, random_state=0), make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

figure = plt.figure(figsize=(6, 9))
i = 1

for ds_index, ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                        random_state=42)
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    array1, array2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), 2, i)
    if ds_index == 0:
        ax.set_title("Input data")

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    ax = plt.subplot(len(datasets), 2, i)

    clf = GaussianNB().fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    Z = clf.predict_proba(np.c_[array1.ravel(), array2.ravel()])[:, 1]
    Z = Z.reshape(array1.shape)
    ax.contourf(array1, array2, Z, cmap=cm, alpha=.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_index == 0:
        ax.set_title("Gaussian Bayes")
    ax.text(array1.max() - .3, array2.min() + .3, ('{:.1f}%'.format(score * 100)), size=15, horizontalalignment='right')
    i += 1
plt.tight_layout()
plt.show()

# brier_score = brier_score_loss(Ytest, prob[:, 1], pos_label=1)


logi = LR(C=1., solver='lbfgs', max_iter=3000, multi_class="auto").fit(Xtrain, Ytrain)
svc = SVC(kernel="linear", gamma=1).fit(Xtrain, Ytrain)
Ytest_copy = Ytest.copy()
for i, v in enumerate(Ytest):
    if Ytest[i] != 1:
        Ytest_copy[i] = 0
    else:
        Ytest_copy[i] = 1
b_score_logi = brier_score_loss(Ytest_copy, logi.predict_proba(Xtest)[:, 1], pos_label=1)

svc_prob = (svc.decision_function(Xtest) - svc.decision_function(Xtest).min()) / (
        svc.decision_function(Xtest).max() - svc.decision_function(Xtest).min())

b_score_svc = brier_score_loss(Ytest_copy, svc_prob[:, 1], pos_label=1)

name = ["Bayes", "Logistic", "SVC"]
color = ["red", "black", "orange"]

df = pd.DataFrame(index=range(10), columns=name)

for i in range(10):
    Ytest_copy = Ytest.copy()
    for ind, v in enumerate(Ytest):
        if Ytest[ind] != i:
            Ytest_copy[ind] = 0
        else:
            Ytest_copy[ind] = 1
    df.loc[i, name[0]] = brier_score_loss(Ytest_copy, prob[:, i], pos_label=1)
    df.loc[i, name[1]] = brier_score_loss(Ytest_copy, logi.predict_proba(Xtest)[:, i], pos_label=1)
    df.loc[i, name[2]] = brier_score_loss(Ytest_copy, svc_prob[:, i], pos_label=1)
for i in range(df.shape[1]):
    plt.plot(range(10), df.iloc[:, i], c=color[i])

plt.legend()
plt.show()
