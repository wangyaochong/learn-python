import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification as mc
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.calibration import calibration_curve

X, y = mc(n_samples=100000,
          n_features=20  # 总共20个特征
          , n_classes=2  # 标签为2分类
          , n_informative=2  # 其中两个代表较多信息
          , n_redundant=10  # 10个都是冗余特征
          , random_state=42)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.99, random_state=42)
gnb = GaussianNB()
gnb.fit(Xtrain, Ytrain)
y_pred = gnb.predict(Xtest)
prob_pos = gnb.predict_proba(Xtest)[:, 1]

clf_score = gnb.score(Xtest, Ytest)

df = pd.DataFrame({"ytrue": Ytest[:500], "probability": prob_pos[:500]})

df = df.sort_values(by="probability")
df.index = range(df.shape[0])

trueproba, predproba = calibration_curve(Ytest, prob_pos, n_bins=10  # 输入希望分箱的个数
                                         )


def plot_calib(models, name, Xtrain, Xtest, Ytrain, Ytest, n_bins=10):
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name_ in zip(models, name):
        clf.fit(Xtrain, Ytrain)
        y_pred = clf.predict(Xtest)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(Xtest)[:, 1]
        else:
            prob_pos = clf.decision_function(Xtest)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        clf_score = brier_score_loss(Ytest, prob_pos, pos_label=y.max())
        trueproba, predproba = calibration_curve(Ytest, prob_pos, n_bins=n_bins)
        ax1.plot(predproba, trueproba, "s-", label="%s (%1.3f)" % (name_, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=n_bins, label=name_, histtype="step", lw=2)

    ax2.set_ylabel("Distribution of probability")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_xlim([-0.05, 1.05])
    ax2.legend(loc=9)
    ax2.set_title("Distribution of probablity")
    ax1.set_ylabel("True probability for class 1")
    ax1.set_xlabel("Mean predcited probability")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.set_title('Calibration plots(reliability curve)')
    plt.show()


from sklearn.calibration import CalibratedClassifierCV

name = ["GaussianBayes", "Logistic", "Bayes+isotonic", "Bayes+sigmoid"]
gnb = GaussianNB()

models = [gnb, LR(C=1., solver='lbfgs', max_iter=3000, multi_class="auto")
          # 定义两种校准方式
    , CalibratedClassifierCV(gnb, cv=2, method='isotonic'), CalibratedClassifierCV(gnb, cv=2, method='sigmoid')]
plot_calib(models, name, Xtrain, Xtest, Ytrain, Ytest)

gnb = GaussianNB().fit(Xtrain, Ytrain)
score1 = gnb.score(Xtest, Ytest)
brier_score_loss(Ytest, gnb.predict_proba(Xtest)[:, 1], pos_label=1)
gnbisotonic = CalibratedClassifierCV(gnb, cv=2, method='isotonic').fit(Xtrain, Ytrain)
score2 = gnbisotonic.score(Xtest, Ytest)
brier_score_loss(Ytest, gnbisotonic.predict_proba(Xtest)[:, 1], pos_label=1)

###############################################################
name_svc = ["SVC", "Logistic", "SVC+isotonic", "SVC+sigmoid"]
svc = SVC(kernel="linear", gamma=1)
models_svc = [svc, LR(C=1., solver='lbfgs', max_iter=3000, multi_class="auto")
              # 依然定义两种校准方式
    , CalibratedClassifierCV(svc, cv=2, method='isotonic'), CalibratedClassifierCV(svc, cv=2, method='sigmoid')]
plot_calib(models_svc, name_svc, Xtrain, Xtest, Ytrain, Ytest)
#######################################################################
name_svc = ["SVC", "SVC+isotonic", "SVC+sigmoid"]
svc = SVC(kernel="linear", gamma=1)
models_svc = [svc, CalibratedClassifierCV(svc, cv=2, method='isotonic'),
              CalibratedClassifierCV(svc, cv=2, method='sigmoid')]

for clf, name in zip(models_svc, name_svc):
    clf.fit(Xtrain, Ytrain)
    y_pred = clf.predict(Xtest)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(Xtest)[:, 1]
    else:
        prob_pos = clf.decision_function(Xtest)
    prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    clf_score = brier_score_loss(Ytest, prob_pos, pos_label=y.max())
    score = clf.score(Xtest, Ytest)
    print("{}:".format(name))
    print("\tBrier:{:.4f}".format(clf_score))
    print("\tAccuracy:{:.4f}".format(score))
