import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from math import radians, sin, cos, acos
import sys
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from time import time
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.linear_model import LogisticRegression as LR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as ROC
from sklearn.metrics import accuracy_score as AC

# 原数据集 https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

weather = pd.read_csv("./weatherAUS5000.csv", index_col=0)
weather.head()

X = weather.iloc[:, :-1]
Y = weather.iloc[:, -1]
# 探索数据
shape = X.shape
X.info()
mean = X.isnull().mean()
unique = np.unique(Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=420)
# 恢复索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

# 是否有样本不平衡问题？（轻微）
train_count = Ytrain.value_counts()
test_count = Ytest.value_counts()

encorder = LabelEncoder().fit(Ytrain)
Ytrain = pd.DataFrame(encorder.transform(Ytrain))
Ytest = pd.DataFrame(encorder.transform(Ytest))

# 描述性统计
x_train_desc = Xtrain.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T
x_test_desc = Xtest.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T

Xtrainc = Xtrain.copy()
Xtrainc.sort_values(by="Location")
Xtrain.iloc[:, 0].value_counts()
Xtrain.iloc[:, 0].value_counts().count()
# Xtrain = Xtrain.drop(["Date"], axis=1)
# Xtest = Xtest.drop(["Date"], axis=1)
Xtrain["Rainfall"].head(20)
Xtrain.loc[Xtrain["Rainfall"] >= 1, "RainToday"] = "Yes"
Xtrain.loc[Xtrain["Rainfall"] < 1, "RainToday"] = "No"
Xtrain.loc[Xtrain["Rainfall"] == np.nan, "RainToday"] = np.nan
Xtest.loc[Xtest["Rainfall"] >= 1, "RainToday"] = "Yes"
Xtest.loc[Xtest["Rainfall"] < 1, "RainToday"] = "No"
Xtest.loc[Xtest["Rainfall"] == np.nan, "RainToday"] = np.nan
Xtrain.head()
Xtest.head()

int(Xtrain.loc[0, "Date"].split("-")[1])  # 提取出月份
Xtrain["Date"] = Xtrain["Date"].apply(lambda x: int(x.split("-")[1]))
# 替换完毕后，我们需要修改列的名称
# rename是比较少有的，可以用来修改单个列名的函数
# 我们通常都直接使用 df.columns = 某个列表 这样的形式来一次修改所有的列名
# 但rename允许我们只修改某个单独的列
Xtrain = Xtrain.rename(columns={"Date": "Month"})
Xtrain.head()
Xtest["Date"] = Xtest["Date"].apply(lambda x: int(x.split("-")[1]))
Xtest = Xtest.rename(columns={"Date": "Month"})
Xtest.head()

cityll = pd.read_csv("./cityll.csv", index_col=0)
city_climate = pd.read_csv("./Cityclimate.csv")
cityll.head()
city_climate.head()

cityll["Latitudenum"] = cityll["Latitude"].apply(lambda x: float(x[:-1]))
cityll["Longitudenum"] = cityll["Longitude"].apply(lambda x: float(x[:-1]))
citylld = cityll.iloc[:, [0, 5, 6]]
citylld["climate"] = city_climate.iloc[:, -1]

samplecity = pd.read_csv("./samplecity.csv", index_col=0)
samplecity["Latitudenum"] = samplecity["Latitude"].apply(lambda x: float(x[:-1]))
samplecity["Longitudenum"] = samplecity["Longitude"].apply(lambda x: float(x[:-1]))
samplecityd = samplecity.iloc[:, [0, 5, 6]]
samplecityd_head = samplecityd.head()

citylld.loc[:, "slat"] = citylld.iloc[:, 1].apply(lambda x: radians(x))
citylld.loc[:, "slon"] = citylld.iloc[:, 2].apply(lambda x: radians(x))
samplecityd.loc[:, "elat"] = samplecityd.iloc[:, 1].apply(lambda x: radians(x))
samplecityd.loc[:, "elon"] = samplecityd.iloc[:, 2].apply(lambda x: radians(x))

for i in range(samplecityd.shape[0]):
    slat = citylld.loc[:, "slat"]
    slon = citylld.loc[:, "slon"]
    elat = samplecityd.loc[i, "elat"]
    elon = samplecityd.loc[i, "elon"]
    dist = 6371.01 * np.arccos(np.sin(slat) * np.sin(elat) + np.cos(slat) * np.cos(elat) * np.cos(slon.values - elon))
    city_index = np.argsort(dist)[0]
    # 每次计算后，取距离最近的城市，然后将最近的城市和城市对应的气候都匹配到samplecityd中
    samplecityd.loc[i, "closest_city"] = citylld.loc[city_index, "City"]
    samplecityd.loc[i, "climate"] = citylld.loc[city_index, "climate"]

locafinal = samplecityd.iloc[:, [0, -1]]
locafinal.columns = ["Location", "Climate"]
locafinal = locafinal.set_index(keys="Location")

samplecityd_head2 = samplecityd.head()
locafinal_head = locafinal.head()

Xtrain["Location"] = Xtrain["Location"].map(locafinal.iloc[:, 0]).apply(lambda x: re.sub(",", "", x.strip()))
Xtest["Location"] = Xtest["Location"].map(locafinal.iloc[:, 0]).apply(lambda x: re.sub(",", "", x.strip()))

# 修改特征内容之后，我们使用新列名“Climate”来替换之前的列名“Location”
# 注意这个命令一旦执行之后，就再没有列"Location"了，使用索引时要特别注意
Xtrain = Xtrain.rename(columns={"Location": "Climate"})
Xtest = Xtest.rename(columns={"Location": "Climate"})

cloud = ["Cloud9am", "Cloud3pm"]
cate = Xtrain.columns[Xtrain.dtypes == "object"].tolist()
cate = cate + cloud
si = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
si.fit(Xtrain.loc[:, cate])
Xtrain.loc[:, cate] = si.transform(Xtrain.loc[:, cate])
Xtest.loc[:, cate] = si.transform(Xtest.loc[:, cate])

oe = OrdinalEncoder()
oe = oe.fit(Xtrain.loc[:, cate])
Xtrain.loc[:, cate] = oe.transform(Xtrain.loc[:, cate])
Xtest.loc[:, cate] = oe.transform(Xtest.loc[:, cate])

col = Xtrain.columns.tolist()
for i in cate:
    col.remove(i)
impmean = SimpleImputer(missing_values=np.nan, strategy="mean")
impmean = impmean.fit(Xtrain.loc[:, col])
Xtrain.loc[:, col] = impmean.transform(Xtrain.loc[:, col])
Xtest.loc[:, col] = impmean.transform(Xtest.loc[:, col])

# 进行无量纲化需要跳过分类型变量
col.remove("Month")
ss = StandardScaler()
ss = ss.fit(Xtrain.loc[:, col])
Xtrain.loc[:, col] = ss.transform(Xtrain.loc[:, col])
Xtest.loc[:, col] = ss.transform(Xtest.loc[:, col])

Ytrain = Ytrain.iloc[:, 0].ravel()
Ytest = Ytest.iloc[:, 0].ravel()
times = time()

for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel, gamma="auto", degree=1,
              cache_size=5000).fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest, Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

print("使用class_weight=balanced,提升recall", "*" * 100)
times = time()
for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel, gamma="auto", degree=1, cache_size=5000,
              class_weight="balanced").fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest, Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

print("使用class_weight=1: 10,提升recall", "*" * 100)
times = time()
clf = SVC(kernel="linear", gamma="auto", cache_size=5000,
          class_weight={1: 10}  # 注意，这里写的其实是，类别1：10，隐藏了类别0：1这个比例
          ).fit(Xtrain, Ytrain)
result = clf.predict(Xtest)
score = clf.score(Xtest, Ytest)
recall = recall_score(Ytest, result)
auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
print("testing accuracy %f, recall is %f', auc is %f" % (score, recall, auc))
print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

clf = SVC(kernel="linear", gamma="auto", cache_size=5000).fit(Xtrain, Ytrain)
result = clf.predict(Xtest)
cm = CM(Ytest, result, labels=(1, 0))
# 几乎所有的0都被判断正确了，还有不少1也被判断正确了
specificity = cm[1, 1] / cm[1, :].sum()

print("配置很小的权重，提高准确度accuracy", "*" * 100)
irange = np.linspace(0.01, 0.05, 7)
for i in irange:
    times = time()
    clf = SVC(kernel="linear", gamma="auto", cache_size=5000, class_weight={1: 1 + i}).fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest, Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
    print("under ratio 1:%f testing accuracy %f, recall is %f', auc is %f" %
          (1 + i, score, recall, auc))
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

print("试试逻辑回归", "*" * 100)
logclf = LR(solver="liblinear").fit(Xtrain, Ytrain)
logclf.score(Xtest, Ytest)
C_range = np.linspace(3, 5, 7)
for C in C_range:
    logclf = LR(solver="liblinear", C=C).fit(Xtrain, Ytrain)
    print(C, logclf.score(Xtest, Ytest))

# print("比较耗时", "*" * 100)
# C_range = np.linspace(0.01, 5, 10)
# recallall = []
# aucall = []
# scoreall = []
# for C in C_range:
#     times = time()
#     clf = SVC(kernel="linear", C=C, cache_size=5000, class_weight="balanced").fit(Xtrain, Ytrain)
#     result = clf.predict(Xtest)
#     score = clf.score(Xtest, Ytest)
#     recall = recall_score(Ytest, result)
#     auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
#     recallall.append(recall)
#     aucall.append(auc)
#     scoreall.append(score)
#     print("under C %f, testing accuracy is %f,recall is %f', auc is %f" %
#           (C, score, recall, auc))
#     print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))
# print(max(aucall), C_range[aucall.index(max(aucall))])
# plt.figure()
# plt.plot(C_range, recallall, c="red", label="recall")
# plt.plot(C_range, aucall, c="black", label="auc")
# plt.plot(C_range, scoreall, c="orange", label="accuracy")
# plt.legend()
# plt.show()

times = time()
clf = SVC(kernel="linear", C=3.1663157894736838, cache_size=5000, class_weight="balanced").fit(Xtrain, Ytrain)
result = clf.predict(Xtest)
score = clf.score(Xtest, Ytest)
recall = recall_score(Ytest, result)
auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
print("testing accuracy %f,recall is %f', auc is %f" % (score, recall, auc))
print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

FPR, Recall, thresholds = ROC(Ytest, clf.decision_function(Xtest), pos_label=1)
area = roc_auc_score(Ytest, clf.decision_function(Xtest))
plt.figure()
plt.plot(FPR, Recall, color='red', label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

maxindex = (Recall - FPR).tolist().index(max(Recall - FPR))
thresholds[maxindex]

times = time()
clf = SVC(kernel="linear", C=3.1663157894736838, cache_size=5000, class_weight="balanced").fit(Xtrain, Ytrain)
prob = pd.DataFrame(clf.decision_function(Xtest))
prob.loc[prob.iloc[:, 0] >= thresholds[maxindex], "y_pred"] = 1
prob.loc[prob.iloc[:, 0] < thresholds[maxindex], "y_pred"] = 0
prob.loc[:, "y_pred"].isnull().sum()
score = AC(Ytest, prob.loc[:, "y_pred"].values)
recall = recall_score(Ytest, prob.loc[:, "y_pred"])
print("testing accuracy %f,recall is %f" % (score, recall))
print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))
