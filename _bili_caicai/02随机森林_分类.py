from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

wine = load_wine()

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
dtc.fit(x_train, y_train)
rfc.fit(x_train, y_train)
for i in rfc.estimators_:
    print(i)

score_d = dtc.score(x_test, y_test)
score_r = rfc.score(x_test, y_test)

dtc = DecisionTreeClassifier(random_state=0)
dtc_score = cross_val_score(dtc, wine.data, wine.target, cv=10)

rfc = RandomForestClassifier(random_state=0, n_estimators=25)
rfc_score = cross_val_score(rfc, wine.data, wine.target, cv=10)

plt.plot(range(1, 11), dtc_score, label="decisionTree")
plt.plot(range(1, 11), rfc_score, label="randomForest")
plt.legend()
plt.show()

# 查看n_estimators对准确率的影响
superpa = []
count = 20
for i in range(count):
    rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=8)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=5).mean()
    superpa.append(rfc_s)
print(max(superpa), superpa.index(max(superpa)))
plt.figure(figsize=[20, 5])
plt.plot(range(1, count + 1), superpa)
plt.show()

dtc_list = []
rfc_list = []
for i in range(10):
    dtc = DecisionTreeClassifier()
    dtc_score = cross_val_score(dtc, wine.data, wine.target, cv=10).mean()
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_score = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    dtc_list.append(dtc_score)
    rfc_list.append(rfc_score)

plt.plot(range(1, 11), dtc_list, label="decisionTree")
plt.plot(range(1, 11), rfc_list, label="randomForest")
plt.show()
