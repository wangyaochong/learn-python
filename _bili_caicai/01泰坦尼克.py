import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

data = pd.read_csv("./titanic/train.csv")

# 数据预处理
data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data = data.dropna()
labels = data['Embarked'].unique().tolist()
data['Embarked'] = data['Embarked'].apply(lambda x: labels.index(x))
# data['Sex'] = (data['Sex'] == 'male').astype(int)
data.loc[:, 'Sex'] = (data.loc[:, 'Sex'] == 'male').astype(int)

# 取出数据以及目标标签
x = data.iloc[:, data.columns != "Survived"]
# y = data.iloc[:, data.columns == "Survived"]
y = data[['Survived']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

for i in [x_train, y_train, x_test, y_test]:  # 恢复索引的位置
    i.index = range(i.shape[0])

clf = DecisionTreeClassifier(random_state=25)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)

clf = DecisionTreeClassifier(random_state=25)
cross_score = cross_val_score(clf, x, y, cv=10).mean()

# 使用交叉验证
score_train_list = []
score_test_list = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25, max_depth=i + 1, criterion="entropy")
    clf.fit(x_train, y_train)
    score_train = clf.score(x_train, y_train)
    score_test = cross_val_score(clf, x, y, cv=10).mean()
    score_train_list.append(score_train)
    score_test_list.append(score_test)

plt.plot(range(1, 11), score_train_list, label="train")
plt.plot(range(1, 11), score_test_list, label="test")
plt.legend()
plt.show()

# 使用网格搜索
clf = DecisionTreeClassifier(random_state=25)
gini_thresholds = np.linspace(0, 0.5, 50)
# entropy_thresholds = np.linspace(0, 1, 50)  #这两种方式的取值不同
parameters = {"criterion": ("gini", "entropy"),
              "splitter": ("best", "random"),
              "max_depth": list(range(1, 5)),
              "min_samples_leaf": list(range(1, 50, 10)),
              "min_impurity_decrease": np.linspace(0, 0.5, 10)
              }
gs = GridSearchCV(clf, parameters, cv=10)
gs.fit(x_train, y_train)
best_param = gs.best_params_
best_score = gs.best_score_
