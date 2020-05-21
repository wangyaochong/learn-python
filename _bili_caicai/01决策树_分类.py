from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
import graphviz
import matplotlib.pyplot as plt
import matplotlib

font = {"weight": "bold", "size": "20", "family": "KaiTi"}
matplotlib.rc("font", **font)

wine = load_wine()
print(dir(wine))
data = pd.concat(
    [pd.DataFrame(wine.data, columns=wine.feature_names), pd.DataFrame(wine.target, columns=['type'])], axis=1)
print(data)

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=0)

print(x_train.shape, x_test.shape)

clf = tree.DecisionTreeClassifier(criterion="entropy"
                                  , random_state=0
                                  , splitter="best"
                                  , max_depth=3
                                  , min_samples_leaf=5
                                  , min_samples_split=5
                                  , min_impurity_decrease=0.01
                                  # , min_weight_fraction_leaf=5
                                  # ,class_weight=
                                  )
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
feature_name = ['酒精', '苹果酸', "灰", "灰的碱性", "镁", "总酚", "类黄酮", "非黄烷酚类", "花青素", "颜色强度", "色调", "稀释葡萄酒", "脯氨酸"]
print("评分和特征重要性", score, *zip(feature_name, clf.feature_importances_))
plt.figure(figsize=(10, 10), dpi=100)
tree.plot_tree(clf, feature_names=feature_name)
plt.show()

# g_data = tree.export_graphviz(clf, feature_names=feature_name, class_names=list('ABC'), filled=True, rounded=True)
# graph = graphviz.Source(g_data)
# print(graph)


# 找到最佳参数
test_result = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i + 1, criterion="entropy", random_state=30, splitter="best")
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    test_result.append(score)
plt.plot(range(1, 11), test_result)
plt.show()
