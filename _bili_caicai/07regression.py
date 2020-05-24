from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

data = load_breast_cancer()
x = data.data
y = data.target
print(x.shape)

lrl1 = LR(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)  # l1可能会让不重要的特征系数为0
lrl2 = LR(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)  # l2会让不重要的特征系数很小

lrl1.fit(x, y)
print(lrl1.coef_)

lrl2.fit(x, y)
print(lrl2.coef_)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

l1_result = []
l2_result = []

l1_result_t = []
l2_result_t = []
for i in np.linspace(0.05, 1, 19):
    lrl1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=1000)  # l1可能会让不重要的特征系数为0
    lrl2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=1000)  # l2会让不重要的特征系数很小

    lrl1.fit(x_train, y_train)
    l1_result.append(accuracy_score(lrl1.predict(x_train), y_train))
    l1_result_t.append(accuracy_score(lrl1.predict(x_test), y_test))

    lrl2.fit(x_train, y_train)
    l2_result.append(accuracy_score(lrl2.predict(x_train), y_train))
    l2_result_t.append(accuracy_score(lrl2.predict(x_test), y_test))

graph = [l1_result, l2_result, l1_result_t, l2_result_t]
color = ['green', 'black', 'red', 'blue']
label = ['l1', 'l2', 'l1_test', 'l2_test']
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
plt.legend()
plt.show()  # 从图片来看，l2的效果更好

# 特征选择（回归最好不要使用特征降维，因为无法解释特征的具体含义）
lr = LR(solver='liblinear', C=0.8, random_state=0)
m = cross_val_score(lr, x, y, cv=10).mean()
x_result = SelectFromModel(lr,
                           norm_order=1  # 使用l1范式
                           ).fit_transform(x, y)
print(x_result.shape)

# 学习曲线选取最佳阈值
# threshold = np.linspace(0, abs(lr.fit(x, y).coef_).max(), 20)  # coef_相关系数
threshold = np.linspace(0, 0.02, 20)  # coef_相关系数
origin_score = []
test_score = []
for i in threshold:
    x_result = SelectFromModel(lr,
                               threshold=i,
                               norm_order=1  # 使用l1范式
                               ).fit_transform(x, y)
    origin_score.append(cross_val_score(lr, x, y, cv=5).mean())
    test_score.append(cross_val_score(lr, x_result, y, cv=5).mean())
    print(i, x_result.shape[1])

plt.plot(threshold, origin_score, label='origin')
plt.plot(threshold, test_score, label='selected')
plt.legend()
plt.show()

# 之前通过降参数不好用，所以直接调模型的参数，这里是调C
c = np.arange(0.01, 10.01, 0.5)
fresult = []
tresult = []
for i in c:
    lr = LR(solver='liblinear', C=i, random_state=0)
    fresult.append(cross_val_score(lr, x, y, cv=10).mean())
    x_result = SelectFromModel(lr, norm_order=1  # 使用l1范式
                               ).fit_transform(x, y)
    tresult.append(cross_val_score(lr, x_result, y, cv=10).mean())

print("最佳的C值", max(tresult), c[tresult.index(max(tresult))])
plt.plot(c, fresult, label='full')
plt.plot(c, tresult, label='feature selection')
plt.show()

# 更近一步
c = np.arange(6.5, 7.5, 0.1)
fresult = []
tresult = []
for i in c:
    lr = LR(solver='liblinear', C=i, random_state=0)
    fresult.append(cross_val_score(lr, x, y, cv=10).mean())
    x_result = SelectFromModel(lr, norm_order=1  # 使用l1范式
                               ).fit_transform(x, y)
    tresult.append(cross_val_score(lr, x_result, y, cv=10).mean())

print("最佳的C值", max(tresult), c[tresult.index(max(tresult))])
plt.plot(c, fresult, label='full')
plt.plot(c, tresult, label='feature selection')
plt.show()

# 根据图像选择最佳的C值
lr = LR(solver='liblinear', C=7.1999999999999975, random_state=0)
r1 = cross_val_score(lr, x, y, cv=10).mean()
x_result = SelectFromModel(lr, norm_order=1  # 使用l1范式
                           ).fit_transform(x, y)
r2 = cross_val_score(lr, x_result, y, cv=10).mean()
# 通过特征选择后准确率更高，并且特征数量更少
print("完整参数", r1, '特征个数', x.shape[1], "选择特征后", r2, '特征个数', x_result.shape[1])
