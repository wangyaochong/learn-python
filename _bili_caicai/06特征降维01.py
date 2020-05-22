import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# 特征降维优先于特征选择

iris = load_iris()
y = iris.target
x = iris.data

pca = PCA(n_components=2)
result = pca.fit_transform(x)
print(result.shape
      , pca.explained_variance_
      , pca.explained_variance_ratio_
      , pca.explained_variance_ratio_.sum()
      )

# 可视化数据分布
color = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(result[y == i, 0], result[y == i, 1], c=color[i], label=iris.target_names[i])

plt.legend()
plt.show()

pca_line = PCA().fit(x)
plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1, 2, 3, 4])  # 这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance")
plt.show()

pca_mle = PCA(n_components="mle")  # 还可以自己选
pca_mle.fit(x)
X_mle = pca_mle.transform(x)
print(pca_mle.explained_variance_ratio_.sum())

pca_f = PCA(n_components=0.97, svd_solver="full")  # 还可以指定信息比例
pca_f.fit(x)
x_f = pca_f.transform(x)
print(pca_f.explained_variance_ratio_.sum())
