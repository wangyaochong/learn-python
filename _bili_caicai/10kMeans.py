from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
import datetime
from time import time

# 自己创建数据集
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
fig, ax1 = plt.subplots(1)  # 生成一个子图
ax1.scatter(X[:, 0], X[:, 1]
            # 点的形状
            , marker='o'
            # 点的大小
            , s=8
            )
plt.show()

color = ["red", "pink", "orange", "gray"]
fig, ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y == i, 0], X[y == i, 1], marker='o'  # 点的形状 ,s=8 #点的大小 ,c=color[i]
                )
plt.show()

n_clusters = 3
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
y_pred = cluster.labels_
pre = cluster.fit_predict(X)
cluster_smallsub = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:200])
y_pred_ = cluster_smallsub.predict(X)
centroid = cluster.cluster_centers_
inertia = cluster.inertia_
print("总距离", inertia)
color = ["red", "pink", "orange", "gray"]
fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[y_pred == i, 0], X[y_pred == i, 1], marker='o', s=8, c=color[i]
                )
ax1.scatter(centroid[:, 0], centroid[:, 1], marker="x", s=15, c="black")
plt.show()

# 轮廓系数，评估模型的分类结果，越接近1越好,可以看到4是最好的聚类分组
n_clusters = 4
cluster_4 = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
n_clusters = 5
cluster_5 = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

score3 = silhouette_score(X, y_pred)
score4 = silhouette_score(X, cluster_4.labels_)
score5 = silhouette_score(X, cluster_5.labels_)
silhouette_samples(X, y_pred)

score_cal = calinski_harabasz_score(X, y_pred)  # 这个运行速度更快很多

time = datetime.datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S")
