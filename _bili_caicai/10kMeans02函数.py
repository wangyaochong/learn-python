from sklearn.datasets import make_blobs
from sklearn.cluster import k_means

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

# k_means函数会一次性返回质心，每个样本对应的分类，inertia以及最佳迭代次数
print(k_means(X, 4, return_n_iter=True))
