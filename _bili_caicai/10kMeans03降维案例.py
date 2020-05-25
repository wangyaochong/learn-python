# https://www.bilibili.com/video/BV1P7411P78r?p=106
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import pandas as pd

china = load_sample_image("china.jpg")
print("图片维度", china.shape)
china = np.array(china, dtype=np.float64) / china.max()
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))
china = np.array(china, dtype=np.float64) / china.max()
w, h, d = original_shape = tuple(china.shape)
image_array = np.reshape(china, (w * h, d))

n_clusters = 64  # 设置为64种颜色
image_array_sample = shuffle(image_array, random_state=0)[:5000]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)
labels = kmeans.predict(image_array)
image_kmeans = image_array.copy()
for i in range(w * h):
    image_kmeans[i] = kmeans.cluster_centers_[labels[i]]
image_kmeans = image_kmeans.reshape(w, h, d)

centroid_random = shuffle(image_array, random_state=0)[:n_clusters]
labels_random = pairwise_distances_argmin(centroid_random, image_array, axis=0)
image_random = image_array.copy()
for i in range(w * h):
    image_random[i] = centroid_random[labels_random[i]]
image_random = image_random.reshape(w, h, d)

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)
plt.show()
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Quantized image ({} colors, K-Means)'.format(n_clusters))
plt.imshow(image_kmeans)
plt.show()
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.title('Quantized image ({} colors, Random)'.format(n_clusters))
plt.imshow(image_random)
plt.show()
