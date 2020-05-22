import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.images.shape)

x = faces.data
fig, axes = plt.subplots(3, 8, figsize=(8, 8)
                         , subplot_kw={"xticks": [], "yticks": []}  # 不显示坐标轴
                         )

# for i in [*axes.flat]:
for index, ax in enumerate([*axes.flat]):
    ax.imshow(faces.images[index], cmap='gray')
plt.show()

pca = PCA(150)
pca.fit(x)
x_result = pca.transform(x)
print(pca.components_.shape)

fig, axes = plt.subplots(3, 8, figsize=(8, 8)
                         , subplot_kw={"xticks": [], "yticks": []}  # 不显示坐标轴
                         )
for index, ax in enumerate([*axes.flat]):
    ax.imshow(pca.components_[index, :].reshape(62, 47), cmap='gray')
plt.show()

# 可以把降维后的数据还原，但是由于有信息缺失，所以会更模糊一些
x_origin = pca.inverse_transform(x_result)
print("原始数据", x.shape,
      "逆转数据", x_origin.shape)

fig, axes = plt.subplots(2, 10
                         , subplot_kw={"xticks": [], "yticks": []}  # 不显示坐标轴
                         )
for i in range(10):
    axes[0, i].imshow(faces.images[i], cmap='gray')
    axes[1, i].imshow(x_origin[i, :].reshape(62, 47), cmap='gray')
plt.show()
