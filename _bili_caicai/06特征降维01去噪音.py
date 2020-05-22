import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def plot_digits(dig):
    fig, axes = plt.subplots(4, 10, figsize=(8, 8)
                             , subplot_kw={"xticks": [], "yticks": []}  # 不显示坐标轴
                             )
    for index, ax in enumerate([*axes.flat]):
        ax.imshow(dig[index, :].reshape(8, 8), cmap='binary')
    plt.show()


digits = load_digits()
plot_digits(digits.data)

# 手动加噪音
rng = np.random.RandomState(0)

noisy = rng.normal(digits.data, 2)
plot_digits(noisy)

# 先降维
pca = PCA(0.9, svd_solver='full')
noisy_result = pca.fit_transform(noisy)

# 再逆转
plot_digits(pca.inverse_transform(noisy_result))
