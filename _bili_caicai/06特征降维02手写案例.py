import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

file = ZipFile('./digit recognizor.zip')
f = file.open('digit recognizor.csv')
df = pd.read_csv(f)
f.close()
file.close()

df.info()
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

pca_line = PCA().fit(x)
plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
plt.show()

score = []
r = range(20, 90, 10)
for i in r:
    x_d = PCA(i).fit_transform(x)
    score.append(cross_val_score(RFC(n_estimators=10, random_state=0), x_d, y, cv=5).mean())
plt.plot(r, score)

# 这个地方的学习曲线有点奇怪，主要是由于zip文件的数据严重缺失
plt.show()
