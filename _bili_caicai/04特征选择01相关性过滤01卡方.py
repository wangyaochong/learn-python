from zipfile import ZipFile
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  # chi2专门用于分类问题，不能计算负数
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

file = ZipFile('./digit recognizor.zip')
f = file.open('digit recognizor.csv')
df = pd.read_csv(f)
f.close()
file.close()

df.info()
x = df.iloc[:, 1:]
y = df.iloc[:, 0]
selector = VarianceThreshold(np.median(x.var().values))  # 先过滤一半特征
result = selector.fit_transform(x)
print(df.shape, result.shape)

ch, p = chi2(result, y)
k = ch.shape[0] - (p > 0.05).sum()  # 大于0.05意思是和结果不相关(或者取值0.01)
result2 = SelectKBest(chi2, k=k).fit_transform(result, y)  # 按照卡方值过滤
print(result2.shape)

# 画出特征数量和精确度的图像
score = []
r = range(350, 250, -10)
for i in r:
    result2 = SelectKBest(chi2, k=i).fit_transform(result, y)
    score.append(cross_val_score(RandomForestClassifier(n_estimators=10, random_state=0), result2, y, cv=5).mean())
plt.plot(r, score)
plt.show()  # 从曲线可以看出卡方过滤的特征都是有用的，不能过滤掉
