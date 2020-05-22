from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

# 一般优先使用过滤法，嵌入法运行速度比较慢


file = ZipFile('./digit recognizor.zip')
f = file.open('digit recognizor.csv')
df = pd.read_csv(f)
f.close()
file.close()

df.info()
x = df.iloc[:, 1:]
y = df.iloc[:, 0]
rfc = RFC(n_estimators=10, random_state=0)

x_result = SelectFromModel(rfc, threshold=0.005).fit_transform(x, y)
print(x_result.shape)

# 学习曲线找到最佳的阈值
# threshold = np.linspace(0, rfc.fit(x, y).feature_importances_.max(), 10) #第一次可以直接跑，
threshold = np.linspace(0, 0.001, 10)  # 然后设置具体参数
score = []
for i in threshold:
    x_result = SelectFromModel(rfc, threshold=i).fit_transform(x, y)
    score.append(cross_val_score(rfc, x_result, y, cv=5).mean())

plt.plot(threshold, score)
plt.show()
