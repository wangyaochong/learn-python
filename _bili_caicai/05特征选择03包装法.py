from sklearn.ensemble import RandomForestClassifier as RFC
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE

# 包装法的时间成本位于过滤法和嵌入法之间


file = ZipFile('./digit recognizor.zip')
f = file.open('digit recognizor.csv')
df = pd.read_csv(f)
f.close()
file.close()

df.info()
x = df.iloc[:, 1:]
y = df.iloc[:, 0]
rfc = RFC(n_estimators=10, random_state=0)

selector = RFE(rfc, n_features_to_select=340, step=50).fit(x, y)
print(selector.support_.sum(), selector.ranking_)

x_result = selector.transform(x)
print(x_result.shape[0])
print("score", cross_val_score(rfc, x_result, y, cv=5).mean())

# 这个也可以使用学习曲线，遍历n_features_to_select
