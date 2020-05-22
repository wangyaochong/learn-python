from zipfile import ZipFile
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np

file = ZipFile('./digit recognizor.zip')
f = file.open('digit recognizor.csv')
df = pd.read_csv(f)
f.close()
file.close()

df.info()

# selector = VarianceThreshold() # 可以根据方差过滤，可以取方差前100的特征（分布最广的特征）
# selector = VarianceThreshold(np.median(df.var().values))  # 过滤一半特征

selector = VarianceThreshold(0.8 * (1 - 0.8))  # 过滤特征列80%数据都是一样的特征
result = selector.fit_transform(df)
print(df.shape, result.shape)

# 为什么需要过滤特征？
# 1、提高运行效率
# 2、去除不重要的噪音特征，可能提高准确率
