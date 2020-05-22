from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

# scaler = MinMaxScaler()  # 默认是0-1的区间
mms = MinMaxScaler(feature_range=[5, 10])  # 默认是0-1的区间
mms.fit(data)
result = mms.transform(data)
result2 = mms.fit_transform(data)

origin = mms.inverse_transform(result)  # inverse_transform还原数据
origin2 = mms.inverse_transform(result2)

ss = StandardScaler()
result_ss = ss.fit_transform(data)
mean = ss.mean_
var = ss.var_

print("mean", result_ss.mean())
print("std", result_ss.std())
