import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./911.csv", encoding="gbk")
data2 = data.copy()
for i in range(data.shape[0]):
    tmp = data.iloc[i, 4].split(":")[0]
    data.iloc[i, 4] = tmp
print(data['title'])
print(data.info())
print(data.head(5))

print("方法一", data.groupby(by='title').count())

# 方法二
temp = data2['title'].str.split(":").tolist()
data2['category'] = pd.DataFrame(np.array([i[0] for i in temp]).reshape(data2.shape[0], 1))
print("方法二", data2.groupby(by='category').count())

temp_date = data2['timeStamp'].str.split("/").tolist()
data2['dateStr'] = pd.DataFrame(np.array([i[0] + i[1] for i in temp_date]).reshape(data2.shape[0], 1))
print("按照月份分布", data2.groupby(by=['dateStr', 'category']).count())
