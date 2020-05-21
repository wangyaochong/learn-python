import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./directory.csv", encoding="gbk")

group = data.groupby(by="Country")
print(group)
print(group.count())
print("所有城市", group['Brand'].count())
print("一个城市", group['Brand'].count()['CA'])

# for i, j in group:
#     print(i, j)

group2=data.groupby(by=[data['Country'], data['State/Province']])
print("多个分组条件", group2)
print("多个分组条件", group2['Brand'].count())
print("多个分组条件", group2.count()['Brand'])
