import pandas as pd
import numpy as np

# dataFrame 是带标签(索引)的二维数组
frame = pd.DataFrame(np.arange(12).reshape(3, 4))
print(frame)

frame2 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("ABCD"))
print(frame2)

d1 = {"name": ["小明", "小红"], "age": [18, 23], "phone": [10086, 10010]}
frame3 = pd.DataFrame(d1)
print(frame3)

d2 = [{"name": "小红", "age": 18, "phone": 10086}, {"name": "小白", "age": 21, "phone": 10010}]
frame4 = pd.DataFrame(d2)
print(frame4)

print("index", frame3.index)
print("columns", frame3.columns)
print("values", frame3.values)

print("head", frame3.head(1))
print("tail", frame3.tail(1))
print("info", frame3.info())
print("describe", frame3.describe())

# 测试get_dummies
d3 = {"color": ["red", "green", "blue"], "age": ['一', '二', '一']}
data3 = pd.DataFrame(d3)
print(data3)
print(pd.get_dummies(data3))
