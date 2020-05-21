import pandas as pd

data = pd.read_csv("./dogNames2.csv")
# print(data)

data = data.sort_values(by="Count_AnimalName", ascending=False)
print(data.head(10))
print("前20行", data[:20])
print("取具体某一列", data["Row_Labels"], type(data["Row_Labels"]))

count = 1000
print("使用数量超过{}".format(count), data[data['Count_AnimalName'] > count])
print("大于800小于1000".format(count), data[(data['Count_AnimalName'] > 800) & (data['Count_AnimalName'] < 1000)])
print("大于1000或小于2".format(count), data[(data['Count_AnimalName'] < 2) | (data['Count_AnimalName'] > 1000)])
