import pandas as pd

# pandas 是基于numpy的

# series是带标签(索引)的一维数组

data = pd.Series([1, 2, 3, 45, 6])
print(data, type(data))
print(data[data > 4])
data2 = pd.Series([1, 2, 34, 5, 6], index=list("abcde"))
print(data2)
print(data2.astype(float))

temp_dict = {"name": "小红", "age": 30, "phone": 10086}
data3 = pd.Series(temp_dict)
print(data3)

print("索引数据1", data3["name"])
print("索引数据2", data3[["name", "age"]])
print("索引数据3", data3[[0, 1]])
print("索引数据4", data3[[0]])
print("索引数据(位置)", data3[0])
print("索引数据(位置)2", data3[0:])
print(data3[data3['name'] == "小红"])

print("数据索引", type(data3.index))
for i in data3.index:
    print(i)
