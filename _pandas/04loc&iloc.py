import pandas as pd
import numpy as np

data = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))
print(type(data['W']), data['W'])
print(type(data[['W']]), data[['W']])  # 可以通过这种方式取一列DataFrame
copy = data.copy()
copy.loc["a"] = 1
print("赋值", copy)
print(data)
print("a行Z列", data.loc["a", "Z"])
print("a行", data.loc["a"])
print("a行c行", data.loc[["a", "c"]])
print("Y列", data.loc[:, "Y"])
print("X列Z列", data.loc[:, ["X", "Z"]])
print("取块", data.loc[["a", "c"], ["X", "Y"]])

print("---使用iloc")
print("---取第0行", data.iloc[0])
print("---取第1行", data.iloc[1])
print("---取第1列", data.iloc[:, 1])
print("---取1、2行", data.iloc[[1, 2]])
print("---取1、2列", data.iloc[:, [1, 2]])
print("---取块", data.iloc[[1, 2], [1, 2]])
