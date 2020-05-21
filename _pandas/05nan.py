import pandas as pd
import numpy as np

data = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("wxyz"))
print(data)
data.iloc[[0, 1], [1, 2]] = np.nan
print(pd.isnull(data))
print(pd.notnull(data))

select_col = 'x'
print(data[select_col])
print(pd.notnull(data[select_col]))
print("选出{}列不是nan的行".format(select_col), data[pd.notnull(data[select_col])])

print("删除nan的行", data.dropna(axis=0))
print("删除nan的列", data.dropna(axis=1))
print("删除全部是nan的行", data.dropna(axis=0, how="all"))
