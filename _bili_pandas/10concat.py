import pandas as pd
import numpy as np

df1 = pd.DataFrame(
    {
        "A": ['A0', 'A1', 'A2', 'A3'],
        "B": ['B0', 'B1', 'B2', 'B3'],
        "C": ['C0', 'C1', 'C2', 'C3'],
        "D": ['D0', 'D1', 'D2', 'D3'],
        "E": ['E0', 'E1', 'E2', 'E3'],
    }
)

df2 = pd.DataFrame(
    {
        "A": ['A4', 'A5', 'A6', 'A7'],
        "B": ['B4', 'B5', 'B6', 'B7'],
        "C": ['C4', 'C5', 'C6', 'C7'],
        "D": ['D4', 'D5', 'D6', 'D7'],
        "F": ['F4', 'F5', 'F6', 'F7'],
    }
)

print(pd.concat([df1, df2], sort=False))  # 默认axis=0，按照行拼接
print(pd.concat([df1, df2], ignore_index=True, sort=False))
print(pd.concat([df1, df2], ignore_index=True, join='inner'))
print(pd.concat([df1, df2], sort=False, axis=1))  # 默认axis=0，按照行拼接

s1 = pd.Series(list(range(4)), name='F')
s2 = df1.apply(lambda x: x['A'] + "_G", axis=1)
print(pd.concat([df1, s1], axis=1))
print(pd.concat([s1, s2], axis=1))  # 也可以用于合并两个Series

df3 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df4 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))  # 合并行
print(df3.append(df4))
print(df3.append(df4,ignore_index=True))
