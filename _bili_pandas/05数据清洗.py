import pandas as pd
import numpy as np

df = pd.read_excel('./student_excel.xlsx', skiprows=2)
print(df)

df.dropna(axis='columns', how='all', inplace=True)  # axis可以使用字符串
df.dropna(axis='index', how='all', inplace=True)  # axis可以使用字符串

df.fillna({'分数': 0})
df['分数'] = df['分数'].fillna(0)
df['姓名'] = df['姓名'].fillna(method='ffill')

print(df)
