import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./beijing_tianqi_2018.csv")
df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)

df.set_index(pd.to_datetime(df['ymd']), inplace=True)  # 转换成日期索引

print(df.index)

print(df.loc['2018-01-05'])
print(df.loc['2018-01-05':'2018-01-10'])
print(df.loc['2018-03'].head())
print(df.loc['2018-03':'2018-05'].head())

# 统计每周温度变化数据
print(df.groupby(df.index.week)['bWendu'].max().head())
df.groupby(df.index.week)['bWendu'].max().plot()
plt.show()

# 统计每月温度变化数据
print(df.groupby(df.index.month)['bWendu'].max().head())
df.groupby(df.index.month)['bWendu'].max().plot()
plt.show()
