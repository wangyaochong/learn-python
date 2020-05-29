import pandas as pd
import numpy as np

df = pd.read_csv("./beijing_tianqi_2018.csv")

df.set_index('ymd', inplace=True)
df.dropna(inplace=True)

df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)

print(df.describe())
print(df['fengxiang'].unique())  # 查看唯一值
print(df['fengxiang'].value_counts())  # 降序排序

print(df.cov())  # 协方差
print(df.corr())  # 相关系数，正相关最强是=1，负相关最强时等于-1

print('单独查看两个列的相关系数', df['aqi'].corr(df['bWendu']))
