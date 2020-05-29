import pandas as pd
import numpy as np

# 只要操作一步到位就不会报settingWithCopyWarning

df = pd.read_csv("./beijing_tianqi_2018.csv")
df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)

print(df['ymd'].str.startswith('2018-03').head())

df['ymd'] = df['ymd'].str.replace("-", "").str.slice(0, 6)  # 可以链式调用
print(df.head())

# ######注意，pandas的str支持正则表达式#################

