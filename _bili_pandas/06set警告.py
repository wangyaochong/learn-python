import pandas as pd
import numpy as np

# 只要操作一步到位就不会报settingWithCopyWarning

df = pd.read_csv("./beijing_tianqi_2018.csv")

df.dropna(inplace=True)

df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)
print(df)

condition = df['ymd'].str.startswith('2018-03')
# df[condition]['wencha'] = df['bWendu'] - df['yWendu']  # 这里会有异常
# 修改成一步到位
df.loc[condition, 'wencha'] = df['bWendu'] - df['yWendu']
print(df[condition])

# 或者使用copy后修改
df2 = df[condition].copy()
df2['wencha'] = df['bWendu'] - df['yWendu']
df[condition] = df2
print(df[condition])
