import pandas as pd
import numpy as np

# 只要操作一步到位就不会报settingWithCopyWarning

df = pd.read_csv("./beijing_tianqi_2018.csv")
df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)

print('series 排序', df['aqi'].sort_values().head())
print('series 排序', df['aqi'].sort_values(ascending=False).head())

print(df.sort_values(by=['aqiLevel', 'bWendu'], ascending=[True, False]).head())  # 多列排序
