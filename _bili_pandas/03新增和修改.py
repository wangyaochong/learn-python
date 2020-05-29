import pandas as pd
import numpy as np


def get_type(x):
    if x['bWendu'] > 33:
        return '高温'
    if x['yWendu'] < -10:
        return '低温'
    return '常温'


df = pd.read_csv("./beijing_tianqi_2018.csv")

df.set_index('ymd', inplace=True)
df.dropna(inplace=True)

df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)
df['wencha'] = df['bWendu'] - df['yWendu']  # 新增温差列
print(df.columns)
print(df.head())
df['type'] = df.apply(get_type, axis=1)
print(df.head())

df = df.assign(  # 新增两个华氏温度列
    yWendu_huashi=lambda x: x['yWendu'] * 9 / 5 + 32,
    bWendu_huashi=lambda x: x['bWendu'] * 9 / 5 + 32
)
print(df.head())

df['wencha_type'] = ''
df.loc[df['bWendu'] - df['yWendu'] > 10, 'wencha_type'] = '温差大'
df.loc[df['bWendu'] - df['yWendu'] <= 10, 'wencha_type'] = '温差小'
print(df.head())
