import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3, 4))

df.index = list('abc')
df.columns = list('ABCD')
print(df)

df = pd.read_csv("./beijing_tianqi_2018.csv")

df.set_index('ymd', inplace=True)
df.dropna(inplace=True)
print(df.columns)
df['bWendu'] = df['bWendu'].str.replace("℃", "").astype(int)
df['yWendu'] = df['yWendu'].str.replace("℃", "").astype(int)
# df['yWendu'] = df['yWendu'] \
#     .apply(lambda x: int(0) if x is None or x is np.nan else int(x.replace("℃", ""))).astype(int)

print(df)
print(df.loc['2018-01-03', 'bWendu'])
print(df.loc['2018-01-03', ['bWendu', 'yWendu']])
print(df.loc[['2018-01-03', '2018-01-04', ], ['bWendu', 'yWendu']])
print(df.loc['2018-01-03': '2018-01-05', ['bWendu', 'yWendu']])
print(df.loc['2018-01-03': '2018-01-05', 'bWendu':'tianqi'])  # 可以通过切片来查询数据

# 以下三种方式都可以
print(df[df['yWendu'] < -10])
print(df.loc[df['yWendu'] < -10])
print(df.loc[df['yWendu'] < -10, :])

print(df[(df['bWendu'] <= 30) & (df['yWendu'] >= 15) & (df['tianqi'] == '晴') & (df['aqiLevel']) == 1])

# 可以使用函数查询，并不一定是lambda
print(df.loc[lambda df_inner: (df_inner['bWendu'] <= 30) & (df_inner['yWendu'] >= 15)])
