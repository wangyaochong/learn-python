import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./movies.csv", encoding='gbk')
print(df.head())

df['genre'] = df['genres'].map(lambda x: x.split("|"))
df_new = df.explode('genre')
print(df_new.head())

print('统计每个分类有多少部电影', df_new['genre'].value_counts())

df_new['genre'].value_counts().plot.bar()
plt.show()