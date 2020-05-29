import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./ratings.csv")

df['pdate'] = pd.to_datetime(df['timestamp'], unit='s')
df_group = df.groupby([df['pdate'].dt.month, 'rating'])['userId'].agg(func_or_funcs=np.sum)
print(df_group.head(20))
print(df_group.unstack())
print(df_group.unstack().plot())
plt.show()
df_reset = df_group.reset_index()
print(df_reset)

df_pivot=df_reset.pivot('pdate','rating','userId')
print(df_pivot)
df_pivot.plot()
plt.show()
