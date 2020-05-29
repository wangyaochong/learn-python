import pandas as pd
import numpy as np


def get_sum_value(x):
    temp = x.iloc[0]
    for i in range(1, x.shape[0]):
        temp += x.iloc[i]
    return temp


df = pd.DataFrame(np.arange(12).reshape(3, 4))
print(df)
df['sum'] = df.apply(get_sum_value, axis=1)
print(df)
