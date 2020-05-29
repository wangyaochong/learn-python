import pandas as pd
import numpy as np

s1 = pd.Series([1, 2, 3, 4], index=list('abcd'))
print(s1)
print(f'索引={s1.index}')
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(s2)
print('获取数据', s1['a'])
print('获取数据', s1[['a', 'b']])
