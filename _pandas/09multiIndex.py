import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

tmp = pd.DataFrame(np.arange(12).reshape((3, 4)), index=list('abc'), columns=list('ABCD'))
print(tmp)
print("reindex", tmp.reindex(list('ab')))
print("set_index", tmp.set_index('A'))
print("set_index", tmp.set_index('A', drop=False))
print("set_multi_index", tmp.set_index(['A', 'B'], drop=False))

a = pd.DataFrame(
    {'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'], 'd': list("hjklmno")})

print(a)

b = a.set_index(['c', 'd'])
c = b['a']
print(c)
print("取一个", c['one']['h'])
print("取一列", c['one'][:])
print()

d = a.set_index(["d", "c"])['a']
print("d=", d)
print("d.swaplevel=", d.swaplevel())  # 使用swaplevel来指定取数的索引位置
print(d.swaplevel()["one"])

f = a.set_index(["d", "c"])
print("f=", f)
print("f.swaplevel=", f.swaplevel())
print("f.swaplevel().loc=", f.swaplevel().loc["one"].loc["h"])
print("f.swaplevel().loc=")
