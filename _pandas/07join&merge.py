import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# join 是按照行索引进行拼接
d1 = pd.DataFrame(np.ones((2, 4)), index=['A', 'B'], columns=list("abcd"))
d2 = pd.DataFrame(np.ones((3, 3)), index=['A', 'B', 'C'], columns=list("xyz"))
d3 = pd.DataFrame(np.zeros((2, 4)), columns=list("axyz"))
print("join", d1.join(d2))
print("join", d2.join(d1))

# merge 按照指定方式和并，类似join
print("merge1", d1.merge(d3, on="a"))
d3.iloc[0, 0] = 1
print(d3)
print("merge2", d1.merge(d3, on="a"))
print("merge outer", d1.merge(d3, on="a", how="outer"))
print("merge left outer", d1.merge(d3, on="a", how="left"))
print("merge right outer", d1.merge(d3, on="a", how="right"))
