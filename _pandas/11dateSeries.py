import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

date_index = pd.date_range(start='20171230', end='20180131', freq='D')
print(date_index)
date_index = pd.date_range(start='20171230', periods=10, freq='D')
print(date_index)
date_index = pd.date_range(start='20171230', periods=10, freq='M')
print(date_index)
date_index = pd.date_range(start='20171230 10:10:30', periods=10, freq='H')
print(date_index)

result = pd.DataFrame(np.arange(10), index=date_index)
print(result)
print(result.resample("D").count())
