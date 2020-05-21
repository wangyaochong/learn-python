import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("./IMDB-Movie-Data.csv", encoding=u"gbk")
# 电影时长
print(data.head(10))
print(data.columns)
runtime_values = data['Runtime (Minutes)'].values
print(runtime_values)

max_time = runtime_values.max()
min_time = runtime_values.min()
range_time = max_time - min_time
bin_count = range_time // 6  # 能够整除6
plt.hist(runtime_values, bin_count)
plt.xticks(list(range(min_time, max_time + 6, 6)))
plt.show()
