import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 统计电影分类

data = pd.read_csv("./IMDB-Movie-Data.csv", encoding=u"gbk")

genre_map = {}
t_list = data['Genre'].str.split(",").tolist()
t_list = np.array(t_list).flatten()
genre = set([i for j in t_list for i in j])
print("总的分类", genre)

zero_df = pd.DataFrame(np.zeros((data.shape[0], len(genre))), columns=list(genre))
print(zero_df)
for i in range(len(t_list)):
    # 示例 zero_df.loc[0,["Sci-fi","Musical"]]=1
    zero_df.loc[i, t_list[i]] = 1
result = zero_df.sum(axis=0)
result = result.sort_values()
print(result)
print(type(result), result.index, result.values)

plt.figure(figsize=(12, 6))
plt.bar(result.index, result.values, width=0.5)
plt.xticks(list(result.index), rotation=45)
plt.show()
