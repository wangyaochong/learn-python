import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 统计电影分类

data = pd.read_csv("./IMDB-Movie-Data.csv", encoding=u"gbk")

genre_map = {}
t_list = data['Genre'].str.split(",").tolist()
t_list = np.array(t_list).flatten()
t_list = [i for j in t_list for i in j]
print("总的分类", set(t_list))
print(t_list)
for i in t_list:
    if i not in genre_map:
        genre_map[i] = 0
    genre_map[i] = genre_map[i] + 1
print(genre_map.keys())
print(genre_map.values())
plt.figure(figsize=(12, 6))
plt.bar(genre_map.keys(), genre_map.values(), width=0.5)
plt.xticks(list(genre_map.keys()), rotation=45)
plt.show()
