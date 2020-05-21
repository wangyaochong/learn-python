import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./IMDB-Movie-Data.csv", encoding=u"gbk")

print(data['Rating'].mean())

print("导演数量", len(data['Director'].tolist()))

print("导演数量去重", len(data['Director'].unique()))

actor_list = data['Actors'].str.split(",").tolist()
print("before flatten", np.array(actor_list).flatten())
print("演员数量", len(np.array(actor_list).flatten()))
actor_list = [i for j in actor_list for i in j]  # 注意这种循环的写法
print("演员数量", len(actor_list))
print("演员数量去重", len(set(actor_list)))
