import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./books.csv", encoding="gbk")
data = data[pd.notnull(data['original_publication_year'])]
print(data.info())
print(data.head(2))
print("每年数量", data.groupby(by="original_publication_year").count()['id'])

result = data.groupby(by='original_publication_year').mean()['average_rating']
print("平均打分1", result)

# 以下是不使用average_rating列，手动进行平均分计算（会有误差）
count = data[['original_publication_year', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']].groupby(
    by="original_publication_year").sum()
print(count)
print(count.T)
count_t = count.T
count_t2 = count.T.copy()
count_t.loc['ratings_2'] = count_t.loc['ratings_2'] * 2
count_t.loc['ratings_3'] = count_t.loc['ratings_3'] * 3
count_t.loc['ratings_4'] = count_t.loc['ratings_4'] * 4
count_t.loc['ratings_5'] = count_t.loc['ratings_5'] * 5
print(count_t)
print(count_t2)
print("平均打分2", count_t.sum() / count_t2.sum())
print(type(count))
