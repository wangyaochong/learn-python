import pandas as pd
import numpy as np

df = pd.read_excel('./crazyant_blog_articles_source.xlsx')
print(df.head())

user_names = list('abcdef')
total_row_count = df.shape[0]
split_size = total_row_count // len(user_names)

if total_row_count % len(user_names) != 0:
    split_size += 1

df_subs = []
for idx, name in enumerate(user_names):
    begin = idx * split_size
    end = begin + split_size
    df_sub = df.iloc[begin:end]
    df_subs.append((idx, name, df_sub))

print(df_subs)
