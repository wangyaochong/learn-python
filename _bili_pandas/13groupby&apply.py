import pandas as pd


def ratings_norm(df):
    min_value = df['rating'].min()
    max_value = df['rating'].max()
    df['rating_norm'] = df['rating'].apply(lambda x: (x - min_value) / (max_value - min_value))
    return df


def getWenduTopN(df, n):
    return df.sort_values(by='bWendu')[['ymd', 'bWendu']][-n:]


ratings = pd.read_csv("./ratings.csv")
ratings = ratings.groupby('userId').apply(ratings_norm)
print(ratings.head())

df = pd.read_csv("./beijing_tianqi_2018.csv")
df.dropna(axis='columns', how='all', inplace=True)  # axis可以使用字符串
df.dropna(axis='index', how='all', inplace=True)  # axis可以使用字符串

df['month'] = df['ymd'].str[:7]

print(df.groupby('month').apply(getWenduTopN, n=2).head())
