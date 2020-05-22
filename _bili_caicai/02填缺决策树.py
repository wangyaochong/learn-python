from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boston = load_boston()

x, y = boston.data, boston.target
n_samples = x.shape[0]
n_features = x.shape[1]

# 制造缺失数据
rng = np.random.RandomState(0)
missing_rate = 0.5  # 缺失率是50%
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))

missing_features_loc = rng.randint(0, n_features, n_missing_samples)  # randint(start,end,count)
missing_samples_loc = rng.randint(0, n_samples, n_missing_samples)
origin = x
x_missing = x.copy()
y_missing = y.copy()
x_missing[missing_samples_loc, missing_features_loc] = np.nan
x_missing = pd.DataFrame(x_missing)

missing_count = x_missing.isnull().sum(axis=0)
temp = np.argsort(missing_count)  # 返回排序后的顺序对应的索引
sort_index = temp.values

for i in sort_index:
    df = x_missing.copy()
    fill_column = df.iloc[:, i]
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y)], axis=1)

    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

    # 造数
    y_train = fill_column[fill_column.notnull()]
    y_test = fill_column[fill_column.isnull()]
    x_train = df_0[y_train.index, :]
    x_test = df_0[y_test.index, :]

    # 计算
    rfc = RandomForestRegressor(n_estimators=10)
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)

    # 填值
    x_missing.loc[y_test.index, i] = y_predict
