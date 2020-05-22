from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

data1 = pd.read_csv("./Narrativedata.csv", index_col=0)

data1.info()  # 填补年龄
Age = data1.loc[:, "Age"].values.reshape(-1, 1)
imp_mean = SimpleImputer()
imp_median = SimpleImputer(strategy="median")  # sklearn当中特征矩阵必须是二维
imp_0 = SimpleImputer(strategy="constant", fill_value=0)  # 用0填补 imp_mean = imp_mean.fit_transform(Age)

imp_mean = imp_mean.fit_transform(Age)
imp_median = imp_median.fit_transform(Age)
imp_0 = imp_0.fit_transform(Age)

Embarked = data1.loc[:, "Embarked"].values.reshape(-1, 1)
imp_mode = SimpleImputer(strategy="most_frequent")
data1.loc[:, "Embarked"] = imp_mode.fit_transform(Embarked)
data1.info()

# 使用pandas填缺
data = pd.read_csv("./Narrativedata.csv", index_col=0)
data.head()
data.loc[:, "Age"] = data.loc[:, "Age"].fillna(data.loc[:, "Age"].median())  # .fillna 在DataFrame里面直接进行填补
data.dropna(axis=0, inplace=True)
data.info()
