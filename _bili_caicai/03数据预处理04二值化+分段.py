from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer

data = pd.read_csv("./Narrativedata.csv", index_col=0)
data.head()
data.loc[:, "Age"] = data.loc[:, "Age"].fillna(data.loc[:, "Age"].median())  # .fillna 在DataFrame里面直接进行填补
data.dropna(axis=0, inplace=True)
data.info()

# 对age列二值化
age = data.iloc[:, 0].values.reshape(-1, 1)
age_after_process = Binarizer(threshold=30).fit_transform(age)
to_print = pd.concat([pd.DataFrame(age), pd.DataFrame(age_after_process)], axis=1)
to_print.columns = (['原始', '转化后'])
print(to_print)

# est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
# age_after_process2 = est.fit_transform(age)

est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
age_after_process2 = est.fit_transform(age).toarray()


to_print = pd.concat([pd.DataFrame(age), pd.DataFrame(age_after_process2)], axis=1)
# to_print.columns = (['原始', '转化后'])
print(to_print)
