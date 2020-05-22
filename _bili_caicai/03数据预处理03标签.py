from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("./Narrativedata.csv", index_col=0)
data.head()
data.loc[:, "Age"] = data.loc[:, "Age"].fillna(data.loc[:, "Age"].median())  # .fillna 在DataFrame里面直接进行填补
data.dropna(axis=0, inplace=True)
data.info()

y = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y)
label = le.transform(y)
print("le.classes_", le.classes_)
print("label", label)
data.iloc[:, -1] = le.fit_transform(y)
le.inverse_transform(label)

data_ = data.copy()
data_.head()
categories = OrdinalEncoder().fit(data_.iloc[:, 1:-1]).categories_
data_.iloc[:, 1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:, 1:-1])
data_.head()

# 使用独热编码
X = data.iloc[:, 1:-1]
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()
feature_names = enc.get_feature_names()
# axis=1,表示跨行进行合并，也就是将量表左右相连，如果是axis=0，就是将量表上下相连
# newdata = pd.concat([data, pd.DataFrame(result)], axis=1)
newdata = data.join(pd.DataFrame(result))
newdata.head()
newdata.drop(["Sex", "Embarked"], axis=1, inplace=True)
newdata.columns = ["Age", "Survived", "Female", "Male", "Embarked_C", "Embarked_Q", "Embarked_S"]
newdata.head()
