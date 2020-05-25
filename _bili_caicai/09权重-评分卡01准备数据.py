import pandas as pd
from zipfile import ZipFile
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def fill_missing_with_rf(x, y, fill_col_name):
    df = x.copy()
    fill = df.loc[:, fill_col_name]
    df = pd.concat([df.loc[:, df.columns != fill_col_name], pd.DataFrame(y)], axis=1)
    y_train = fill[fill.notnull()]
    y_test = fill[fill.isnull()]
    x_train = df.iloc[y_train.index, :]
    x_test = df.iloc[y_test.index, :]
    from sklearn.ensemble import RandomForestRegressor as RFR  # 确保导库了
    rfr = RFR(n_estimators=10)
    rfr.fit(x_train, y_train)
    y_predict = rfr.predict(x_test)
    return y_predict


file = ZipFile('./rankingcard.zip')
f = file.open('rankingcard.csv')
data = pd.read_csv(f, index_col=0)
f.close()
file.close()

data.info()

data.drop_duplicates(inplace=True)  # 去除重复数据
data.index = range(data.shape[0])  # 恢复索引，这个很重要

# #######################去除异常数据#####################################
desc = data.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
data = data[data['age'] != 0]  # 去除年龄为0的列
data = data[data.loc[:, 'NumberOfTimes90DaysLate'] < 90]
print("第一种bool索引", data.loc[:, 'NumberOfTimes90DaysLate'] < 90)
desc2 = data.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
data.index = range(data.shape[0])  # 恢复索引，这个很重要


print("数据缺失比例", data.isnull().sum() / data.shape[0])
print("数据缺失比例", data.isnull().mean())

# #######################数据填充#####################################
data['NumberOfDependents'].fillna(data['NumberOfDependents'].mean(), inplace=True)
print("数据缺失比例NumberOfDependents", data.isnull().mean())

x = data.iloc[:, 1:]
y = data.iloc[:, 0]
result = fill_missing_with_rf(x, y, 'MonthlyIncome')
print("填充的MonthlyIncome", result)
x.loc[x.loc[:, 'MonthlyIncome'].isnull(), 'MonthlyIncome'] = result  # 使用随机森林填充缺失的MonthlyIncome
print(x.info())


# data = data[data['NumberOfTimes90DaysLate'] < 90]
# print("第二种bool索引", data['NumberOfTimes90DaysLate'] < 90)
data.index = range(data.shape[0])  # 恢复索引，这个很重要

# 由于样本中逾期的人数很少，但是这部分数据反而是最重要的，所以需要升采样
print("升采样前的标签数据", pd.Series(y).value_counts())  # 逾期和非逾期的人数相差很大
sm = SMOTE(random_state=0)
x, y = sm.fit_sample(x, y)  # 返回上采样的数据
print(x.shape)
print("升采样后的标签数据", pd.Series(y).value_counts())  # 逾期和非逾期的人数相同了

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_data = pd.concat([y_train, x_train], axis=1)
train_data.index = range(train_data.shape[0])  # 恢复索引
train_data.columns = data.columns

test_data = pd.concat([y_test, x_test], axis=1)
test_data.index = range(test_data.shape[0])  # 恢复索引
test_data.columns = data.columns

train_data.to_csv("./train_data.csv")
test_data.to_csv("./test_data.csv")
