import pandas as pd

df_train = pd.read_csv("./titanic_train.csv")
df_train.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

print(pd.get_dummies(df_train['Sex']).head())

onehot_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

df_coded = pd.get_dummies(df_train, columns=onehot_columns, prefix=onehot_columns, dummy_na=True
                          # , drop_first=True #可以删除第一个编码（互斥关系）
                          )
print(df_coded.head())
