import pandas as pd
import numpy as np


# map方法只能用于Series，map可以传入一个dict或者一个函数
# apply方法可以用于处理Series的值或者DataFrame的某列
# applymap 只能用于DataFrame，用于处理DataFrame的每个元素

def my_fun(x):
    print(x)
    print('*' * 100)


stocks = pd.read_excel("./互联网公司股票.xlsx")
print(stocks['公司'].unique())

dict_company_names = {
    'bidu': "百度",
    'baba': '阿里巴巴',
    'iq': '爱奇艺',
    'jd': '京东'
}

stocks['中文名'] = stocks['公司'].str.lower().map(dict_company_names)
stocks['中文名2'] = stocks.apply(lambda x: dict_company_names[x['公司'].lower()], axis=1)
print(stocks)

columns_process = ['收盘', '开盘', '高', '低', '交易量']
print(stocks[columns_process].applymap(lambda x: int(x)))
