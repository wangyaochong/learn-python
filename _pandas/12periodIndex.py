import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("./BeijingPM20100101_20151231.csv")
period = pd.PeriodIndex(year=data['year'], month=data['month'], day=data['day'], hour=data['hour'], freq="H")
print(type(period))

data['datetime'] = period
print(data.head())
data.set_index("datetime", inplace=True)

data = data.resample("D").mean()
# data = data[pd.notnull(data['PM_US Post'])]
data = data['PM_US Post'].dropna()
print(data)
x = data.index
x = [i.strftime("%Y%m%d") for i in x]
y = data.values
plt.plot(range(len(x)), y)
plt.xticks(range(0, len(x), 5), list(x)[::5], rotation=45)
plt.show()
