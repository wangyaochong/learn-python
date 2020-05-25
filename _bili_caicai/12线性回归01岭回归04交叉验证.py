import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt

housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目", "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]

Ridge_ = RidgeCV(
    alphas=np.arange(1, 1001, 100)  # ,scoring="neg_mean_squared_error"
    , store_cv_values=True  # ,cv=5
).fit(X, y)

scores = Ridge_.score(X, y)
shape = Ridge_.cv_values_.shape

mean = Ridge_.cv_values_.mean(axis=0)
best_alpha = Ridge_.alpha_
