import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from sklearn.model_selection import cross_val_score as CVS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing as fch
from sklearn.ensemble import RandomForestRegressor as RFR
from time import time

X = np.arange(9).reshape(3, 3)
poly = PolynomialFeatures(degree=5).fit(X)
feature_names_temp = poly.get_feature_names()
housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
housevalue.feature_names
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目", "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]
poly = PolynomialFeatures(degree=2).fit(X, y)
feature_names = poly.get_feature_names(X.columns)
X_ = poly.transform(X)
reg = LinearRegression().fit(X_, y)
coeff = pd.DataFrame([poly.get_feature_names(X.columns), reg.coef_.tolist()]).T
coeff.columns = ["feature", "coef"]

poly = PolynomialFeatures(degree=4).fit(X, y)
X_ = poly.transform(X)
reg = LinearRegression().fit(X, y)
reg.score(X, y)

time0 = time()
reg_ = LinearRegression().fit(X_, y)
print("R2:{}".format(reg_.score(X_, y)))
print("time:{}".format(time() - time0))
time0 = time()
print("R2:{}".format(RFR(n_estimators=20).fit(X, y).score(X, y)))
print("time:{}".format(time() - time0))
