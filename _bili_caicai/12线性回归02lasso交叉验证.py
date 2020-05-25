import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目", "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]
X.head()
Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])


alpharange = np.logspace(-10, -2, 200, base=10)

lasso_ = LassoCV(alphas=alpharange  # 自行输入的alpha的取值范围
                 , cv=5  # 交叉验证的折数
                 ).fit(Xtrain, Ytrain)
alpha = lasso_.alpha_
score = lasso_.score(Xtest, Ytest)

reg = LinearRegression().fit(Xtrain, Ytrain)
reg_score = reg.score(Xtest, Ytest)

ls_ = LassoCV(eps=0.00001, n_alphas=300, cv=5).fit(Xtrain, Ytrain)
ls_alpha = ls_.alpha_
ls_shape = ls_.alphas_.shape
ls_.score(Xtest, Ytest)
ls_coef = ls_.coef_
