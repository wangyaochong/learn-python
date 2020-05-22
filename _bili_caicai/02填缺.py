from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boston = load_boston()

x, y = boston.data, boston.target
n_samples = x.shape[0]
n_features = x.shape[1]

# 制造缺失数据
rng = np.random.RandomState(0)
missing_rate = 0.5  # 缺失率是50%
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))

missing_features_loc = rng.randint(0, n_features, n_missing_samples)  # randint(start,end,count)
missing_samples_loc = rng.randint(0, n_samples, n_missing_samples)

x_missing = x.copy()
y_missing = y.copy()
x_missing[missing_samples_loc, missing_features_loc] = np.nan
x_missing = pd.DataFrame(x_missing)

# imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean") #可以使用不同的填充策略
imp_mean = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
x_missing_mean = imp_mean.fit_transform(x_missing)
x_missing_mean = pd.DataFrame(x_missing_mean)
print("都不为空，和为0", x_missing_mean.isnull().sum())
