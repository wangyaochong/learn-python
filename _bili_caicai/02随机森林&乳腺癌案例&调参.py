from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()

# 先调参n_estimators，这个参数是最重要的
scorel = []
r = range(35, 45)
for i in r:
    rfc = RandomForestClassifier(n_estimators=i, n_jobs=8, random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    scorel.append(score)
print(max(scorel), list(r)[(scorel.index(max(scorel)))])
plt.figure(figsize=[20, 5])
plt.plot(r, scorel)
plt.show()

# param_grid = {'max_depth': np.arange(1, 20, 1)}
# rfc = RandomForestClassifier(n_estimators=39, random_state=90)
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(data.data, data.target)
# best_param = GS.best_params_
# best_core = GS.best_score_  # 使用max_depth参数导致分数下降了，所以这个参数不调

param_grid = {'max_features': np.arange(5, 30, 1)}
rfc = RandomForestClassifier(n_estimators=39, random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)
best_param = GS.best_params_
best_core = GS.best_score_  # 使用max_features参数导致分数下降了，所以这个参数不调
