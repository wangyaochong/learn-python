from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)

result = cross_val_score(regressor, boston.data, boston.target, cv=10,
                         scoring="neg_mean_squared_error"
                         )
print(result)
