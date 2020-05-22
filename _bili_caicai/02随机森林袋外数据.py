from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

wine = load_wine()

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0, oob_score=True)
dtc.fit(x_train, y_train)
rfc.fit(x_train, y_train)

print("袋外数据", rfc.oob_score_)

rfc = RandomForestClassifier(n_estimators=25)
rfc.fit(x_train, y_train)
score = rfc.score(x_test, y_test)
fi = rfc.feature_importances_
result_apply = rfc.apply(x_test)
result_predict = rfc.predict(x_test)
result_proba = rfc.predict_proba(x_test)
