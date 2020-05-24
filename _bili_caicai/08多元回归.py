from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as LR

iris = load_iris()

for multi_class in ('multinomial',  # 一般优先选择multinomial
                    'ovr'):
    lr = LR(solver='sag', max_iter=100000, random_state=0, multi_class=multi_class)
    lr.fit(iris.data, iris.target)
    print("train score: %.3f (%s)" % (lr.score(iris.data, iris.target), multi_class))
