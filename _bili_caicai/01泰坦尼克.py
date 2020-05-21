import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("./titanic/train.csv")
data.info()

data.drop(['Cabin','Name','Ticket'],inplace=True)