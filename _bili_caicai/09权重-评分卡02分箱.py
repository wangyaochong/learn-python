import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy


def get_woe(num_bins):
    # 通过 num_bins 数据计算 woe
    columns = ["min", "max", "count_0", "count_1"]
    df = pd.DataFrame(num_bins, columns=columns)
    df["total"] = df.count_0 + df.count_1
    df["percentage"] = df.total / df.total.sum()
    df["bad_rate"] = df.count_1 / df.total
    df["good%"] = df.count_0 / df.count_0.sum()
    df["bad%"] = df.count_1 / df.count_1.sum()
    df["woe"] = np.log(df["good%"] / df["bad%"])
    return df


# 计算IV值
def get_iv(df):
    rate = df["good%"] - df["bad%"]
    iv = np.sum(rate * df.woe)
    return iv


def get_bin(num_bins_, n):
    while len(num_bins_) > n:
        pvs = []
        for i in range(len(num_bins_) - 1):
            x1 = num_bins_[i][2:]
            x2 = num_bins_[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            # chi2 = scipy.stats.chi2_contingency([x1,x2])[0]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins_[i:i + 2] = [(
            num_bins_[i][0],
            num_bins_[i + 1][1],
            num_bins_[i][2] + num_bins_[i + 1][2],
            num_bins_[i][3] + num_bins_[i + 1][3])]
    return num_bins_


def graphforbestbin(DF, X, Y, n=5, q=20, graph=True):
    """
   自动最优分箱函数，基于卡方检验的分箱
   参数：
   DF: 需要输入的数据
   X: 需要分箱的列名
   Y: 分箱数据对应的标签 Y 列名
   n: 保留分箱个数
   q: 初始分箱的个数
   graph: 是否要画出IV图像
   区间为前开后闭 (]
   """
    DF = DF[[X, Y]].copy()
    DF["qcut"], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates="drop")
    coount_y0 = DF.loc[DF[Y] == 0].groupby(by="qcut").count()[Y]
    coount_y1 = DF.loc[DF[Y] == 1].groupby(by="qcut").count()[Y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]
    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [(
                num_bins[0][0],
                num_bins[1][1],
                num_bins[0][2] + num_bins[1][2],
                num_bins[0][3] + num_bins[1][3])]

            continue

        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i - 1:i + 1] = [(
                    num_bins[i - 1][0],
                    num_bins[i][1],
                    num_bins[i - 1][2] + num_bins[i][2],
                    num_bins[i - 1][3] + num_bins[i][3])]

                break
        else:
            break

    IV = []
    axisx = []
    bins_df = pd.DataFrame()
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins) - 1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i + 1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins[i:i + 2] = [(
            num_bins[i][0],
            num_bins[i + 1][1],
            num_bins[i][2] + num_bins[i + 1][2],
            num_bins[i][3] + num_bins[i + 1][3])]
        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        IV.append(get_iv(bins_df))
    if graph:
        plt.figure()
        plt.title(X)
        plt.plot(axisx, IV)
        plt.xticks(axisx)
        plt.xlabel("number of box")
        plt.ylabel("IV")
        plt.show()
    return bins_df


def get_woe2(df, col, y, bins):
    df = df[[col, y]].copy()
    df["cut"] = pd.cut(df[col], bins)
    bins_df = df.groupby("cut")[y].value_counts().unstack()
    woe = bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))
    return woe


train_data = pd.read_csv("./train_data.csv", index_col=0)  # 如果没有数据，则需要执行准备数据的代码
test_data = pd.read_csv("./test_data.csv", index_col=0)  # 如果没有数据，则需要执行准备数据的代码
desc = train_data.describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

train_data['qcut'], up_down = pd.qcut(train_data['age'], retbins=True, q=20)
train_data.info()

count_y0 = train_data[train_data['SeriousDlqin2yrs'] == 0].groupby(by='qcut').count()['SeriousDlqin2yrs']
count_y1 = train_data[train_data['SeriousDlqin2yrs'] == 1].groupby(by='qcut').count()['SeriousDlqin2yrs']

num_bins = [*zip(up_down, up_down[1:], count_y0, count_y1)]

x_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# 确保每个箱中都有0和1
for i in range(20):
    # 如果第一个组没有包含正样本或负样本，向后合并
    if 0 in num_bins[0][2:]:
        num_bins[0:2] = [(
            num_bins[0][0],
            num_bins[1][1],
            num_bins[0][2] + num_bins[1][2],
            num_bins[0][3] + num_bins[1][3])]
        continue
    for i in range(len(num_bins)):
        if 0 in num_bins[i][2:]:
            num_bins[i - 1:i + 1] = [(
                num_bins[i - 1][0],
                num_bins[i][1],
                num_bins[i - 1][2] + num_bins[i][2], num_bins[i - 1][3] + num_bins[i][3])]
            break
    else:
        break

# num_bins_ = num_bins.copy()

#
# IV = []
# axisx = []
# while len(num_bins_) > 2:
#     pvs = []
#     # 获取 num_bins_两两之间的卡方检验的置信度（或卡方值）
#     for i in range(len(num_bins_) - 1):
#         x1 = num_bins_[i][2:]
#         x2 = num_bins_[i + 1][2:]
#         # 0 返回 chi2 值，1 返回 p 值。
#         pv = scipy.stats.chi2_contingency([x1, x2])[1]  # 卡方检验是检验两列数据的相似性
#         # chi2 = scipy.stats.chi2_contingency([x1,x2])[0]
#         pvs.append(pv)
#     # 通过 p 值进行处理。合并 p 值最大的两组
#     i = pvs.index(max(pvs))
#     num_bins_[i:i + 2] = [(
#         num_bins_[i][0],
#         num_bins_[i + 1][1],
#         num_bins_[i][2] + num_bins_[i + 1][2],
#         num_bins_[i][3] + num_bins_[i + 1][3])]
#     bins_df = get_woe(num_bins_)
#     axisx.append(len(num_bins_))
#     IV.append(get_iv(bins_df))
# plt.figure()
# plt.plot(axisx, IV)
# plt.xticks(axisx)
# plt.xlabel("number of box")
# plt.ylabel("IV")
# plt.show()

for i in train_data.columns[1:-1]:
    graphforbestbin(train_data, i, "SeriousDlqin2yrs", n=2, q=20, graph=True)

auto_col_bins = {"RevolvingUtilizationOfUnsecuredLines": 6,
                 "age": 5,
                 "DebtRatio": 4,
                 "MonthlyIncome": 3,
                 "NumberOfOpenCreditLinesAndLoans": 5}

# 不能使用自动分箱的变量，可以通过观察数据分布确定分箱区间
hand_bins = {"NumberOfTime30-59DaysPastDueNotWorse": [0, 1, 2, 13]
    , "NumberOfTimes90DaysLate": [0, 1, 2, 17]
    , "NumberRealEstateLoansOrLines": [0, 1, 2, 4, 54]
    , "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 2, 8]
    , "NumberOfDependents": [0, 1, 2, 3]}
hand_bins = {k: [-np.inf, *v[:-1], np.inf] for k, v in hand_bins.items()}

bins_of_col = {}
# 生成自动分箱的分箱区间和分箱后的 IV 值
for col in auto_col_bins:
    bins_df = graphforbestbin(train_data, col
                              , "SeriousDlqin2yrs"
                              , n=auto_col_bins[col]
                              # 使用字典的性质来取出每个特征所对应的箱的数量
                              , q=20
                              , graph=False)
    bins_list = sorted(set(bins_df["min"]).union(bins_df["max"]))
    # 保证区间覆盖使用 np.inf 替换最大值 -np.inf 替换最小值
    bins_list[0], bins_list[-1] = -np.inf, np.inf
    bins_of_col[col] = bins_list
bins_of_col.update(hand_bins)

data = train_data.copy()
# 函数pd.cut，可以根据已知的分箱间隔把数据分箱
# 参数为 pd.cut(数据，以列表表示的分箱间隔)
data = data[["age", "SeriousDlqin2yrs"]].copy()
data["cut"] = pd.cut(data["age"], [-np.inf, 48.49986200790144, 58.757170160044694, 64.0,
                                   74.0, np.inf])
# 将数据按分箱结果聚合，并取出其中的标签值
data.groupby("cut")["SeriousDlqin2yrs"].value_counts()
# 使用unstack()来将树状结构变成表状结构
data.groupby("cut")["SeriousDlqin2yrs"].value_counts().unstack()
bins_df = data.groupby("cut")["SeriousDlqin2yrs"].value_counts().unstack()
bins_df["woe"] = np.log((bins_df[0] / bins_df[0].sum()) / (bins_df[1] / bins_df[1].sum()))

woeall = {}
for col in bins_of_col:
    woeall[col] = get_woe2(train_data, col, "SeriousDlqin2yrs", bins_of_col[col])

# 不希望覆盖掉原本的数据，创建一个新的DataFrame，索引和原始数据model_data一模一样
model_woe = pd.DataFrame(index=train_data.index)  # 将原数据分箱后，按箱的结果把WOE结构用map函数映射到数据中
model_woe["age"] = pd.cut(train_data["age"], bins_of_col["age"]).map(woeall["age"])
# 对所有特征操作可以写成：
for col in bins_of_col:
    model_woe[col] = pd.cut(train_data[col], bins_of_col[col]).map(woeall[col])
# 将标签补充到数据中
model_woe["SeriousDlqin2yrs"] = train_data["SeriousDlqin2yrs"]  # 这就是我们的建模数据了
model_woe.head()

vali_woe = pd.DataFrame(index=test_data.index)
for col in bins_of_col:
    vali_woe[col] = pd.cut(test_data[col], bins_of_col[col]).map(woeall[col])
vali_woe["SeriousDlqin2yrs"] = test_data["SeriousDlqin2yrs"]
vali_X = vali_woe.iloc[:, :-1]
vali_y = vali_woe.iloc[:, -1]
X = model_woe.iloc[:, :-1]
y = model_woe.iloc[:, -1]

from sklearn.linear_model import LogisticRegression as LR

lr = LR().fit(X, y)
lr.score(vali_X, vali_y)
c_1 = np.linspace(0.01, 1, 20)
c_2 = np.linspace(0.01, 0.2, 20)
score = []
for i in c_2:
    lr = LR(solver='liblinear', C=i).fit(X, y)
    score.append(lr.score(vali_X, vali_y))
plt.figure()
plt.plot(c_2, score)
plt.show()
lr.n_iter_
score = []
for i in [1, 2, 3, 4, 5, 6]:
    lr = LR(solver='liblinear', C=0.025, max_iter=i).fit(X, y)
    score.append(lr.score(vali_X, vali_y))
plt.figure()
plt.plot([1, 2, 3, 4, 5, 6], score)
plt.show()

import scikitplot as skplt

vali_proba_df = pd.DataFrame(lr.predict_proba(vali_X))
skplt.metrics.plot_roc(vali_y, vali_proba_df,  # 这里有个ROC曲线
                       plot_micro=False, figsize=(6, 6),
                       plot_macro=False)
plt.show()

B = 20 / np.log(2)
A = 600 + B * np.log(1 / 60)
base_score = A - B * lr.intercept_
score_age = woeall["age"] * (-B * lr.coef_[0][1])

file = './score.csv'
with open(file, "w") as fdata:
    fdata.write("base_score,{}\n".format(base_score))
for i, col in enumerate(X.columns):
    score = woeall[col] * (-B * lr.coef_[0][i])
    score.name = "Score"
    score.index.name = col
    score.to_csv(file, header=True, mode="a")
