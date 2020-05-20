from matplotlib import pyplot as plt
import matplotlib
import random

font = {"weight": "bold", "family": "KaiTi"}
matplotlib.rc("font", **font)
a = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17, 18, 21, 16, 17, 20, 14, 15, 15, 15, 19, 21, 22, 22, 22,
     23]
b = [26, 26, 28, 19, 21, 17, 16, 19, 18, 20, 20, 19, 22, 23, 17, 20, 21, 20, 22, 15, 11, 15, 5, 13, 17, 10, 11, 13, 12,
     13, 6]

x1 = range(1, 1 + len(a))
x2 = range(51, 51 + len(b))

plt.scatter(x1, a, label="3月份")
plt.scatter(x2, b, label="10月份")

xtick = ["3月{}日".format(i) for i in x1]
xtick += ["10月{}日".format(i - 50) for i in x2]
x = list(x1) + list(x2)
plt.xticks(x[::3], xtick[::3], rotation=45)
plt.legend()
plt.show()
