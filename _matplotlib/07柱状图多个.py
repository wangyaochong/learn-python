from matplotlib import pyplot as plt
import matplotlib
import random

font = {"family": "KaiTi"}
matplotlib.rc("font", **font)

a = ["猩球崛起3：终极之战", "敦刻尔克", "蜘蛛侠：英雄归来", "战狼2"]
b_16 = [15746, 312, 4497, 319]
b_15 = [12357, 156, 2045, 168]
b_14 = [2358, 399, 2358, 362]

width = 0.2

x14 = list(range(len(a)))
x15 = [i + width for i in x14]
x16 = [i + width * 2 for i in x14]

plt.bar(x14, b_14, width=width)
plt.bar(x15, b_15, width=width)
plt.bar(x16, b_16, width=width)
plt.xticks(range(len(a)), a, rotation=45)
plt.show()
