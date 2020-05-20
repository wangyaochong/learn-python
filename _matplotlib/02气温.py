from matplotlib import pyplot as plt
import matplotlib
import random

font = {"weight": "bold", "size": "20", "family": "KaiTi"}
matplotlib.rc("font", **font)
# matplotlib.rc("font", size="20")

# 绘制气温
x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

_x = list(x)

_xtick_labels = ["10点{}分".format(i) for i in range(60)]
_xtick_labels += ["11点{}分".format(i) for i in range(60)]
plt.figure(figsize=(20, 8), dpi=60)
plt.xticks(_x[::5], _xtick_labels[::5], rotation=20)
plt.grid(alpha=0.5)
plt.plot(x, y)
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("10点-12点温度变化")
plt.show()
