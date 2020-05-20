from matplotlib import pyplot as plt
import matplotlib
import random

font = {"weight": "bold", "size": "20", "family": "KaiTi"}
matplotlib.rc("font", **font)
# matplotlib.rc("font", size="20")

y1 = [1, 0, 1, 1, 2, 4]
y2 = [1, 0, 0, 0, 1, 3]
x = range(11, 17)

plt.figure(figsize=(20, 8), dpi=60)
plt.grid(alpha=0.5)
plt.plot(x, y1, label="自己",linestyle=":",color="r")
plt.plot(x, y2, label="测试",linestyle="--",color="b")
plt.legend(loc="upper left")
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("10点-12点温度变化")
plt.show()
