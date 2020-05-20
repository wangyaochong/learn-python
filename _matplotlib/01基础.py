from matplotlib import pyplot as plt

# 绘制折线
x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 24, 22, 18, 15]

plt.figure(figsize=(15, 15), dpi=50)
# 每两个取一个坐标
plt.xticks(x[::2])
plt.yticks(range(min(y), max(y) + 1))
plt.plot(x, y)
plt.savefig("./test.png")
plt.show()
