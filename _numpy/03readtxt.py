import numpy as np

data = np.loadtxt("./USvideos.csv", skiprows=1, delimiter=",", dtype=float)
print(data)

print("取第二行", data[1, :])
print("取第三行", data[2])
print("取多行", data[2:])
print("取分散的多行", data[[1, 3, 4, 5]])

print("取第三列", data[:, 2])
print("取连续多列", data[:, 2:])
print("取不连续多列", data[:, [1, 3]])

print("取值", data[1, 2])

print("取块", data[2:5, 1:4])

print("取独立的点", data[[0, 1], [0, 1]])  # 两个数组分别是x和y坐标

data[0, 0] = 1
data[0, 1] = np.nan
print("赋值", data[0, 0], data[0, 1])
data[data > 100] = 100
print("条件赋值", data)
print("条件赋值2", np.where(data < 100, 100, 0))
print("条件赋值3", data.clip(5, 90))  # 小于5替换成5，大于90替换成90

print("计算nan的个数", np.count_nonzero(data != data))
print("计算nan的个数", np.count_nonzero(np.isnan(data)))  # nan和任何值计算都是nan
# data[data[:5] < 100] = 99
