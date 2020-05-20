import numpy as np

t1 = np.array([[1, 2, 3], [4, 5, 6]])
print(t1)

t2 = np.arange(12)
t2 = t2.reshape((3, 4))
print(t2)
print("转置", t2.T)
print("转置", t2.transpose())
print("转置", t2.swapaxes(1, 0))
print("转成一维数组最简单", t2.reshape(-1))
print("转成一维数组最简单2", t2.flatten())
print("转成一维数组", len(t2), t2.shape[0] * t2.shape[1])
count = t2.shape[0] * t2.shape[1]
print(t2.reshape((count,)))

t3 = np.arange(24)
t3 = t3.reshape((2, 3, 4))
print(t3)
print("转成一维数组", t3.reshape(-1))
print("转成一维数组", t3.flatten())
t3 = t3.reshape((4, 6))
print(t3)

print("加法", t3 + 1000)
print("加法", t3 + t3)
print("加法维度不同", t3 + np.array([1, 2, 3, 4, 5, 6]))
print("减法", t3 - t3)
print("减法维度不同", t3 - np.array([1, 2, 3, 4, 5, 6]))
print("乘法", t3 * 2)
print("除法", t3 / 2)
