import numpy as np

a = np.arange(24).reshape(4, 6)
b = np.arange(24).reshape(4, 6)
b = b + 100

print(a, b)

print(np.vstack((a, b)))
print(np.hstack((a, b)))

a[[1, 2], :] = a[[2, 1], :]
print("交换行", a)

b[:, [1, 2]] = b[:, [2, 1]]
print("交换列", b)

print("全一", np.ones((2, 3)))
print("全零", np.zeros((2, 3)))
