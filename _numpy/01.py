import numpy as np
import random

t1 = np.array([1, 2, 3, 4])
print(t1)
print(type(t1))

t2 = np.array(range(10))
print(t2)
print(type(t2))

t3=np.arange(10)
print(t3)
print(type(t3))

t4 = np.array(range(10),dtype=float)
print(t4)
print(type(t4),t4.dtype)

t5 = t4.astype(int)
print(t5)
print(type(t5),t5.dtype)

t6 = np.array([random.random() for i in range(10)])
print(t6)
print(type(t6),t6.dtype)

t7=np.round(t6,2)
print(t7)