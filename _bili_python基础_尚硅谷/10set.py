a = {1, 3, 2, 9, 4, 5}
print(a, type(a))

a.add(7)
print(a)
b = {22, 33, 3}
a.update(b)
print(a)

print("取交集", a & b)
print("取并集", a | b)
print("差集", a - b)
print("异或集", a ^ b)
print("检查是否是子集", {1, 2} <= {1, 2})
print("检查是否是真子集", {1, 2} < {1, 2})
print("检查是否是真子集", {1, 2} < {1, 2, 3})
