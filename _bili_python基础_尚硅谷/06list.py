list1 = [1, 1, 2, 3, 4, 5]
print(list1)
print(type(list1))

list2 = list([1, 2, 3])
print(type(list2))

print(list1[:-1])

print(list1[::2])
print(list1[::-2])

print(list1 + list2)
print(list1 * 2)
print(1 in list1)
print(1 not in list1)
print(max(list1))
print(min(list1))
print(list1.count(1))
print(list1.index(1))

# 如果要修改list，可以通过索引或切片，使用切片时，需要使用列表作为参数，且参数的长度和切片的长度相同
list1[1] = 333
list1[3:5] = [555, 666]
print(list1)

list1[0:0] = [0]  # 插入元素
print(list1)

# 删除元素
del list1[-1]
print("删除最后一个元素", list1)

# list是可变序列

# str、tuple是不可变序列

list1.append(11111)
list1.insert(0, '0000')
print(list1)

list1.extend([2222, 3333, 4444])
print('扩展list', list1)

t = list1.pop(0)
print("删除第一个元素", list1)
t1 = list1.pop()
print("删除最后一个元素", list1)

list1.remove(2222)
print("删除指定值的元素", list1)

list1.reverse()
print("反转列表", list1)
list1.sort()
print("排序列表", list1)

i = 0
while i < len(list1):
    print(list1[i])
    i += 1
for x in list1:
    print(x)

for index, x in enumerate(list1):
    print(index, x, list1[index])
