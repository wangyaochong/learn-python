t1 = (1, 2, 3, 4)
print(t1)

t2 = 3, 4, 5, 6  # 创建元组时可以省略括号，如果要创建一个元素的元组，则需要使用逗号
print(t2)

t3 = 3,
print(t3)

a, b, c, d = t2
print(a, b, c, d)  # 使用元组可以解构

a, b = b, a
print(a, b)  # 使用元组交换值

a, b, *c = t2  # 可以在解构时使用*
print(a, b, c)

a, *b, c = t2  # 可以在解构时使用*，但只能使用一个*
print(a, b, c)
