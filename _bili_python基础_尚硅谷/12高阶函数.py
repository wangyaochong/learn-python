def apply(list_input, fun):
    for i in range(len(list_input)):
        list_input[i] = fun(list_input[i])


def mul2(param):
    return param * 2


l1 = [1, 2, 3, 4]

apply(l1, mul2)
print(l1)

apply(l1, lambda x: x * 3)
print(l1)

print(*filter(lambda x: x > 2, l1))
print(*map(lambda x: x + 1.1, l1))

l1.sort(key=lambda x: -x)
print(l1)
print(sorted(l1, key=lambda x: x))
