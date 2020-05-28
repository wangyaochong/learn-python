d = {'a': 1, 'b': 2}
print(d, type(d))
print("d['a']的值等于{}".format(d['a']))

# 字典的值可以是任意对象，字典的key只能是不可变对象


d = dict([('name', '小米'), ('price', 123)])  # 可以使用双值序列创建dict
print(d, type(d), len(d), 'name' in d)  # in 可以判断dict中是否有某个键
print(d.get('name'))  # 可以使用get获取值，如果没有key不会报错
print(d.get('不存在的key', '默认值'))  # 可以使用get获取值，如果没有key不会报错

name = d.setdefault('name', '华为')
print(d, name)
size = d.setdefault('size', 'big')
print(d, size)

d2 = {'name3': '苹果', 'price': 4399}
d.update(d2)  # 如果有重复的属性，后面的会覆盖前面的
print("合并dict", d)

del d['name']
print("删除name", d)

d.pop('name3')
print("删除name3", d)
# d.pop('abcdd') # 如果删除一个不存在的key，会报错

print("打印keys")
for i in d.keys():
    print(i)
print("打印values")
for i in d.values():
    print(i)
print("打印items")
for k, v in d.items():
    print(k, v)
