class MyClass:
    pass


print(MyClass)
mc = MyClass()
print(isinstance(mc, MyClass))  # 判断一个对象是否是一个类的实例
mc.test = 'test'
print(mc.test)
