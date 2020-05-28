class Animal:
    def __init__(self, name):
        self._name = name

    def run(self):
        print("动物会跑")

    def sleep(self):
        print("动物会睡觉")


class Dog(Animal):  # 实现继承的方式，python支持多重继承
    def __init__(self, name):
        super().__init__(name)  # 执行父类的初始化方法

    def bark(self):
        print("狗会叫")

    def run(self):
        print("狗会跑")


dog = Dog('dogName')
dog.run()

print(isinstance(dog, Dog))
print(isinstance(dog, Animal))
print(issubclass(Dog, Animal))
