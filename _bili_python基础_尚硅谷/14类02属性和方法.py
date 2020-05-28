class Person:
    a = 10  # 公共属性

    # 可以在__init__方法定义类属性
    def __init__(self, name='default', age=0):
        # 隐藏的属性，外部无法访问(其实还是可以通过 obj._Person__age 访问)
        self.name = name
        self.__age = age

    def say_hello(self):
        print("你好", self.name, "我的年龄是", self.__age)

    # 这个注解可以在使用obj.name时调用name()方法
    # @property
    # def name(self):
    #     return self.name
    #
    # @name.setter
    # def name(self, name):
    #     self.name = name

    @classmethod
    def cla_method(cls):
        print(f'this is {cls}')

    @staticmethod
    def sta_method():
        print("这是一个静态方法")


p1 = Person(name='猪八戒', age=100)
p2 = Person()
print(p1.a)
print(p2.a)
p1.say_hello()
#
print(p1.name, p1._Person__age)
Person.cla_method()
Person.sta_method()
