def say_hello(name):
    """
    这是一个函数，用于打印name
    :param name: 输入的名字
    :return:  没有返回值
    """
    print("hello", name)


def say_hello2(prefix, *names):  # *只能接受位置参数，而不能接受关键词参数
    for i in names:
        print(prefix, i)


def say_hello3(prefix, **kwargs):  # **可以接受关键词参数，使用字典保存参数名和参数值
    for key, value in kwargs.items():
        print(prefix, key, value)


def say_hello4(prefix, name):
    print(prefix, name)


say_hello('wyc')
say_hello2('hello', 'b', 'c')
say_hello3('hello', arg1='b', arg2='c')
args = ['hello', 'lisi']
args2 = ('hello2', 'lisi2')
say_hello4(*args)  # 传参时可以使用*进行参数解包
say_hello4(*args2)  # 传参时可以使用*进行参数解包
args3 = {'prefix': 'hello3', 'name': 'lisi3'}
say_hello4(**args3)  # 也可以使用**对字典进行解包

print(help(print))  # 可以使用help函数查看帮助
print(help(say_hello))  # 可以使用help函数查看帮助

# 可以使用global关键字修改全局变量
