# 以下是一个日志装饰器
def log(fun):  # 装包

    def new_fun(*args, **kwargs):
        print("开始执行")
        result = fun(*args, **kwargs)  # 下面的*号是解包
        print("结束执行")
        return result

    return new_fun


def sum_ab(a, b):
    return a + b


@log  # 直接通过注解使用装饰器，装饰器可以嵌套，执行顺序由内到外
def sub_ab(a, b):
    return a - b


sum_ab_log = log(sum_ab)
print(sum_ab_log(2, 3))

print(sub_ab(1, 3))
