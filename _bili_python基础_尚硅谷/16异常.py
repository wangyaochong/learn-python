class MyException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


print("hello")
try:
    # raise Exception('exception message')
    raise MyException('exception my ex')
    print(c)
    print(10 / 0)
except ZeroDivisionError:
    print(f"出异常了异常类型={ZeroDivisionError}")
except NameError:
    print(f"异常类型={NameError}")
except Exception as e:
    print("异常出现，属于其他异常", e)
else:
    print("程序正确运行")
finally:
    print("finally 被调用")
print("程序结束")
