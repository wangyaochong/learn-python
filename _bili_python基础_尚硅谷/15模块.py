from _bili_python基础_尚硅谷.my_package.util import add
from _bili_python基础_尚硅谷.my_package.util import add as add_name
import _bili_python基础_尚硅谷.my_package.util as util

print("使用模块的方法", add(1, 3))
print("使用模块的方法别名", add_name(2, 3))

print("使用模块的变量", util.one, util.zero)
