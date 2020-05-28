s = "abc\
sdf"
print(s)

s2 = """abc
efgg"""
print(s2)

print("s=" + s)
print("s=", s)

b = "hello %s" % "aaa"
b2 = "hello %s %s" % ("aaa", "bbb")
b3 = "hello%3.5s" % "aaaaaaa"  # 限制长度是3-5
f = "x=%.2f" % 123.333

print(b)
print(b2)
print(b3)
print(f)
print(f"格式化字符串{b}{f}")

print("格式化2={}".format(b))
print("*" * 20)
