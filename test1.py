import time
from bitarray import bitarray

# 两个测试用整数
a = 123456789
b = 123456789  # 可改为不同值测试性能变化

# 方法一：直接 ==
start1 = time.perf_counter()
for _ in range(10**6):
    result = (a == b)
end1 = time.perf_counter()

# 方法二：使用 bitarray 异或比较
def int_to_bitarray(x, width=64):
    return bitarray(bin(x)[2:].zfill(width))

ba = int_to_bitarray(a)
bb = int_to_bitarray(b)

start2 = time.perf_counter()
for _ in range(10**6):
    diff = ba ^ bb
    result = not diff.any()  # 如果全部为0，说明相等
end2 = time.perf_counter()

print(f"方法一（==）耗时: {end1 - start1:.6f} 秒")
print(f"方法二（bitarray异或）耗时: {end2 - start2:.6f} 秒")
